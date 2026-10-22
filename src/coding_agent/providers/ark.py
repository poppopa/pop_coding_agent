"""Ark Responses 适配：终态决定调用是否可执行，供应商输出原样回放。"""
from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Mapping
from copy import deepcopy
from email.utils import parsedate_to_datetime
import inspect
import json
import math
import os
import time
from typing import Any, TypeVar
from uuid import uuid4

import httpx
from volcenginesdkarkruntime import AsyncArk
from volcenginesdkarkruntime._exceptions import ArkAPIConnectionError

from coding_agent.config import AgentConfig
from coding_agent.domain import ModelFrame, ModelRequest, ToolCall

_T = TypeVar('_T')
_MISSING = object()
_MAX_RETRIES = 2
_MAX_RETRY_AFTER = 30.0


class _ProtocolError(ValueError):
    """只携带固定诊断，避免供应商异常中的请求头或凭证进入事件。"""


def _field(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _plain(value: Any) -> Any:
    """完整保留 SDK 的额外字段；不按已知 output 类型重新拼装记录。"""
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    for method in ('model_dump', 'dict'):
        dump = getattr(value, method, None)
        if callable(dump):
            return _plain(dump())
    # 注入的 SDK 替身也可用简单属性对象表达事件与输出项。
    if hasattr(value, '__dict__'):
        return _plain(vars(value))
    raise _ProtocolError('供应商输出不是可保存的数据')


def _record(value: Any) -> dict[str, Any]:
    result = _plain(value)
    if not isinstance(result, dict):
        raise _ProtocolError('供应商输出项必须是对象')
    return result


class ArkProvider:
    """客户端可注入；只有自建客户端时读取 ARK_API_KEY。"""

    def __init__(
        self,
        config: AgentConfig | None = None,
        client: Any = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.config = config if config is not None else AgentConfig(model=model or '')
        self.model = model if model is not None else self.config.model
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError('必须配置 Ark 模型标识')
        if self.model == '__unconfigured__':
            raise ValueError('离线诊断模型标记不能用于 Ark 请求')
        if client is None:
            key = api_key if api_key is not None else os.environ.get('ARK_API_KEY')
            if not isinstance(key, str) or not key.strip():
                raise ValueError('缺少 ARK_API_KEY，无法初始化 Ark 客户端')
            client = AsyncArk(
                api_key=key,
                base_url=self.config.base_url,
                max_retries=0,
                timeout=httpx.Timeout(
                    self.config.read_timeout,
                    connect=self.config.connect_timeout,
                ),
            )
        self._client = client
        self._streams: dict[int, Any] = {}
        self._closing_streams: set[asyncio.Task[None]] = set()
        self._operations: set[asyncio.Future[Any]] = set()
        self._closed = asyncio.Event()
        self._close_task: asyncio.Task[None] | None = None

    @staticmethod
    def _input(request: ModelRequest) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for entry in request.history:
            kind = entry.get('kind')
            if kind == 'message':
                role, content = entry.get('role'), entry.get('content')
                if role not in ('user', 'assistant') or not isinstance(content, str):
                    raise _ProtocolError('消息历史的角色或内容无效')
                items.append({'role': role, 'content': content})
            elif kind == 'response':
                output = entry.get('items')
                if not isinstance(output, (list, tuple)):
                    raise _ProtocolError('响应历史缺少完整输出项')
                items.extend(_record(item) for item in output)
            elif kind == 'tool_result':
                call_id, result = entry.get('call_id'), entry.get('result')
                if not isinstance(call_id, str) or not call_id.strip():
                    raise _ProtocolError('工具结果缺少 call_id')
                if not isinstance(result, Mapping):
                    raise _ProtocolError('工具结果必须是 ToolResult 的字典')
                if result.get('call_id', call_id) != call_id:
                    raise _ProtocolError('工具结果的 call_id 不一致')
                items.append({
                    'type': 'function_call_output',
                    'call_id': call_id,
                    'output': json.dumps(dict(result), ensure_ascii=False),
                })
            else:
                raise _ProtocolError('未知的模型历史类型')
        return items

    @staticmethod
    def _finished(event: Any, request_id: str) -> ModelFrame:
        response = _field(event, 'response')
        if response is None:
            raise _ProtocolError('终态事件缺少完整响应')
        event_type = _field(event, 'type')
        status = {
            'response.completed': 'completed',
            'response.incomplete': 'incomplete',
            'response.failed': 'failed',
        }[event_type]
        response_status = _field(response, 'status')
        if status == 'completed' and response_status not in (None, 'completed'):
            status = 'incomplete' if response_status == 'incomplete' else 'failed'
        if _field(response, 'error') is not None:
            status = 'failed'

        output = _field(response, 'output', _MISSING)
        if output is _MISSING and status != 'completed':
            output = []
        if not isinstance(output, (list, tuple)):
            raise _ProtocolError('终态响应缺少完整 output')
        raw_output = tuple(_record(item) for item in output)
        raw_usage = _field(response, 'usage')
        usage = {} if raw_usage is None else {
            key: value for key, value in _record(raw_usage).items()
            if isinstance(value, int) and not isinstance(value, bool)
        }
        calls: list[ToolCall] = []
        seen_ids: set[str] = set()
        diagnostic = ''
        if status == 'completed':
            for item in raw_output:
                if item.get('type') != 'function_call':
                    continue
                call_id, name, arguments = (
                    item.get('call_id'), item.get('name'), item.get('arguments')
                )
                complete = all(
                    isinstance(value, str) and value.strip()
                    for value in (call_id, name, arguments)
                )
                if (not complete or call_id in seen_ids
                        or item.get('status') not in (None, 'completed')):
                    # 任一调用残缺时整轮作废；绝不放行已经收集的其他调用。
                    status = 'failed'
                    diagnostic = 'PROVIDER_PROTOCOL_ERROR: 工具调用不完整或 call_id 重复'
                    calls.clear()
                    break
                seen_ids.add(call_id)
                # JSON 和工具参数校验属于 engine，适配层只确认原始字符串完整。
                calls.append(ToolCall(request_id, call_id, name, arguments))
        return ModelFrame(
            'response_finished', text=diagnostic, status=status,
            calls=tuple(calls), provider_output=raw_output, usage=usage,
        )

    async def _await_cancel(
        self, operation: Awaitable[_T], cancel_event: asyncio.Event,
        timeout: float | None = None,
    ) -> _T:
        work = asyncio.ensure_future(operation)
        self._operations.add(work)
        cancelled = asyncio.create_task(cancel_event.wait())
        closed = asyncio.create_task(self._closed.wait())
        tasks = (work, cancelled, closed)
        try:
            if cancel_event.is_set() or self._closed.is_set():
                raise asyncio.CancelledError
            done, _ = await asyncio.wait(
                tasks, timeout=timeout, return_when=asyncio.FIRST_COMPLETED,
            )
            # 取消与完成同时到达时，取消优先，避免继续放行工具调用。
            if cancel_event.is_set() or self._closed.is_set():
                raise asyncio.CancelledError
            if work not in done:
                raise TimeoutError
            return work.result()
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._operations.discard(work)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        status = getattr(exc, 'status_code', None)
        if isinstance(status, int):
            return status == 429 or 500 <= status <= 599
        return isinstance(exc, (
            ArkAPIConnectionError, httpx.ConnectError, httpx.ConnectTimeout,
            ConnectionError,
        ))

    @staticmethod
    def _retry_delay(exc: Exception, attempt: int) -> float:
        delay = float(2 ** attempt)
        headers = _field(_field(exc, 'response'), 'headers', {})
        retry_after = headers.get('retry-after') if hasattr(headers, 'get') else None
        if retry_after is None and hasattr(headers, 'get'):
            retry_after = headers.get('Retry-After')
        if retry_after is not None:
            try:
                seconds = float(retry_after)
            except (TypeError, ValueError):
                try:
                    seconds = parsedate_to_datetime(str(retry_after)).timestamp() - time.time()
                except (TypeError, ValueError, OverflowError):
                    seconds = 0.0
            if math.isfinite(seconds):
                delay = max(delay, min(_MAX_RETRY_AFTER, seconds))
        return delay

    @staticmethod
    async def _close_resource(resource: Any) -> None:
        close = getattr(resource, 'close', None) or getattr(resource, 'aclose', None)
        if callable(close):
            try:
                result = close()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                # 清理错误不能覆盖取消或终态，也不转发可能包含凭证的异常。
                pass

    async def _close_stream(self, stream: Any) -> None:
        if stream is not None and self._streams.pop(id(stream), None) is not None:
            # 清理不随读取任务的再次取消而中断，close() 也会等待在途清理。
            closing = asyncio.create_task(self._close_resource(stream))
            self._closing_streams.add(closing)
            closing.add_done_callback(self._closing_streams.discard)
            await asyncio.shield(closing)

    async def stream(
        self, request: ModelRequest, cancel_event: asyncio.Event,
    ) -> AsyncIterator[ModelFrame]:
        if cancel_event.is_set():
            raise asyncio.CancelledError
        if self._closed.is_set():
            raise RuntimeError('PROVIDER_UNAVAILABLE: ArkProvider 已关闭')
        try:
            inputs = self._input(request)
        except (ValueError, TypeError, AttributeError):
            yield ModelFrame(
                'response_finished', status='failed',
                text='PROVIDER_PROTOCOL_ERROR: 模型历史无法编码',
            )
            return

        request_id = uuid4().hex
        for attempt in range(_MAX_RETRIES + 1):
            sdk_stream = None
            received_event = False
            retry_delay = None
            terminal = ModelFrame('response_finished', status='incomplete')

            async def open_stream() -> Any:
                nonlocal sdk_stream
                sdk_stream = await self._client.responses.create(
                    model=self.model, instructions=request.instructions,
                    input=deepcopy(inputs), tools=deepcopy(list(request.tool_schemas)),
                    max_output_tokens=request.max_output_tokens,
                    store=False, stream=True,
                )
                # 在协程内登记资源，保证连接成功与取消同时到达也能关闭它。
                self._streams[id(sdk_stream)] = sdk_stream
                return sdk_stream

            try:
                await self._await_cancel(open_stream(), cancel_event)
                iterator = sdk_stream.__aiter__()
                while True:
                    try:
                        event = await self._await_cancel(
                            anext(iterator), cancel_event, self.config.read_timeout,
                        )
                    except StopAsyncIteration:
                        terminal = ModelFrame(
                            'response_finished', status='incomplete',
                            text='PROVIDER_PROTOCOL_ERROR: 流在完整终态前结束',
                        )
                        break
                    received_event = True
                    event_type = _field(event, 'type')
                    if event_type == 'response.output_text.delta':
                        delta = _field(event, 'delta')
                        if not isinstance(delta, str):
                            raise _ProtocolError('文本增量无效')
                        if delta:
                            yield ModelFrame('text_delta', text=delta)
                    elif event_type in (
                        'response.completed', 'response.failed', 'response.incomplete',
                    ):
                        terminal = self._finished(event, request_id)
                        break
                    elif event_type in ('error', 'response.error'):
                        terminal = ModelFrame(
                            'response_finished', status='failed',
                            text='PROVIDER_UNAVAILABLE: 模型流返回错误',
                        )
                        break
                    # 参数 delta/done 与 output_item.done 都不产生可执行 calls。
            except asyncio.CancelledError:
                raise
            except _ProtocolError:
                terminal = ModelFrame(
                    'response_finished', status='failed',
                    text='PROVIDER_PROTOCOL_ERROR: 模型响应字段不完整或无效',
                )
            except Exception as exc:
                # SDK 重试为 0；一旦收到任何事件便不重试，避免重复展示增量。
                if not received_event and attempt < _MAX_RETRIES and self._retryable(exc):
                    retry_delay = self._retry_delay(exc, attempt)
                else:
                    terminal = ModelFrame(
                        'response_finished',
                        status='incomplete' if received_event else 'failed',
                        text='PROVIDER_UNAVAILABLE: 模型请求失败或流被中断',
                    )
            finally:
                await self._close_stream(sdk_stream)

            if retry_delay is not None:
                await self._await_cancel(asyncio.sleep(retry_delay), cancel_event)
                continue
            if cancel_event.is_set() or self._closed.is_set():
                raise asyncio.CancelledError
            # 先释放 SDK 连接再交付终态；完成响应的全文不能重复作为增量显示。
            if terminal.usage:
                yield ModelFrame('usage', usage=terminal.usage)
            if cancel_event.is_set() or self._closed.is_set():
                raise asyncio.CancelledError
            yield terminal
            return

    async def _shutdown(self) -> None:
        operations = tuple(self._operations)
        for operation in operations:
            if not operation.done():
                operation.cancel()
        await asyncio.gather(*operations, return_exceptions=True)
        for stream in tuple(self._streams.values()):
            await self._close_stream(stream)
        await asyncio.gather(*tuple(self._closing_streams), return_exceptions=True)
        await self._close_resource(self._client)

    async def close(self) -> None:
        """关闭在途连接与客户端；重复调用等待同一次清理完成。"""
        if self._close_task is None:
            self._closed.set()
            self._close_task = asyncio.create_task(self._shutdown())
        await asyncio.shield(self._close_task)
