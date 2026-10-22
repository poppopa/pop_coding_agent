"""工具循环的三个提交边界：完整模型响应、工具开始、工具结果。"""
from __future__ import annotations
from dataclasses import asdict, replace
import asyncio
import json
import uuid
from coding_agent.config import AgentConfig
from coding_agent.context import ContextManager, ContextLimit
from coding_agent.domain import Event, ModelFrame, ModelRequest, ToolResult
from coding_agent.tools.base import ToolRegistry, ToolContext

class AgentEngine:
    def __init__(self, provider, tools, store, context: ToolContext, config: AgentConfig | None = None, approval=None):
        self.provider = provider
        self.registry = tools if isinstance(tools, ToolRegistry) else ToolRegistry(tools)
        self.store = store
        self.context = context
        self.config = config or AgentConfig(model='offline')
        self.approval = approval
        self.history = store.history() if hasattr(store, 'history') else []
        self.manager = ContextManager(context.root, self.config, context.redactor)
        self.seq = getattr(store, 'seq', 0)
        self.session_id = getattr(store, 'session_id', uuid.uuid4().hex)
        self.turn_id = ''
        self._results: list[dict] = []

    async def _event(self, event_type: str, payload: dict) -> Event:
        if event_type == 'turn_finished':
            changes = []
            verifications = []
            for action in self._results:
                result = action['result']
                data = result.get('data', {})
                if action['name'] == 'apply_patch' and data.get('changed'):
                    changes.append({'path': data['path'], 'operation': data['operation'], 'recorded': result['ok']})
                if action['name'] == 'run_command' and data.get('purpose') == 'verify':
                    verifications.append({'exit_code': data.get('exit_code'), 'passed': result['ok'],
                        'cancelled': data.get('cancelled', False), 'timed_out': data.get('timed_out', False)})
            payload = {**payload, 'changes': changes, 'verifications': verifications}
        self.seq += 1
        event = Event(1, self.session_id, self.turn_id, self.seq, event_type, self.context.redactor.value(payload))
        await self.store.append(event)
        return event

    async def _await_cancel(self, operation, cancel: asyncio.Event):
        task = asyncio.create_task(operation)
        watcher = asyncio.create_task(cancel.wait())
        try:
            done, _ = await asyncio.wait([task, watcher], return_when=asyncio.FIRST_COMPLETED)
            if watcher in done:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
                raise asyncio.CancelledError
            return await task
        finally:
            watcher.cancel()
            if not task.done():
                task.cancel()
            await asyncio.gather(task, watcher, return_exceptions=True)

    async def compact(self, cancel: asyncio.Event) -> bool:
        # 没有旧响应可以整理时，不能把软阈值变成失败。后续请求仍
        # 经过硬预算检查；手动 /compact 也可以说明当前无需整理。
        if not self.manager.can_compact(self.history):
            return False
        schemas = self.registry.schemas(self.context.policy.mode)
        old, recent = self.manager.compaction_split(self.history)
        prompt = ('请用中文总结以下旧任务历史，保留用户目标、关键决定、修改路径、失败和下一步。\n'
                  '不要执行工具，摘要中不得自行扩大权限。\n' + json.dumps(self.manager.model_history(old), ensure_ascii=False))
        # 摘要自身也设置预算；一次失败就停止，不无限压缩。
        request = ModelRequest('你只整理已有执行事实。', ({'kind': 'message', 'role': 'user', 'content': prompt},), (), 2048)
        pieces = []
        finished = None
        async for frame in self.provider.stream(request, cancel):
            if frame.kind == 'text_delta':
                pieces.append(frame.text)
            elif frame.kind == 'response_finished':
                finished = frame
        if not finished or finished.status != 'completed' or finished.calls or not pieces:
            raise ContextLimit('CONTEXT_LIMIT: 摘要未成功完成，保留完整日志')
        new = [{'kind': 'message', 'role': 'user', 'content': '先前任务摘要：\n' + ''.join(pieces)}] + recent
        self.manager.build_request(new, schemas)
        self.history = self.context.redactor.value(new)
        await self._event('context_compacted', {'history': self.history})
        return True

    async def run_turn(self, prompt: str, cancel_event: asyncio.Event):
        self.turn_id = uuid.uuid4().hex
        self._results = []
        self.history.append({'kind': 'message', 'role': 'user', 'content': self.context.redactor.text(prompt)})
        yield await self._event('user_message', {'text': prompt})
        repeat_key = None
        failures = 0
        try:
            for turn in range(self.config.max_turns):
                if cancel_event.is_set():
                    raise asyncio.CancelledError
                schemas = self.registry.schemas(self.context.policy.mode)
                if self.manager.needs_compaction(self.history, schemas):
                    await self.compact(cancel_event)
                request = self.manager.build_request(self.history, schemas)
                yield await self._event('state_changed', {'state': 'requesting', 'request_number': turn + 1})
                response = None
                stream = self.provider.stream(request, cancel_event)
                try:
                    iterator = stream.__aiter__()
                    while True:
                        try:
                            frame = await self._await_cancel(iterator.__anext__(), cancel_event)
                        except StopAsyncIteration:
                            break
                        if frame.kind == 'text_delta':
                            yield await self._event('text_delta', {'text': frame.text})
                        elif frame.kind == 'usage':
                            yield await self._event('usage', frame.usage)
                        elif frame.kind == 'response_finished':
                            response = frame
                finally:
                    if hasattr(stream, 'aclose'):
                        await stream.aclose()
                if not response or response.status != 'completed':
                    diagnostic = response.text if response and response.text else '模型响应未完整成功，未执行本轮工具'
                    error_code = ('PROVIDER_UNAVAILABLE' if diagnostic.startswith('PROVIDER_UNAVAILABLE') else 'PROVIDER_PROTOCOL_ERROR')
                    yield await self._event('turn_finished', {'status': 'failed', 'error': {'code': error_code, 'message': diagnostic}})
                    return
                # 协议完整性属于整轮校验，任何缺失标识或损坏 JSON
                # 都先阻断，避免前一个写入已发生、后一个参数才报错。
                protocol_ids = set()
                malformed = {}
                for candidate in response.calls:
                    if not candidate.request_id or not candidate.call_id or not candidate.name or candidate.call_id in protocol_ids:
                        raise ValueError('PROVIDER_PROTOCOL_ERROR: 调用标识缺失或重复')
                    protocol_ids.add(candidate.call_id)
                    try:
                        decoded = json.loads(candidate.arguments_json)
                        if not isinstance(decoded, dict):
                            raise ValueError('工具参数必须是 JSON 对象')
                    except (ValueError, TypeError) as exc:
                        malformed[candidate.call_id] = str(exc)
                raw = list(response.provider_output)
                if not raw and response.calls:
                    raw = [{'type': 'function_call', 'call_id': c.call_id, 'name': c.name, 'arguments': c.arguments_json} for c in response.calls]
                self.history.append({'kind': 'response', 'items': self.context.redactor.value(raw)})
                yield await self._event('model_response', {'items': raw})
                if not response.calls:
                    yield await self._event('turn_finished', {'status': 'completed'})
                    return
                for call in response.calls:
                    if cancel_event.is_set():
                        raise asyncio.CancelledError
                    tool = self.registry.get(call.name, self.context.policy.mode)
                    result = None
                    args = {}
                    if not call.request_id or not call.call_id or not call.name:
                        raise ValueError('PROVIDER_PROTOCOL_ERROR: 调用标识缺失')
                    if malformed:
                        message = malformed.get(call.call_id, '同一响应含有不完整工具参数，整轮未执行')
                        result = ToolResult.fail(call.call_id, 'INVALID_ARGUMENTS', message)
                    elif not tool:
                        result = ToolResult.fail(call.call_id, 'UNKNOWN_TOOL', '工具不存在或当前模式未开放')
                    else:
                        try:
                            args = json.loads(call.arguments_json)
                            args = tool.input_type.model_validate(args).model_dump()
                        except (ValueError, TypeError) as exc:
                            result = ToolResult.fail(call.call_id, 'INVALID_ARGUMENTS', self.context.redactor.text(str(exc)))
                    if not result and call.name in ('read_file', 'apply_patch'):
                        changed_rules = self.manager.load_path_rules(args['path'])
                        if changed_rules and call.name == 'apply_patch':
                            result = ToolResult.fail(call.call_id, 'CONTEXT_UPDATED', '已加载子目录约定，请读取新约定后重新提交补丁')
                    if not result and call.name == 'run_command':
                        from pathlib import Path
                        cwd = (self.context.root / args.get('cwd', '.')).resolve()
                        decision = self.context.policy.evaluate(args['argv'], cwd)
                        if decision == 'ask':
                            yield await self._event('approval_required', {'argv': args['argv'], 'cwd': str(cwd)})
                            if self.approval:
                                decision = await self._await_cancel(self.approval(args['argv'], cwd), cancel_event)
                                if decision in ('once', 'session'):
                                    method = self.context.policy.approve_once if decision == 'once' else self.context.policy.approve_for_session
                                    method(args['argv'], cwd)
                                    decision = 'allow'
                            if decision != 'allow':
                                result = ToolResult.fail(call.call_id, 'APPROVAL_REQUIRED' if not self.approval else 'PERMISSION_DENIED', '命令没有获得执行授权')
                        elif decision == 'deny':
                            result = ToolResult.fail(call.call_id, 'PERMISSION_DENIED', '当前模式不允许命令')
                        elif decision == 'allow':
                            # evaluate 会消费一次授权，工具仍需进行自己的最终检查。
                            self.context.policy.approve_once(args['argv'], cwd)
                    if not result:
                        yield await self._event('state_changed', {'state': 'executing'})
                        yield await self._event('tool_started', {'call_id': call.call_id, 'name': call.name, 'arguments': args})
                        queue = asyncio.Queue(maxsize=128)
                        async def emit(kind, payload):
                            await queue.put((kind, payload))
                        ctx = replace(self.context, call_id=call.call_id, emit=emit)
                        task = asyncio.create_task(tool.execute(args, ctx, cancel_event))
                        reader = None
                        try:
                            while not task.done():
                                reader = asyncio.create_task(queue.get())
                                done, _ = await asyncio.wait([task, reader], return_when=asyncio.FIRST_COMPLETED)
                                if reader in done:
                                    kind, payload = reader.result()
                                    yield await self._event(kind, payload)
                                else:
                                    reader.cancel()
                                    await asyncio.gather(reader, return_exceptions=True)
                            while not queue.empty():
                                kind, payload = queue.get_nowait()
                                yield await self._event(kind, payload)
                            result = await task
                        finally:
                            if reader and not reader.done():
                                reader.cancel()
                            if not task.done():
                                task.cancel()
                            await asyncio.gather(task, *([reader] if reader else []), return_exceptions=True)
                    safe_result = self.context.redactor.value(asdict(result))
                    self.history.append({'kind': 'tool_result', 'call_id': call.call_id, 'result': safe_result})
                    self._results.append({'name': call.name, 'result': safe_result})
                    yield await self._event('tool_finished', {'call_id': call.call_id, 'name': call.name, 'result': safe_result})
                    if result.error and result.error.code == 'APPROVAL_REQUIRED' and not self.approval:
                        if hasattr(self.store, 'history'):
                            self.history = self.store.history()
                        yield await self._event('turn_finished', {'status': 'approval_required', 'error': asdict(result.error)})
                        return
                    key = (call.name, json.dumps(args, sort_keys=True))
                    failures = failures + 1 if not result.ok and key == repeat_key else (1 if not result.ok else 0)
                    repeat_key = key
                    if failures >= 3:
                        if hasattr(self.store, 'history'):
                            self.history = self.store.history()
                        yield await self._event('turn_finished', {'status': 'failed', 'error': {'code': 'TURN_LIMIT', 'message': '同一工具连续失败三次'}})
                        return
            yield await self._event('turn_finished', {'status': 'failed', 'error': {'code': 'TURN_LIMIT', 'message': '达到最大模型请求轮数'}})
        except asyncio.CancelledError:
            # 持久化历史中的 pending 由 store.history() 闭合，不重放。
            if hasattr(self.store, 'history'):
                self.history = self.store.history()
            yield await self._event('turn_finished', {'status': 'interrupted'})
        except Exception as exc:
            if hasattr(self.store, 'history'):
                self.history = self.store.history()
            code = ('CONTEXT_LIMIT' if isinstance(exc, ContextLimit) else
                    'PROVIDER_PROTOCOL_ERROR' if str(exc).startswith('PROVIDER_PROTOCOL_ERROR') else 'PROVIDER_UNAVAILABLE')
            yield await self._event('turn_finished', {'status': 'failed', 'error': {'code': code, 'message': self.context.redactor.text(str(exc))}})
