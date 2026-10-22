"""技术方案§Ark请求、流式响应与凭证；所有客户端及事件均为本地替身。"""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from coding_agent.domain import ModelRequest
from coding_agent.providers import ark
from coding_agent.providers.ark import ArkProvider


class FakeStream:
    def __init__(self, events):
        self.events = iter(events)
        self.closed = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            item = next(self.events)
        except StopIteration:
            raise StopAsyncIteration
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed += 1


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.requests = []
        self.streams = []
        self.responses = SimpleNamespace(create=self.create)
        self.closed = 0

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        value = self.outcomes.pop(0)
        if isinstance(value, Exception):
            raise value
        stream = value if hasattr(value, "__aiter__") else FakeStream(value)
        self.streams.append(stream)
        return stream

    async def close(self):
        self.closed += 1


def request(history=()):
    return ModelRequest("固定测试约定", tuple(history), ({"type": "function", "name": "read_file"},), 4096)


def terminal(status="completed", output=None):
    return {"type": "response." + status, "response": {"status": status, "output": output or []}}


def function(call_id="c1", **overrides):
    return {"type": "function_call", "call_id": call_id, "name": "read_file", "arguments": '{"path":"a.py"}', **overrides}


async def collect(provider, model_request=None, cancel=None):
    async with asyncio.timeout(3):
        return [f async for f in provider.stream(model_request or request(), cancel or asyncio.Event())]


async def test_text_delta_and_complete_call_preserve_all_output_fields():
    output = [function(), {"type": "reasoning", "id": "opaque", "extra": {"retain": [1, 2]}}]
    end = terminal(output=output)
    end["response"]["usage"] = {"input_tokens": 12, "output_tokens": 3}
    client = FakeClient([[{"type": "response.output_text.delta", "delta": "文本"}, end]])
    provider = ArkProvider(client=client, model="test-model")
    frames = await collect(provider)
    assert [(f.kind, f.text) for f in frames if f.kind == "text_delta"] == [("text_delta", "文本")]
    final = frames[-1]
    assert final.status == "completed"
    assert final.provider_output == tuple(output)
    assert len(final.calls) == 1
    assert (final.calls[0].call_id, final.calls[0].name, final.calls[0].arguments_json) == ("c1", "read_file", '{"path":"a.py"}')
    assert final.calls[0].request_id
    assert any(f.kind == "usage" and f.usage == {"input_tokens": 12, "output_tokens": 3} for f in frames)
    assert client.requests[0]["store"] is False
    assert client.requests[0]["stream"] is True
    assert client.requests[0]["max_output_tokens"] == 4096
    assert client.streams[0].closed == 1


async def test_sdk_object_events_are_supported():
    event = SimpleNamespace(type="response.completed", response=SimpleNamespace(status="completed", output=[SimpleNamespace(**function())]))
    frames = await collect(ArkProvider(client=FakeClient([[event]]), model="test-model"))
    assert frames[-1].calls[0].call_id == "c1"


@pytest.mark.parametrize("status", ["failed", "incomplete"])
async def test_arguments_done_and_non_success_terminal_expose_zero_calls(status):
    events = [{"type": "response.function_call_arguments.delta", "delta": '{"path":'}, {"type": "response.function_call_arguments.done", "arguments": '{"path":"a.py"}'}, terminal(status, [function()])]
    frames = await collect(ArkProvider(client=FakeClient([events]), model="test-model"))
    assert all(frame.calls == () for frame in frames)
    assert frames[-1].status == status


async def test_arguments_done_without_completed_response_exposes_zero_calls():
    frames = await collect(ArkProvider(client=FakeClient([[{"type": "response.function_call_arguments.done", "arguments": '{"path":"a.py"}'}]]), model="test-model"))
    assert frames[-1].status == "incomplete"
    assert not any(f.calls for f in frames)


@pytest.mark.parametrize("output", [
    pytest.param([function(call_id="")], id="missing-id"),
    pytest.param([function(name="")], id="missing-name"),
    pytest.param([function(arguments="")], id="missing-arguments"),
    pytest.param([function(), function()], id="duplicate-id"),
    pytest.param([function(status="in_progress")], id="unfinished-call"),
])
async def test_incomplete_call_invalidates_whole_completed_response(output):
    frames = await collect(ArkProvider(client=FakeClient([[terminal(output=output)]]), model="test-model"))
    assert frames[-1].status == "failed"
    assert not any(f.calls for f in frames)


async def test_complete_history_encodes_original_output_and_matching_tool_result():
    output = [function(), {"type": "reasoning", "additional": {"opaque": "unchanged"}}]
    result = {"call_id": "c1", "ok": True, "data": {"text": "result"}, "error": None, "truncated": False}
    history = [{"kind": "message", "role": "user", "content": "读取"}, {"kind": "response", "items": output}, {"kind": "tool_result", "call_id": "c1", "result": result}]
    original = deepcopy(history)
    client = FakeClient([[terminal()]])
    await collect(ArkProvider(client=client, model="test-model"), request(history))
    encoded = client.requests[0]["input"]
    assert encoded[1:3] == output
    assert encoded[3]["type"] == "function_call_output"
    assert encoded[3]["call_id"] == "c1"
    assert json.loads(encoded[3]["output"]) == result
    assert history == original


async def test_no_event_connection_failures_retry_twice_with_one_two_delays(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(ark.asyncio, "sleep", sleep)
    client = FakeClient([ConnectionError("test"), ConnectionError("test"), [terminal()]])
    frames = await collect(ArkProvider(client=client, model="test-model"))
    assert len(client.requests) == 3
    assert [c.args[0] for c in sleep.await_args_list] == [1.0, 2.0]
    assert frames[-1].status == "completed"


async def test_retry_budget_is_finite(monkeypatch):
    monkeypatch.setattr(ark.asyncio, "sleep", AsyncMock())
    client = FakeClient([ConnectionError("test")] * 3)
    frames = await collect(ArkProvider(client=client, model="test-model"))
    assert len(client.requests) == 3
    assert frames[-1].status == "failed"


async def test_stream_failure_after_delta_is_not_retried():
    client = FakeClient([[{"type": "response.output_text.delta", "delta": "前文"}, ConnectionError("test")]])
    frames = await collect(ArkProvider(client=client, model="test-model"))
    assert len(client.requests) == 1
    assert [f.text for f in frames if f.kind == "text_delta"] == ["前文"]
    assert frames[-1].status == "incomplete"
    assert not any(f.calls for f in frames)
    assert client.streams[0].closed == 1


async def test_blocked_read_is_cancelled_and_stream_closed():
    entered = asyncio.Event()

    class WaitingStream(FakeStream):
        async def __anext__(self):
            entered.set()
            await asyncio.Event().wait()

    stream = WaitingStream([])
    client = FakeClient([stream])
    provider = ArkProvider(client=client, model="test-model")
    cancel = asyncio.Event()
    task = asyncio.create_task(collect(provider, cancel=cancel))
    try:
        async with asyncio.timeout(3):
            await entered.wait()
        cancel.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert stream.closed == 1
        await provider.close()
        assert client.closed == 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


def test_client_constructor_disables_sdk_retries(monkeypatch):
    constructor = Mock(return_value=FakeClient([]))
    monkeypatch.setattr(ark, "AsyncArk", constructor)
    ArkProvider(model="test-model", api_key="<your-test-key>")
    assert constructor.call_args.kwargs["max_retries"] == 0
    assert constructor.call_args.kwargs["timeout"].connect == 10


@pytest.mark.parametrize("status", [429, 503, 401, 400])
async def test_status_retries_respect_retry_after_and_auth_errors(monkeypatch, status):
    class Error(Exception):
        status_code = status
        response = SimpleNamespace(headers={"Retry-After": "100"})

    sleeper = AsyncMock()
    monkeypatch.setattr(ark.asyncio, "sleep", sleeper)
    client = FakeClient([Error("test-error"), [terminal()]])
    frames = await collect(ArkProvider(client=client, model="test-model"))
    if status in (429, 503):
        assert len(client.requests) == 2
        assert frames[-1].status == "completed"
        # 方案只约定Retry-After有上限，没有规定上限数值，不钉死实现常量。
        assert sleeper.await_args.args[0] >= 1
    else:
        assert len(client.requests) == 1
        assert frames[-1].status == "failed"
        sleeper.assert_not_awaited()


async def test_provider_close_cancels_blocked_open_and_closes_client():
    entered = asyncio.Event()

    class WaitingClient(FakeClient):
        async def create(self, **kwargs):
            entered.set()
            await asyncio.Event().wait()

    client = WaitingClient([])
    provider = ArkProvider(client=client, model="test-model")
    task = asyncio.create_task(collect(provider))
    try:
        async with asyncio.timeout(3):
            await entered.wait()
        await provider.close()
        with pytest.raises(asyncio.CancelledError):
            await task
        await provider.close()
        assert client.closed == 1
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
