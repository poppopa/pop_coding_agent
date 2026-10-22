"""验收依据：技术方案§核心契约与状态机、Ark 请求、规格说明 SC-001/002。

模型流、工具外部行为和持久化使用替身，注册表、Policy、参数校验和内核保持真实。
异步信号提供明确的执行观察时机；不访问模型客户端或真实凭证。
"""
import asyncio
from dataclasses import asdict, is_dataclass
import json

import pytest

from coding_agent.config import AgentConfig
from coding_agent.domain import ModelFrame, ToolCall, ToolResult
from coding_agent.engine import AgentEngine
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.tools.base import Tool, ToolContext, ToolInput


class ReadArguments(ToolInput):
    path: str


class RecordingTool(Tool):
    name = "read_file"
    description = "可观察的外部读取依赖"
    input_type = ReadArguments

    def __init__(self, *, name="read_file", mutating=False, fail=False):
        self.name, self.mutating, self.fail = name, mutating, fail
        self.executions = []

    async def execute(self, arguments, context, cancel_event):
        self.executions.append((dict(arguments), context.call_id))
        if self.fail:
            return ToolResult.fail(context.call_id, "FILE_NOT_FOUND", "测试文件不存在")
        return ToolResult(context.call_id, True, {"text": "answer = 7\n", "path": arguments["path"]})


class MemoryStore:
    def __init__(self):
        self.events = []

    async def append(self, event):
        self.events.append(event)

    async def load(self, session_id):
        return list(self.events)


class FakeProvider:
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.requests = []
        self.closed = 0

    async def stream(self, request, cancel_event):
        self.requests.append(request)
        if not self.rounds:
            raise AssertionError("模型请求超过测试预置轮数")
        frames = self.rounds.pop(0)
        try:
            for frame in frames:
                if isinstance(frame, Exception):
                    raise frame
                yield frame
        finally:
            self.closed += 1


def tool_call(call_id="call-1", *, name="read_file", arguments='{"path":"a.py"}', request_id="request-1"):
    return ToolCall(request_id, call_id, name, arguments)


def completed(*calls):
    return ModelFrame("response_finished", status="completed", calls=tuple(calls))


def setup_engine(tmp_path, provider, tools, *, mode="execute", max_turns=30):
    store = MemoryStore()

    async def emit(_kind, _payload):
        pass

    context = ToolContext(tmp_path, "", None, Policy(mode=mode), emit, Redactor(), tmp_path / "state")
    config = AgentConfig(model="test-model", mode=mode, max_turns=max_turns, state_dir=tmp_path / "state")
    return AgentEngine(provider, tools, store, context, config=config), store


async def collect(engine, cancel_event=None):
    async with asyncio.timeout(3):
        return [event async for event in engine.run_turn("读取 a.py", cancel_event or asyncio.Event())]


def dictionaries(value):
    """观察领域结构或 JSON 编码；不依赖供应商专有字段重建被测逻辑。"""
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            return
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from dictionaries(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from dictionaries(child)


def assert_tool_result(value, call_id, *, ok, data=None, error_code=None):
    matches = [item for item in dictionaries(value) if item.get("call_id") == call_id and "ok" in item]
    assert matches, f"缺失 {call_id} 的结构化工具结果"
    result = matches[-1]
    assert result["ok"] is ok
    assert "data" in result and "error" in result and "truncated" in result
    if data is not None:
        assert result["data"] == data
        assert result["error"] is None
    if error_code is not None:
        assert result["error"]["code"] == error_code
        assert isinstance(result["error"]["message"], str) and result["error"]["message"]
        assert isinstance(result["error"]["retryable"], bool)


def assert_terminal(events, status, error_code=None):
    assert events[-1].type == "turn_finished"
    assert events[-1].payload["status"] == status
    if error_code:
        assert any(item.get("code") == error_code for item in dictionaries(events[-1].payload))


async def test_tool_result_with_same_call_id_reaches_second_request(tmp_path):
    output = ({"type": "function_call", "call_id": "call-1", "name": "read_file", "arguments": '{"path":"a.py"}', "opaque": {"keep": "original"}},)
    first = ModelFrame("response_finished", status="completed", calls=(tool_call(),), provider_output=output)
    provider = FakeProvider([[first], [ModelFrame("text_delta", text="读取完成"), completed()]])
    tool = RecordingTool()
    engine, store = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == [({"path": "a.py"}, "call-1")]
    assert len(provider.requests) == 2
    assert_tool_result(provider.requests[1].history, "call-1", ok=True, data={"text": "answer = 7\n", "path": "a.py"})
    assert any(item == output[0] for item in dictionaries(provider.requests[1].history))
    assert_terminal(events, "completed")
    assert store.events == events
    assert all(event.schema_version == 1 for event in events)
    assert len({event.session_id for event in events}) == 1
    assert len({event.turn_id for event in events}) == 1
    assert all(a.seq < b.seq for a, b in zip(events, events[1:]))


@pytest.mark.parametrize("status", ["incomplete", "failed", "in_progress"])
async def test_non_completed_response_executes_zero_tools(tmp_path, status):
    provider = FakeProvider([[ModelFrame("response_finished", status=status, calls=(tool_call(),))]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == []
    assert len(provider.requests) == 1
    assert_terminal(events, "failed")


async def test_stream_ending_without_finished_frame_executes_zero_tools(tmp_path):
    provider = FakeProvider([[ModelFrame("text_delta", text="准备读取")]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == []
    assert_terminal(events, "failed", "PROVIDER_PROTOCOL_ERROR")


async def test_stream_exception_after_delta_executes_zero_tools(tmp_path):
    provider = FakeProvider([[ModelFrame("text_delta", text="准备读取"), ConnectionError("测试断流")]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == []
    assert len(provider.requests) == 1
    assert_terminal(events, "failed")


async def test_invalid_json_returns_structured_error_without_execution(tmp_path):
    provider = FakeProvider([[completed(tool_call(arguments='{"path":'))], [completed()]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == []
    assert_tool_result(events, "call-1", ok=False, error_code="INVALID_ARGUMENTS")


async def test_invalid_json_in_later_call_prevents_whole_response_execution(tmp_path):
    provider = FakeProvider([[completed(tool_call(), tool_call("call-2", arguments='{"path":'))], [completed()]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == []
    assert_tool_result(events, "call-2", ok=False, error_code="INVALID_ARGUMENTS")


@pytest.mark.parametrize("arguments", [
    pytest.param('{"path":"a.py","extra":true}', id="extra-field"),
    pytest.param('{"path":12}', id="wrong-type"),
    pytest.param('{}', id="missing-field"),
    pytest.param('[]', id="non-object"),
])
async def test_invalid_tool_parameters_are_structured_errors(tmp_path, arguments):
    provider = FakeProvider([[completed(tool_call(arguments=arguments))], [completed()]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == []
    assert_tool_result(events, "call-1", ok=False, error_code="INVALID_ARGUMENTS")


async def test_unknown_tool_error_does_not_block_other_valid_call(tmp_path):
    provider = FakeProvider([[completed(tool_call(name="unknown"), tool_call("call-2"))], [completed()]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == [({"path": "a.py"}, "call-2")]
    assert_tool_result(events, "call-1", ok=False, error_code="UNKNOWN_TOOL")
    assert_tool_result(provider.requests[1].history, "call-1", ok=False, error_code="UNKNOWN_TOOL")


@pytest.mark.parametrize("field", ["call_id", "request_id", "name"])
async def test_missing_call_identifiers_prevent_execution(tmp_path, field):
    values = {"request_id": "request-1", "call_id": "call-1", "name": "read_file", "arguments_json": '{"path":"a.py"}'}
    values[field] = ""
    provider = FakeProvider([[completed(ToolCall(**values))]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert tool.executions == []
    assert_terminal(events, "failed", "PROVIDER_PROTOCOL_ERROR")


async def test_analysis_hides_mutating_tool_and_rejects_hallucinated_call(tmp_path):
    read = RecordingTool()
    patch = RecordingTool(name="apply_patch", mutating=True)
    provider = FakeProvider([[completed(tool_call(name="apply_patch"))], [completed()]])
    engine, _ = setup_engine(tmp_path, provider, [read, patch], mode="analysis")
    events = await collect(engine)
    assert {schema["name"] for schema in provider.requests[0].tool_schemas} == {"read_file"}
    assert patch.executions == []
    assert_tool_result(events, "call-1", ok=False, error_code="UNKNOWN_TOOL")


async def test_model_request_limit_is_thirty(tmp_path):
    rounds = [[completed(tool_call(f"call-{n}", request_id=f"request-{n}"))] for n in range(40)]
    provider = FakeProvider(rounds)
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert len(provider.requests) == 30
    assert len(tool.executions) == 30
    assert_terminal(events, "failed", "TURN_LIMIT")


async def test_configured_smaller_request_budget_is_enforced(tmp_path):
    provider = FakeProvider([[completed(tool_call(str(n)))] for n in range(5)])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool], max_turns=2)
    events = await collect(engine)
    assert len(provider.requests) == 2
    assert_terminal(events, "failed", "TURN_LIMIT")


async def test_same_tool_and_arguments_stop_after_three_failures(tmp_path):
    provider = FakeProvider([[completed(tool_call(str(n)))] for n in range(5)])
    tool = RecordingTool(fail=True)
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert len(tool.executions) == 3
    assert len(provider.requests) == 3
    assert_terminal(events, "failed")
    assert any(item.get("code") == "FILE_NOT_FOUND" for item in dictionaries(events))


async def test_different_arguments_do_not_share_failure_counter(tmp_path):
    rounds = [[completed(tool_call(str(n), arguments=json.dumps({"path": f"{n}.py"})))] for n in range(4)]
    provider = FakeProvider([*rounds, [completed()]])
    tool = RecordingTool(fail=True)
    engine, _ = setup_engine(tmp_path, provider, [tool])
    events = await collect(engine)
    assert len(tool.executions) == 4
    assert len(provider.requests) == 5
    assert_terminal(events, "completed")


async def test_pre_cancelled_turn_does_not_request_or_execute(tmp_path):
    provider = FakeProvider([[completed(tool_call())]])
    tool = RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    cancel = asyncio.Event()
    cancel.set()
    events = await collect(engine, cancel)
    assert provider.requests == []
    assert tool.executions == []
    assert_terminal(events, "interrupted")


async def test_tool_execution_waits_for_completed_response(tmp_path):
    delta_seen, finish_response = asyncio.Event(), asyncio.Event()

    class DelayedProvider(FakeProvider):
        async def stream(self, request, cancel_event):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelFrame("text_delta", text="准备读取")
                delta_seen.set()
                await finish_response.wait()
                yield completed(tool_call())
            else:
                yield completed()

    provider, tool = DelayedProvider([]), RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    task = asyncio.create_task(collect(engine))
    try:
        async with asyncio.timeout(3):
            await delta_seen.wait()
        assert tool.executions == []
        finish_response.set()
        events = await task
        assert tool.executions == [({"path": "a.py"}, "call-1")]
        assert_terminal(events, "completed")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_cancellation_during_provider_stream_closes_stream(tmp_path):
    started, closed = asyncio.Event(), asyncio.Event()

    class WaitingProvider(FakeProvider):
        async def stream(self, request, cancel_event):
            self.requests.append(request)
            try:
                started.set()
                yield ModelFrame("text_delta", text="等待中")
                await cancel_event.wait()
                yield completed(tool_call())
            finally:
                closed.set()

    provider, tool = WaitingProvider([]), RecordingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    cancel = asyncio.Event()
    task = asyncio.create_task(collect(engine, cancel))
    try:
        async with asyncio.timeout(3):
            await started.wait()
        cancel.set()
        events = await task
        async with asyncio.timeout(3):
            await closed.wait()
        assert tool.executions == []
        assert len(provider.requests) == 1
        assert_terminal(events, "interrupted")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)


async def test_cancellation_during_tool_prevents_next_tool_and_request(tmp_path):
    started = asyncio.Event()

    class WaitingTool(RecordingTool):
        async def execute(self, arguments, context, cancel_event):
            self.executions.append((dict(arguments), context.call_id))
            started.set()
            await cancel_event.wait()
            return ToolResult.fail(context.call_id, "COMMAND_CANCELLED", "测试取消")

    provider = FakeProvider([[completed(tool_call(), tool_call("call-2"))], [completed()]])
    tool = WaitingTool()
    engine, _ = setup_engine(tmp_path, provider, [tool])
    cancel = asyncio.Event()
    task = asyncio.create_task(collect(engine, cancel))
    try:
        async with asyncio.timeout(3):
            await started.wait()
        cancel.set()
        events = await task
        assert tool.executions == [({"path": "a.py"}, "call-1")]
        assert len(provider.requests) == 1
        assert_terminal(events, "interrupted")
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
