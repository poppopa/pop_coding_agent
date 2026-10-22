"""验收依据：技术方案§上下文与会话恢复、核心契约与状态机、FR-006/008。"""
import asyncio
from dataclasses import asdict
import json
import stat

import pytest

from coding_agent.config import AgentConfig
from coding_agent.domain import Event, ModelFrame
from coding_agent.engine import AgentEngine
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.storage.session import JsonlSessionStore, SessionError
from coding_agent.tools.base import Tool, ToolContext, ToolInput


@pytest.fixture
def store(tmp_path):
    instance = JsonlSessionStore(tmp_path / "state", tmp_path, "testsession")
    try:
        yield instance
    finally:
        instance.close()


def event(store, seq, kind="user_message", payload=None):
    return Event(1, store.session_id, "testturn", seq, kind, payload or {"text": "测试任务"})


async def test_append_load_and_resume_preserve_sequence(store, tmp_path):
    first, second = event(store, 1), event(store, 2, "turn_finished", {"status": "completed"})
    await store.append(first)
    await store.append(second)
    assert await store.load(store.session_id) == [first, second]
    assert [json.loads(x)["seq"] for x in store.path.read_text().splitlines()] == [1, 2]
    store.close()
    resumed = JsonlSessionStore(tmp_path / "state", tmp_path, "testsession")
    try:
        assert resumed.seq == 2
        await resumed.append(event(resumed, 3))
        assert [x.seq for x in await resumed.load()] == [1, 2, 3]
    finally:
        resumed.close()


async def test_append_rejects_non_contiguous_sequence_without_write(store):
    await store.append(event(store, 1))
    before = store.path.read_bytes()
    with pytest.raises(SessionError, match="SESSION_CORRUPTED"):
        await store.append(event(store, 3))
    assert store.path.read_bytes() == before
    assert store.seq == 1


def test_lock_conflict_then_reacquisition(store, tmp_path):
    with pytest.raises(SessionError, match="SESSION_LOCKED"):
        JsonlSessionStore(tmp_path / "state", tmp_path, "testsession")
    store.close()
    resumed = JsonlSessionStore(tmp_path / "state", tmp_path, "testsession")
    resumed.close()


def test_resume_rejects_other_repository(store, tmp_path):
    store.close()
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SessionError):
        JsonlSessionStore(tmp_path / "state", other, "testsession")
    resumed = JsonlSessionStore(tmp_path / "state", tmp_path, "testsession")
    resumed.close()


async def test_corrupted_tail_preserves_backup_and_last_complete_event(store, tmp_path):
    first = event(store, 1)
    await store.append(first)
    valid = store.path.read_bytes()
    broken = valid + b'{"schema_version":1'
    store.path.write_bytes(broken)
    store.close()
    resumed = JsonlSessionStore(tmp_path / "state", tmp_path, "testsession")
    try:
        assert await resumed.load() == [first]
        assert resumed.path.read_bytes() == valid
        assert resumed.path.with_suffix(".corrupted.jsonl").read_bytes() == broken
    finally:
        resumed.close()


async def test_corrupted_middle_stops_recovery_without_modification(store, tmp_path):
    first = event(store, 1)
    await store.append(first)
    damaged = store.path.read_bytes() + b'not-json\n' + json.dumps(asdict(event(store, 2))).encode() + b'\n'
    store.path.write_bytes(damaged)
    path = store.path
    store.close()
    with pytest.raises(SessionError, match="SESSION_CORRUPTED"):
        JsonlSessionStore(tmp_path / "state", tmp_path, "testsession")
    assert path.read_bytes() == damaged


def test_session_files_are_private(store):
    assert stat.S_IMODE(store.directory.stat().st_mode) == 0o700
    assert stat.S_IMODE(store.path.stat().st_mode) == 0o600
    assert stat.S_IMODE((store.directory / "metadata.json").stat().st_mode) == 0o600


async def test_saved_events_and_loaded_history_redact_placeholder(tmp_path):
    placeholder = "<your-test-key>"
    store = JsonlSessionStore(tmp_path / "state", tmp_path, "redacted", Redactor([placeholder]))
    try:
        await store.append(event(store, 1, payload={"text": "测试占位符 " + placeholder, "api_key": placeholder}))
        assert placeholder not in store.path.read_text()
        assert "<REDACTED>" in store.path.read_text()
        assert placeholder not in json.dumps(store.history())
        assert (await store.load())[0].payload["api_key"] == "<REDACTED>"
    finally:
        store.close()


class NoArguments(ToolInput):
    pass


class CommandCounter(Tool):
    name = "run_command"
    description = "检测恢复期间是否自动执行"
    input_type = NoArguments
    mutating = True

    def __init__(self):
        self.executions = 0

    async def execute(self, arguments, context, cancel_event):
        self.executions += 1
        raise AssertionError("恢复不能自动执行命令")


@pytest.mark.parametrize("started", [False, True], ids=["pending", "started"])
async def test_restored_pending_call_is_closed_without_reexecution(store, tmp_path, started):
    call = {"type": "function_call", "call_id": "call-pending", "name": "run_command", "arguments": '{"argv":["python","-V"]}'}
    await store.append(event(store, 1, "model_response", {"items": [call]}))
    if started:
        await store.append(event(store, 2, "tool_started", {"call_id": "call-pending", "name": "run_command"}))
    path = store.path
    store.close()
    resumed = JsonlSessionStore(tmp_path / "state", tmp_path, "testsession")

    class Provider:
        def __init__(self):
            self.requests = []

        async def stream(self, request, cancel_event):
            self.requests.append(request)
            yield ModelFrame("response_finished", status="completed")

    async def emit(kind, payload):
        pass

    provider, tool = Provider(), CommandCounter()
    context = ToolContext(tmp_path, "", None, Policy(), emit, Redactor(), tmp_path / "state")
    engine = AgentEngine(provider, [tool], resumed, context, AgentConfig(model="test-model", state_dir=tmp_path / "state"))
    try:
        events = [x async for x in engine.run_turn("继续说明任务", asyncio.Event())]
        assert tool.executions == 0
        pending = [x for x in provider.requests[0].history if x.get("kind") == "tool_result" and x.get("call_id") == "call-pending"]
        assert len(pending) == 1
        assert pending[0]["result"]["ok"] is False
        assert pending[0]["result"]["error"]["code"] == "COMMAND_CANCELLED"
        assert events[-1].payload["status"] == "completed"
        assert "tool_started" not in [x.type for x in events]
    finally:
        resumed.close()


async def test_completed_call_reuses_saved_result(store):
    call = {"type": "function_call", "call_id": "call-done", "name": "read_file", "arguments": '{"path":"a.py"}'}
    saved = {"call_id": "call-done", "ok": True, "data": {"text": "saved"}, "error": None, "truncated": False}
    await store.append(event(store, 1, "model_response", {"items": [call]}))
    await store.append(event(store, 2, "tool_finished", {"call_id": "call-done", "result": saved}))
    results = [x for x in store.history() if x.get("kind") == "tool_result"]
    assert results == [{"kind": "tool_result", "call_id": "call-done", "result": saved}]


async def test_repeated_call_id_closes_only_later_pending_response(store):
    call = {"type": "function_call", "call_id": "repeated", "name": "read_file", "arguments": '{"path":"a.py"}'}
    result = {"call_id": "repeated", "ok": True, "data": {"text": "first result"}, "error": None, "truncated": False}
    await store.append(event(store, 1, "model_response", {"items": [call]}))
    await store.append(event(store, 2, "tool_finished", {"call_id": "repeated", "result": result}))
    await store.append(event(store, 3, "model_response", {"items": [call]}))
    history = store.history()
    responses = [i for i, item in enumerate(history) if item["kind"] == "response"]
    assert history[responses[0] + 1]["result"] == result
    assert history[responses[1] + 1]["result"]["ok"] is False
    assert history[responses[1] + 1]["result"]["error"]["code"] == "COMMAND_CANCELLED"
    assert history[responses[1] + 1]["call_id"] == "repeated"


async def test_authorization_failure_closes_pending_and_allows_next_turn(store, tmp_path):
    from coding_agent.domain import ToolCall
    from coding_agent.tools.command import RunCommandTool

    class Provider:
        def __init__(self):
            self.requests = []

        async def stream(self, request, cancel_event):
            self.requests.append(request)
            if len(self.requests) == 1:
                calls = (ToolCall("r1", "c1", "run_command", '{"argv":["python","-V"]}'), ToolCall("r1", "c2", "run_command", '{"argv":["python","-V"]}'))
                yield ModelFrame("response_finished", status="completed", calls=calls)
            else:
                yield ModelFrame("response_finished", status="completed")

    async def emit(kind, payload):
        pass

    provider = Provider()
    context = ToolContext(tmp_path, "", None, Policy(), emit, Redactor(), tmp_path / "state")
    engine = AgentEngine(provider, [RunCommandTool()], store, context, AgentConfig(model="test-model", state_dir=tmp_path / "state"))
    first = [e async for e in engine.run_turn("未授权命令", asyncio.Event())]
    assert first[-1].payload["error"]["code"] == "APPROVAL_REQUIRED"
    second = [e async for e in engine.run_turn("继续分析", asyncio.Event())]
    assert second[-1].payload["status"] == "completed"
    results = [h for h in provider.requests[1].history if h.get("kind") == "tool_result"]
    assert [h["call_id"] for h in results] == ["c2", "c1"] or [h["call_id"] for h in results] == ["c1", "c2"]
    assert {h["call_id"]: h["result"]["error"]["code"] for h in results} == {"c1": "APPROVAL_REQUIRED", "c2": "COMMAND_CANCELLED"}
    assert not any(e.type == "tool_started" for e in first + second)
