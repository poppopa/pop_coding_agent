"""验收依据：技术方案§上下文与会话恢复；预算、仓库约定、完整工具协议单元。"""
import asyncio
from copy import deepcopy

import pytest

from coding_agent.config import AgentConfig
from coding_agent.context import ContextLimit, ContextManager
from coding_agent.domain import ModelFrame
from coding_agent.engine import AgentEngine
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.tools.base import ToolContext
from coding_agent.domain import ToolCall, ToolResult
from coding_agent.tools.base import Tool, ToolInput


def manager(tmp_path):
    return ContextManager(tmp_path, AgentConfig(model="test-model", state_dir=tmp_path / "state"))


def history_round(number):
    call_id = f"call-{number}"
    return [
        {"kind": "message", "role": "user", "content": f"目标{number}"},
        {"kind": "response", "items": [{"type": "function_call", "call_id": call_id, "name": "read_file", "arguments": '{"path":"a.py"}'}]},
        {"kind": "tool_result", "call_id": call_id, "result": {"call_id": call_id, "ok": True, "data": {"text": f"结果{number}"}, "error": None, "truncated": False}},
    ]


def test_agents_file_has_priority_over_claude(tmp_path):
    (tmp_path / "AGENTS.md").write_text("测试约定甲", encoding="utf-8")
    (tmp_path / "CLAUDE.md").write_text("测试约定乙", encoding="utf-8")
    instructions = manager(tmp_path).instructions()
    assert "测试约定甲" in instructions
    assert "测试约定乙" not in instructions


def test_claude_is_loaded_when_agents_is_absent(tmp_path):
    (tmp_path / "CLAUDE.md").write_text("测试备用约定", encoding="utf-8")
    assert "测试备用约定" in manager(tmp_path).instructions()


def test_model_request_retains_complete_protocol_and_output_budget(tmp_path):
    history = history_round(1)
    schemas = ({"type": "function", "name": "read_file"},)
    original = deepcopy(history)
    request = manager(tmp_path).build_request(history, schemas)
    assert request.history == tuple(history)
    assert request.tool_schemas == schemas
    assert request.max_output_tokens == 4096
    assert history == original
    assert request.instructions


def test_compaction_starts_above_seventy_percent(tmp_path):
    context = manager(tmp_path)
    assert context.needs_compaction([{"content": "x" * 1000}], ()) is False
    assert context.needs_compaction([{"content": "x" * 24000}], ()) is True


def test_over_budget_request_returns_context_limit(tmp_path):
    history = [{"kind": "message", "role": "user", "content": "x" * 40000}]
    original = deepcopy(history)
    with pytest.raises(ContextLimit, match="CONTEXT_LIMIT"):
        manager(tmp_path).build_request(history, ())
    assert history == original


def test_compaction_split_keeps_last_two_complete_tool_rounds(tmp_path):
    rounds = [history_round(i) for i in range(4)]
    history = sum(rounds, []) + [{"kind": "message", "role": "user", "content": "当前目标"}]
    original = deepcopy(history)
    old, recent = manager(tmp_path).compaction_split(history)
    assert old == rounds[0] + rounds[1]
    assert recent == rounds[2] + rounds[3] + history[-1:]
    assert history == original
    call_ids = [x["items"][0]["call_id"] for x in recent if x.get("kind") == "response"]
    result_ids = [x["call_id"] for x in recent if x.get("kind") == "tool_result"]
    assert call_ids == result_ids == ["call-2", "call-3"]


@pytest.mark.parametrize("count", [1, 2])
def test_recent_rounds_are_not_discarded_for_compaction(tmp_path, count):
    history = sum([history_round(i) for i in range(count)], [])
    with pytest.raises(ContextLimit, match="CONTEXT_LIMIT"):
        manager(tmp_path).compaction_split(history)


async def test_failed_summary_is_attempted_once_and_history_is_preserved(tmp_path):
    class Provider:
        def __init__(self):
            self.requests = []

        async def stream(self, request, cancel_event):
            self.requests.append(request)
            yield ModelFrame("response_finished", status="incomplete")

    class Store:
        def __init__(self):
            self.events = []

        async def append(self, event):
            self.events.append(event)

    async def emit(kind, payload):
        pass

    provider, store = Provider(), Store()
    config = AgentConfig(model="test-model", state_dir=tmp_path / "state")
    context = ToolContext(tmp_path, "", None, Policy(), emit, Redactor(), tmp_path / "state")
    engine = AgentEngine(provider, [], store, context, config)
    engine.history = sum([history_round(i) for i in range(4)], [])
    original = deepcopy(engine.history)
    with pytest.raises(ContextLimit, match="CONTEXT_LIMIT"):
        await engine.compact(asyncio.Event())
    assert len(provider.requests) == 1
    assert provider.requests[0].tool_schemas == ()
    assert engine.history == original
    assert store.events == []


def test_path_rules_follow_root_to_child_and_agents_priority(tmp_path):
    (tmp_path / "AGENTS.md").write_text("根规则", encoding="utf-8")
    parent = tmp_path / "pkg"
    child = parent / "nested"
    child.mkdir(parents=True)
    (parent / "AGENTS.md").write_text("父规则", encoding="utf-8")
    (parent / "CLAUDE.md").write_text("父备用不选", encoding="utf-8")
    (child / "CLAUDE.md").write_text("子规则", encoding="utf-8")
    context = manager(tmp_path)
    assert context.load_path_rules("pkg/nested/a.py") is True
    instructions = context.instructions()
    assert instructions.index("根规则") < instructions.index("父规则") < instructions.index("子规则")
    assert "父备用不选" not in instructions
    assert context.load_path_rules("pkg/nested/a.py") is False


def test_path_rules_cannot_load_external_symlink(tmp_path):
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    (outside / "AGENTS.md").write_text("外部规则不加载", encoding="utf-8")
    (tmp_path / "alias").symlink_to(outside, target_is_directory=True)
    context = manager(tmp_path)
    assert context.load_path_rules("alias/a.py") is False
    assert "外部规则不加载" not in context.instructions()


@pytest.mark.parametrize("name", ["read_file", "apply_patch"])
async def test_new_child_rules_are_seen_before_patch_is_executed(tmp_path, name):
    child = tmp_path / "pkg"
    child.mkdir()
    (child / "AGENTS.md").write_text("子目录测试规则", encoding="utf-8")

    class Arguments(ToolInput):
        path: str

    class RecordingTool(Tool):
        input_type = Arguments
        description = "记录约定生效时机"
        executions = 0

        async def execute(self, arguments, context, cancel_event):
            self.executions += 1
            return ToolResult(context.call_id, True, {"path": arguments["path"]})

    class Provider:
        def __init__(self):
            self.requests = []

        async def stream(self, request, cancel_event):
            self.requests.append(request)
            if len(self.requests) == 1:
                yield ModelFrame("response_finished", status="completed", calls=(ToolCall("r1", "c1", name, '{"path":"pkg/a.py"}'),))
            else:
                yield ModelFrame("response_finished", status="completed")

    class Store:
        events = None

        async def append(self, event):
            pass

    async def emit(kind, payload):
        pass

    tool = RecordingTool()
    tool.name, tool.mutating = name, name == "apply_patch"
    provider = Provider()
    context = ToolContext(tmp_path, "", None, Policy(), emit, Redactor(), tmp_path / "state")
    engine = AgentEngine(provider, [tool], Store(), context, AgentConfig(model="test-model", state_dir=tmp_path / "state"))
    events = [event async for event in engine.run_turn("读取或修改子目录", asyncio.Event())]
    assert "子目录测试规则" not in provider.requests[0].instructions
    assert "子目录测试规则" in provider.requests[1].instructions
    if name == "apply_patch":
        assert tool.executions == 0
        results = [event.payload["result"] for event in events if event.type == "tool_finished"]
        assert results[0]["error"]["code"] == "CONTEXT_UPDATED"
        assert not (child / "a.py").exists()
    else:
        assert tool.executions == 1


def test_repository_rules_redact_known_placeholder(tmp_path):
    (tmp_path / "AGENTS.md").write_text("规则 <your-test-key>")
    child = tmp_path / "pkg"
    child.mkdir()
    (child / "AGENTS.md").write_text("子规则 <your-test-key>")
    context = ContextManager(tmp_path, AgentConfig(model="test-model"), Redactor(["<your-test-key>"]))
    context.load_path_rules("pkg/a.py")
    assert "<your-test-key>" not in context.instructions()
    assert context.instructions().count("<REDACTED>") == 2
