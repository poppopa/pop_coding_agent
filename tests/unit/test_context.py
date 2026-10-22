"""验收依据：技术方案§上下文与会话恢复；预算、仓库约定、完整工具协议单元。"""
import asyncio
from copy import deepcopy
import json

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


async def test_failed_summary_is_attempted_once_and_original_recent_history_is_preserved(tmp_path):
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
    # 摘要恢复契约修订：失败后允许本地事实整理，最近两轮仍保持原文。
    assert await engine.compact(asyncio.Event()) is True
    assert len(provider.requests) == 1
    assert provider.requests[0].tool_schemas == ()
    assert engine.history[1:] == original[-6:]
    assert store.events[0].type == "context_compacted"
    assert store.events[0].payload["source"] == "local"
    assert "目标0" in engine.history[0]["content"]


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


# 本轮新增依据：CLI 编程助手上下文修复契约§上下文行为。
def file_history(*, count=4, response_count=1, body_size=5500):
    history = [{"kind": "message", "role": "user", "content": "解释一下当前这个项目，保留原文目标"}]
    for group in range(response_count):
        items, results = [], []
        for number in range(group, count, response_count):
            call_id = f"file-{number}"
            items.append({"type": "function_call", "call_id": call_id, "name": "read_file",
                          "arguments": json.dumps({"path": f"{number}.py"}), "opaque": {"keep": number}})
            text = chr(97 + number) * body_size + "\n"
            results.append({"kind": "tool_result", "call_id": call_id, "result": {
                "call_id": call_id, "ok": True, "error": None, "truncated": True,
                "data": {"path": f"{number}.py", "text": text, "start_line": 1, "end_line": 1,
                         "sha256": str(number) * 64, "size_bytes": body_size + 1,
                         "lines": [{"line_number": 1, "text": text}]}}})
        history.extend([{"kind": "response", "items": items}, *results])
    return history


def without_line_copies(history):
    expected = deepcopy(history)
    for entry in expected:
        if entry.get("kind") == "tool_result":
            entry["result"]["data"].pop("lines", None)
    return expected


def utf8_request_size(context, history, schemas=()):
    # 契约定义保守UTF-8字节估算；不调用被测estimate来生成期望。
    return len(json.dumps([context.instructions(), history, schemas], ensure_ascii=False).encode("utf-8")) + 512


class ContextProvider:
    def __init__(self, *, summary="旧执行事实摘要"):
        self.requests = []
        self.summary = summary

    async def stream(self, request, cancel_event):
        self.requests.append(request)
        # 测试保留真实异步边界，模型是唯一被替换的执行依赖。
        await asyncio.sleep(0)
        yield ModelFrame("text_delta", text=self.summary)
        yield ModelFrame("response_finished", status="completed")


class ContextStore:
    def __init__(self):
        self.events = []

    async def append(self, event):
        self.events.append(event)


def context_engine(file_context, provider, **overrides):
    config = AgentConfig(model="test-model", state_dir=file_context.state_dir, **overrides)
    store = ContextStore()
    return AgentEngine(provider, [], store, file_context, config), store


@pytest.mark.parametrize("response_count", [1, 2], ids=["one-response-four-files", "two-responses-four-files"])
async def test_soft_budget_with_four_restored_files_continues_without_summary(file_context, response_count):
    provider = ContextProvider(summary="项目解释完成")
    engine, store = context_engine(file_context, provider)
    engine.history = file_history(response_count=response_count)
    original = deepcopy(engine.history)
    prompt = "继续解释项目，当前目标原文不可替换"
    expected = without_line_copies(original + [{"kind": "message", "role": "user", "content": prompt}])
    used = utf8_request_size(engine.manager, expected)
    assert 32_000 * .7 <= used <= 32_000 - 4096 - 2048
    events = [event async for event in engine.run_turn(prompt, asyncio.Event())]
    assert events[-1].payload["status"] == "completed"
    assert len(provider.requests) == 1
    assert provider.requests[0].history == tuple(expected)
    assert engine.history[:len(original)] == original
    assert not any(event.type == "context_compacted" for event in store.events)
    assert "项目解释完成" == "".join(event.payload["text"] for event in events if event.type == "text_delta")


def test_model_projection_preserves_protocol_and_does_not_alias_original(tmp_path):
    history = file_history(body_size=12)
    original = deepcopy(history)
    schemas = ({"type": "function", "name": "read_file", "description": "中文工具说明"},)
    context = manager(tmp_path)
    request = context.build_request(history, schemas)
    expected = without_line_copies(original)
    assert request.history == tuple(expected)
    assert request.tool_schemas == schemas
    assert context.request_size(history, schemas) == utf8_request_size(context, expected, schemas)
    assert history == original
    # 修改请求侧对象不得污染恢复历史；调用、文件指纹、错误字段都由整对象比较保护。
    request.history[1]["items"][0]["opaque"]["keep"] = "模型侧修改"
    request.history[2]["result"]["data"]["text"] = "模型侧修改"
    assert history == original


@pytest.mark.parametrize("case", [
    "different-text", "noncontinuous-lines", "missing-start", "invalid-line", "other-tool", "unlinked-result",
], ids=["different-text", "noncontinuous-lines", "missing-start", "invalid-line", "other-tool", "unlinked-result"])
def test_uncertain_or_non_file_body_is_retained_in_model_request(tmp_path, case):
    history = file_history(count=1, body_size=12)
    data = history[-1]["result"]["data"]
    if case == "different-text":
        data["lines"][0]["text"] = "不同正文\n"
    elif case == "noncontinuous-lines":
        data["lines"][0]["line_number"] = 2
    elif case == "missing-start":
        del data["start_line"]
    elif case == "invalid-line":
        data["lines"] = ["不是行字典"]
    elif case == "other-tool":
        history[1]["items"][0]["name"] = "search_text"
    else:
        history[-1]["call_id"] = "没有关联的结果"
    original = deepcopy(history)
    assert manager(tmp_path).build_request(history, ()).history == tuple(original)
    assert history == original


@pytest.mark.parametrize("delta", [-1, 0, 1], ids=["below-hard-budget", "equal-hard-budget", "above-hard-budget"])
def test_hard_input_budget_boundary_counts_utf8_instructions_and_schemas(tmp_path, delta):
    history = [{"kind": "message", "role": "user", "content": "中文原文" * 1200}]
    schemas = ({"type": "function", "name": "read_file", "description": "工具协议内容"},)
    used = utf8_request_size(manager(tmp_path), history, schemas)
    config = AgentConfig(model="test-model", context_tokens=used + 4096 + 2048 - delta)
    context = ContextManager(tmp_path, config)
    original = deepcopy(history)
    assert context.input_budget == used - delta
    assert context.request_size(history, schemas) == used
    if delta > 0:
        with pytest.raises(ContextLimit) as caught:
            context.build_request(history, schemas)
        message = str(caught.value)
        assert "CONTEXT_LIMIT" in message and str(used) in message and str(used - delta) in message
        assert "估算" in message and "预算" in message and "读取" in message
    else:
        assert context.build_request(history, schemas).history == tuple(history)
    assert history == original


async def test_over_hard_budget_without_old_rounds_never_requests_model(file_context):
    provider = ContextProvider()
    engine, store = context_engine(file_context, provider)
    prompt = "超出预算的用户原文" * 4000
    events = [event async for event in engine.run_turn(prompt, asyncio.Event())]
    assert provider.requests == []
    assert events[-1].payload["status"] == "failed"
    diagnostic = events[-1].payload["error"]
    assert diagnostic["code"] == "CONTEXT_LIMIT"
    assert str(32_000 - 4096 - 2048) in diagnostic["message"]
    assert "估算" in diagnostic["message"] and "读取" in diagnostic["message"]
    assert engine.history[0]["content"] == prompt
    assert not any(event.type == "context_compacted" for event in store.events)


@pytest.mark.parametrize("response_count", [0, 1, 2], ids=["empty", "one-response", "two-responses"])
async def test_manual_compact_without_old_rounds_is_noop(file_context, response_count):
    provider = ContextProvider()
    engine, store = context_engine(file_context, provider)
    engine.history = [] if response_count == 0 else file_history(count=2, response_count=response_count, body_size=12)
    original = deepcopy(engine.history)
    assert await engine.compact(asyncio.Event()) is False
    assert provider.requests == [] and store.events == []
    assert engine.history == original


async def test_auto_compacts_when_hard_budget_is_below_soft_threshold(file_context):
    provider = ContextProvider(summary="摘要事实")
    engine, store = context_engine(file_context, provider, max_output_tokens=20000)
    engine.history = file_history(count=3, response_count=3, body_size=3250)
    original = deepcopy(engine.history)
    prompt = "保留用户当前目标"
    before = without_line_copies(original + [{"kind": "message", "role": "user", "content": prompt}])
    used = utf8_request_size(engine.manager, before)
    assert 32_000 - 20000 - 2048 < used < 32_000 * .7
    assert engine.manager.needs_compaction(original + [{"kind": "message", "role": "user", "content": prompt}], ())
    events = [event async for event in engine.run_turn(prompt, asyncio.Event())]
    assert events[-1].payload["status"] == "completed"
    assert len(provider.requests) == 2
    assert provider.requests[0].tool_schemas == ()
    assert "lines" not in provider.requests[0].history[0]["content"]
    normal_history = provider.requests[1].history
    assert normal_history[-1]["content"] == prompt
    assert [entry for entry in normal_history if entry.get("kind") == "response"] == [original[3], original[5]]
    assert [entry["call_id"] for entry in normal_history if entry.get("kind") == "tool_result"] == ["file-1", "file-2"]
    assert any(event.type == "context_compacted" for event in store.events)
