"""审查后新增修复验收：仅本消息授权的新增行为，旧单元历史保持不变。

依据技术方案§Key隔离、固定当前目标、稳定错误类型及FR010；2026审查修复派发。
"""
import asyncio
from dataclasses import asdict
import json
import sys

import pytest

from coding_agent.config import AgentConfig
from coding_agent.context import ContextManager
from coding_agent.domain import ModelFrame, ToolCall, ToolResult
from coding_agent.engine import AgentEngine
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.tools.base import Tool, ToolInput
from coding_agent.tools.command import RunCommandTool
from coding_agent.tools.patch import ApplyPatchTool


@pytest.mark.parametrize("configured", [[], ["EXTRA_TEST_CREDENTIAL"]], ids=["empty", "append-only"])
def test_model_credential_filter_cannot_remove_required_names(configured):
    config = AgentConfig(model="test-model", filtered_env_names=configured)
    assert {"ARK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"} <= set(config.filtered_env_names)
    if configured:
        assert "EXTRA_TEST_CREDENTIAL" in config.filtered_env_names


def test_compaction_keeps_original_goal_before_latest_two_tool_rounds(tmp_path):
    goal = {"kind": "message", "role": "user", "content": "用户当前目标原文：只改这里"}
    rounds = []
    for i in range(4):
        rounds.extend([{"kind": "response", "items": [{"type": "function_call", "call_id": str(i)}]}, {"kind": "tool_result", "call_id": str(i), "result": {"call_id": str(i), "ok": True}}])
    history = [goal, *rounds]
    old, recent = ContextManager(tmp_path, AgentConfig(model="test-model")).compaction_split(history)
    assert goal not in old
    assert recent == [goal, *rounds[-4:]]
    assert old == rounds[:-4]
    assert history == [goal, *rounds]


@pytest.mark.parametrize("code", ["PROVIDER_UNAVAILABLE", "PROVIDER_PROTOCOL_ERROR"])
async def test_provider_failed_frame_preserves_error_identity(file_context, code):
    class Provider:
        async def stream(self, request, cancel_event):
            yield ModelFrame("response_finished", status="failed", text=f"{code}: 测试诊断")

    class Store:
        async def append(self, event):
            pass

    engine = AgentEngine(Provider(), [], Store(), file_context, AgentConfig(model="test-model"))
    events = [e async for e in engine.run_turn("测试失败", asyncio.Event())]
    assert events[-1].payload["error"] == {"code": code, "message": f"{code}: 测试诊断"}
    assert not any(e.type == "tool_started" for e in events)


@pytest.mark.parametrize("verified", [False, True], ids=["not-run", "failed-verification"])
async def test_final_summary_comes_from_actual_actions(file_context, verified):
    argv = [sys.executable, "-c", "import sys; sys.exit(7)"]
    file_context.policy.approve_once(argv, file_context.root)
    class Provider:
        def __init__(self):
            self.step = 0

        async def stream(self, request, cancel_event):
            self.step += 1
            if self.step == 1:
                yield ModelFrame("response_finished", status="completed", calls=(ToolCall("r1", "c1", "apply_patch", '{"path":"a.py","operation":"create","content":"created"}'),))
            elif verified and self.step == 2:
                yield ModelFrame("response_finished", status="completed", calls=(ToolCall("r2", "c2", "run_command", json.dumps({"argv": argv, "purpose": "verify"})),))
            else:
                yield ModelFrame("response_finished", status="completed")

    class Store:
        async def append(self, event):
            pass

    engine = AgentEngine(Provider(), [ApplyPatchTool(), RunCommandTool()], Store(), file_context, AgentConfig(model="test-model"))
    events = [e async for e in engine.run_turn("只改不验证", asyncio.Event())]
    assert events[-1].payload["changes"] == [{"path": "a.py", "operation": "create", "recorded": True}]
    assert file_context.root.joinpath("a.py").read_text() == "created"
    assert events[-1].payload["verifications"] == ([{"exit_code": 7, "passed": False, "cancelled": False, "timed_out": False}] if verified else [])
