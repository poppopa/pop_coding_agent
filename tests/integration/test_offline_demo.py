"""实现计划§综合验收：执行真实离线demo四轮工具循环，不请求模型。"""
from pathlib import Path
import runpy

import pytest


async def test_offline_demo_executes_read_patch_verify_then_finishes(monkeypatch, capsys):
    for name in ("ARK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    demo = runpy.run_path(str(Path(__file__).resolve().parents[2] / "examples/offline_demo.py"))
    events = []

    class Renderer:
        def handle(self, event):
            events.append(event)

    demo["main"].__globals__["TerminalRenderer"] = Renderer
    await demo["main"]()
    results = [e.payload for e in events if e.type == "tool_finished"]
    assert [r["name"] for r in results] == ["read_file", "apply_patch", "run_command"]
    assert all(r["result"]["ok"] for r in results)
    assert results[-1]["result"]["data"]["exit_code"] == 0
    assert "passed" in results[-1]["result"]["data"]["stdout"]
    assert events[-1].payload["status"] == "completed"
    assert events[-1].payload["changes"] == [{"path": "calculator.py", "operation": "update", "recorded": True}]
    assert events[-1].payload["verifications"] == [{"exit_code": 0, "passed": True, "cancelled": False, "timed_out": False}]
    assert "if values else 0" in capsys.readouterr().out
