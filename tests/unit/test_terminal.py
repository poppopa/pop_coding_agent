"""技术方案§CLI与配置行为；仅测试键绑定与本地命令，不启动真实终端。"""
from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.keys import Keys
from rich.console import Console

from coding_agent.domain import Event, ModelFrame
from coding_agent.config import AgentConfig
from coding_agent.engine import AgentEngine
from coding_agent.policy import Policy
from coding_agent.security import Redactor
from coding_agent.ui import terminal
from coding_agent.ui.renderer import TerminalRenderer


def fake_engine(tmp_path):
    return SimpleNamespace(store=SimpleNamespace(directory=tmp_path), context=SimpleNamespace(policy=Policy(), redactor=Redactor(), snapshots=SimpleNamespace(diff=lambda: "")), session_id="test-session")


async def test_keyboard_bindings_submit_newline_clear_and_empty_eof(tmp_path, monkeypatch):
    captured = {}

    class Prompt:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        async def prompt_async(self, text):
            raise EOFError

    monkeypatch.setattr(terminal, "PromptSession", Prompt)
    monkeypatch.setattr(terminal, "patch_stdout", lambda **kwargs: nullcontext())
    assert await terminal.interactive(fake_engine(tmp_path), SimpleNamespace(), Console(file=StringIO())) == 0
    assert captured["multiline"] is True
    bindings = captured["key_bindings"]
    buffer = SimpleNamespace(text="", validate_and_handle=Mock(), insert_text=Mock(), reset=Mock())
    app = SimpleNamespace(exit=Mock())
    event = SimpleNamespace(current_buffer=buffer, app=app)
    for keys, check in [((Keys.ControlM,), buffer.validate_and_handle), ((Keys.Escape, Keys.ControlM), buffer.insert_text), ((Keys.ControlC,), buffer.reset)]:
        bindings.get_bindings_for_keys(keys)[-1].handler(event)
        check.assert_called_once()
    buffer.insert_text.assert_called_once_with("\n")
    bindings.get_bindings_for_keys((Keys.ControlD,))[-1].handler(event)
    assert isinstance(app.exit.call_args.kwargs["exception"], EOFError)
    app.exit.reset_mock()
    buffer.text = "还未提交"
    bindings.get_bindings_for_keys((Keys.ControlD,))[-1].handler(event)
    app.exit.assert_not_called()


async def test_slash_commands_are_local_and_mode_changes_policy(tmp_path, monkeypatch):
    prompts = iter(["/help", "/mode analysis", "/unknown", "/exit"])

    class Prompt:
        def __init__(self, **kwargs):
            pass

        async def prompt_async(self, text):
            return next(prompts)

    monkeypatch.setattr(terminal, "PromptSession", Prompt)
    monkeypatch.setattr(terminal, "patch_stdout", lambda **kwargs: nullcontext())
    engine = fake_engine(tmp_path)
    engine.run_turn = Mock(side_effect=AssertionError("斜杠命令不发送模型"))
    output = StringIO()
    assert await terminal.interactive(engine, SimpleNamespace(), Console(file=output)) == 0
    engine.run_turn.assert_not_called()
    assert engine.context.policy.mode == "analysis"
    assert "Alt+Enter" in output.getvalue()
    assert "未知本地命令" in output.getvalue()


def test_stream_text_is_literal_without_rich_markup():
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, force_terminal=False, width=100))
    renderer.handle(Event(1, "s", "t", 1, "text_delta", {"text": "[red]普通文本[/red]"}))
    assert output.getvalue() == "[red]普通文本[/red]"


def test_terminal_interruption_reports_preserved_changes():
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, force_terminal=False, width=100))
    renderer.handle(Event(1, "s", "t", 1, "turn_finished", {"status": "interrupted"}))
    assert "已发生的修改保留" in output.getvalue()


def test_summary_shows_unverified_change_and_true_pass_count():
    output = StringIO()
    renderer = TerminalRenderer(Console(file=output, force_terminal=False, width=100))
    renderer.handle(Event(1, "s", "t", 1, "turn_finished", {"status": "completed", "changes": [{"path": "a.py", "operation": "update", "recorded": True}], "verifications": []}))
    assert "a.py" in output.getvalue()
    assert "本轮未执行验证命令" in output.getvalue()
    output.seek(0)
    output.truncate()
    renderer.handle(Event(1, "s", "t", 2, "turn_finished", {"status": "completed", "changes": [], "verifications": [{"exit_code": 0, "passed": True, "cancelled": False, "timed_out": False}, {"exit_code": 1, "passed": False, "cancelled": False, "timed_out": False}]}))
    assert "1/2 条命令通过" in output.getvalue()


async def test_context_and_noop_compact_show_input_budget_without_model_request(file_context, tmp_path, monkeypatch):
    # 修复契约§上下文行为：终端显示真实输入预算并区分无需整理。
    prompts = iter(["/context", "/compact", "/exit"])

    class Prompt:
        def __init__(self, **kwargs):
            pass

        async def prompt_async(self, text):
            return next(prompts)

    class Store:
        directory = tmp_path

        async def append(self, event):
            raise AssertionError("本地查看及无需整理不新增事件")

    class Provider:
        async def stream(self, request, cancel_event):
            raise AssertionError("无旧轮次时不请求模型")
            yield

    config = AgentConfig(model="test-model", state_dir=file_context.state_dir)
    engine = AgentEngine(Provider(), [], Store(), file_context, config)
    engine.history = [{"kind": "message", "role": "user", "content": "当前目标原文"}]
    monkeypatch.setattr(terminal, "PromptSession", Prompt)
    monkeypatch.setattr(terminal, "patch_stdout", lambda **kwargs: nullcontext())
    output = StringIO()
    assert await terminal.interactive(engine, config, Console(file=output, width=160)) == 0
    text = output.getvalue()
    assert "保守估算" in text and "可用输入预算 25856" in text
    assert str(engine.manager.request_size(engine.history, ())) in text
    assert "无需整理" in text
    assert "历史已整理" not in text and "CONTEXT_LIMIT" not in text


async def test_successful_manual_compact_reports_completed_and_preserves_current_goal(file_context, tmp_path, monkeypatch):
    # 修复契约：可整理时保留最后两轮，终端仅在实际成功后显示已整理。
    prompts = iter(["/compact", "/exit"])

    class Prompt:
        def __init__(self, **kwargs):
            pass

        async def prompt_async(self, text):
            return next(prompts)

    class Store:
        directory = tmp_path

        def __init__(self):
            self.events = []

        async def append(self, event):
            self.events.append(event)

    class Provider:
        def __init__(self):
            self.requests = []

        async def stream(self, request, cancel_event):
            self.requests.append(request)
            yield ModelFrame("text_delta", text="保留旧事实的摘要")
            yield ModelFrame("response_finished", status="completed")

    config = AgentConfig(model="test-model", state_dir=file_context.state_dir)
    provider, store = Provider(), Store()
    engine = AgentEngine(provider, [], store, file_context, config)
    goal = {"kind": "message", "role": "user", "content": "当前目标原文"}
    engine.history = [goal, *[{"kind": "response", "items": []} for _ in range(3)]]
    monkeypatch.setattr(terminal, "PromptSession", Prompt)
    monkeypatch.setattr(terminal, "patch_stdout", lambda **kwargs: nullcontext())
    output = StringIO()
    assert await terminal.interactive(engine, config, Console(file=output, width=160)) == 0
    assert "历史已整理" in output.getvalue() and "无需整理" not in output.getvalue()
    assert len(provider.requests) == 1
    assert engine.history[1] == goal
    assert [event.type for event in store.events] == ["context_compacted"]
