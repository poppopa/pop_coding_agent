"""技术方案§CLI与配置行为；仅测试键绑定与本地命令，不启动真实终端。"""
from contextlib import nullcontext
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from prompt_toolkit.keys import Keys
from rich.console import Console

from coding_agent.domain import Event
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
    monkeypatch.setattr(terminal, "patch_stdout", nullcontext)
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
    monkeypatch.setattr(terminal, "patch_stdout", nullcontext)
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
