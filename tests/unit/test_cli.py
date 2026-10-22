"""技术方案§CLI与配置行为：临时配置、注入模型、JSONL与退出码、离线doctor。"""
import json
import sys

import pytest

from coding_agent.cli import run_cli
from coding_agent.domain import ModelFrame, ToolCall
from coding_agent.tools.command import RunCommandTool


class FakeProvider:
    def __init__(self, frames=None):
        self.frames = frames or [ModelFrame("text_delta", text="测试完成"), ModelFrame("response_finished", status="completed")]
        self.requests = []
        self.closed = 0

    async def stream(self, request, cancel_event):
        self.requests.append(request)
        for frame in self.frames:
            yield frame

    async def close(self):
        self.closed += 1


@pytest.fixture
def cli_args(tmp_path, monkeypatch):
    for name in ("ARK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(name, "<your-test-key>")
    monkeypatch.delenv("ARK_MODEL", raising=False)
    config = tmp_path / "config.toml"
    config.write_text('model = "test-model"\n', encoding="utf-8")
    return ["--config", str(config), "--state-dir", str(tmp_path / "state"), "--root", str(tmp_path)]


def json_events(capsys):
    output = capsys.readouterr()
    assert "\x1b" not in output.out
    rows = [json.loads(line) for line in output.out.splitlines()]
    assert rows
    return rows, output.err


def test_exec_json_lines_are_events_and_provider_is_closed(cli_args, capsys):
    provider = FakeProvider()
    assert run_cli([*cli_args, "exec", "测试任务", "--json"], {"provider": provider}) == 0
    rows, err = json_events(capsys)
    assert err == ""
    assert rows[-1]["type"] == "turn_finished"
    assert rows[-1]["payload"]["status"] == "completed"
    assert all(set(row) == {"schema_version", "session_id", "turn_id", "seq", "type", "payload"} for row in rows)
    assert provider.closed == 1
    assert len(provider.requests) == 1


def test_unapproved_exec_returns_exit_three_and_zero_side_effect(cli_args, tmp_path, capsys):
    argv = [sys.executable, "-c", "from pathlib import Path; Path('marker').touch()"]
    call = ToolCall("r1", "c1", "run_command", json.dumps({"argv": argv}))
    provider = FakeProvider([ModelFrame("response_finished", status="completed", calls=(call,))])
    assert run_cli([*cli_args, "exec", "测试命令", "--json"], {"provider": provider, "tools": [RunCommandTool()]}) == 3
    rows, _ = json_events(capsys)
    assert rows[-1]["payload"]["error"]["code"] == "APPROVAL_REQUIRED"
    assert not (tmp_path / "marker").exists()
    assert not any(row["type"] == "tool_started" for row in rows)


def test_analysis_schemas_have_only_four_read_tools(cli_args, capsys):
    provider = FakeProvider()
    assert run_cli([*cli_args, "--mode", "analysis", "exec", "只读分析", "--json"], {"provider": provider}) == 0
    names = {item["name"] for item in provider.requests[0].tool_schemas}
    assert names == {"read_file", "find_files", "search_text", "show_diff"}
    json_events(capsys)


def test_doctor_is_offline_and_shows_only_key_presence(cli_args, capsys):
    provider = FakeProvider()
    assert run_cli([*cli_args, "doctor"], {"provider": provider}) == 0
    output = capsys.readouterr()
    result = json.loads(output.out)
    assert result["key_present"] is True
    assert result["model_configured"] is True
    assert "<your-test-key>" not in output.out + output.err
    assert provider.requests == []


def test_doctor_without_key_reports_absence_offline(cli_args, capsys, monkeypatch):
    monkeypatch.delenv("ARK_API_KEY", raising=False)
    provider = FakeProvider()
    assert run_cli([*cli_args, "doctor"], {"provider": provider}) == 2
    assert json.loads(capsys.readouterr().out)["key_present"] is False
    assert provider.requests == []


def test_fake_provider_does_not_require_key_or_model(cli_args, tmp_path, capsys, monkeypatch):
    for name in ("ARK_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    provider = FakeProvider()
    assert run_cli([*cli_args, "exec", "离线测试", "--json"], {"provider": provider}) == 0
    json_events(capsys)
    assert len(provider.requests) == 1


def test_failed_response_returns_exit_one(cli_args, capsys):
    provider = FakeProvider([ModelFrame("response_finished", status="incomplete")])
    assert run_cli([*cli_args, "exec", "测试失败响应", "--json"], {"provider": provider}) == 1
    rows, _ = json_events(capsys)
    assert rows[-1]["payload"]["status"] == "failed"


def test_missing_model_is_startup_error(cli_args, tmp_path, capsys):
    (tmp_path / "config.toml").write_text("", encoding="utf-8")
    assert run_cli([*cli_args, "exec", "测试启动", "--json"]) == 2
    assert capsys.readouterr().err
