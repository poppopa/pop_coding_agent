"""验收依据：技术方案§Ark 请求、CLI 与配置、上下文与会话恢复。"""
from pathlib import Path

import pytest
from pydantic import ValidationError

from coding_agent.config import AgentConfig, load_config


def test_documented_defaults(tmp_path):
    config = AgentConfig(model="test-model", state_dir=tmp_path)
    assert (config.provider, config.api, config.mode) == ("ark", "responses", "execute")
    assert config.max_turns == 30
    assert config.context_tokens == 32_000
    assert config.max_output_tokens == 4_096
    assert config.connect_timeout == 10
    assert config.read_timeout == 120


@pytest.mark.parametrize("model", [pytest.param(None, id="missing"), pytest.param("", id="empty")])
def test_model_is_required(model, tmp_path):
    values = {"state_dir": tmp_path}
    if model is not None:
        values["model"] = model
    with pytest.raises(ValidationError):
        AgentConfig(**values)


@pytest.mark.parametrize("override", [
    pytest.param({"provider": "other"}, id="provider"),
    pytest.param({"api": "chat"}, id="api"),
    pytest.param({"mode": "unsafe"}, id="mode"),
    pytest.param({"connect_timeout": 0}, id="connect-timeout"),
    pytest.param({"read_timeout": 0}, id="read-timeout"),
    pytest.param({"max_turns": 0}, id="zero-turns"),
    pytest.param({"max_turns": 31}, id="over-turn-limit"),
    pytest.param({"context_tokens": 8192, "max_output_tokens": 7000}, id="no-protocol-margin"),
])
def test_invalid_configuration_is_rejected(override, tmp_path):
    with pytest.raises(ValidationError):
        AgentConfig(model="test-model", state_dir=tmp_path, **override)


def test_explicit_options_override_user_configuration(tmp_path, monkeypatch):
    monkeypatch.delenv("ARK_MODEL", raising=False)
    config_file = tmp_path / "config.toml"
    config_file.write_text('model = "test-file-model"\nmode = "analysis"\n', encoding="utf-8")
    config = load_config(config_file, {"model": "test-cli-model", "mode": None})
    assert config.model == "test-cli-model"
    assert config.mode == "analysis"
    assert config_file.read_text(encoding="utf-8") == 'model = "test-file-model"\nmode = "analysis"\n'


def test_non_sensitive_config_has_no_credential_field(tmp_path):
    config = AgentConfig(model="test-model", state_dir=tmp_path)
    assert not {"api_key", "token", "password", "secret"} & config.model_dump().keys()
