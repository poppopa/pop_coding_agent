"""非敏感配置与 API Key 分离；CLI 只覆盖用户明确给出的字段。"""
from pathlib import Path
import os
import tomllib
from pydantic import BaseModel, ConfigDict, Field, model_validator

class AgentConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    provider: str = 'ark'
    api: str = 'responses'
    model: str = Field(min_length=1)
    base_url: str = 'https://ark.cn-beijing.volces.com/api/v3'
    mode: str = 'execute'
    max_turns: int = Field(default=30, ge=1, le=30)
    context_tokens: int = Field(default=32000, ge=8192)
    max_output_tokens: int = Field(default=4096, ge=256)
    read_timeout: float = Field(default=120, gt=0)
    connect_timeout: float = Field(default=10, gt=0)
    state_dir: Path = Field(default_factory=lambda: Path.home() / '.local/share/cagent')
    filtered_env_names: list[str] = Field(default_factory=lambda: ['ARK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY'])

    @model_validator(mode='after')
    def valid_protocol(self):
        required = ['ARK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY']
        self.filtered_env_names = list(dict.fromkeys(required + self.filtered_env_names))
        if self.provider != 'ark' or self.api != 'responses':
            raise ValueError('首版仅支持 ark/responses')
        if self.mode not in ('analysis', 'execute'):
            raise ValueError('mode 必须是 analysis 或 execute')
        if self.max_output_tokens + 2048 >= self.context_tokens:
            raise ValueError('上下文必须预留输出预算与协议余量')
        return self

def load_config(path: Path | None = None, overrides: dict | None = None, *, allow_unconfigured: bool = False) -> AgentConfig:
    path = path or Path.home() / '.config/cagent/config.toml'
    values = tomllib.loads(path.read_text()) if path.exists() else {}
    if os.environ.get('ARK_MODEL') and not values.get('model'):
        values['model'] = os.environ['ARK_MODEL']
    values.update({k: v for k, v in (overrides or {}).items() if v is not None})
    if allow_unconfigured and not values.get('model'):
        values['model'] = '__unconfigured__'
    return AgentConfig(**values)
