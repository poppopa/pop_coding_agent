"""工具只接触执行上下文，无法访问模型客户端的 Key。"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any
import asyncio
from pydantic import BaseModel, ConfigDict
from coding_agent.domain import ToolResult
from coding_agent.policy import Policy
from coding_agent.security import Redactor

class ToolInput(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)

@dataclass
class ToolContext:
    root: Path
    call_id: str
    snapshots: Any
    policy: Policy
    emit: Callable[[str, dict], Awaitable[None]]
    redactor: Redactor
    state_dir: Path
    filtered_env_names: tuple[str, ...] = ('ARK_API_KEY', 'OPENAI_API_KEY', 'ANTHROPIC_API_KEY')

class Tool:
    name: str
    description: str
    input_type: type[ToolInput]
    mutating = False

    @property
    def input_schema(self) -> dict:
        return self.input_type.model_json_schema()

    def schema(self) -> dict:
        return {'type': 'function', 'name': self.name, 'description': self.description, 'parameters': self.input_schema}

    async def execute(self, arguments: dict, context: ToolContext, cancel_event: asyncio.Event) -> ToolResult:
        raise NotImplementedError

class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self, mode: str) -> tuple[dict, ...]:
        return tuple(t.schema() for t in self.tools.values() if mode != 'analysis' or not t.mutating)

    def get(self, name: str, mode: str) -> Tool | None:
        tool = self.tools.get(name)
        return None if tool and mode == 'analysis' and tool.mutating else tool
