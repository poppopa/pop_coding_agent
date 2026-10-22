"""内核、工具和界面共享的数据契约；供应商协议被封装在适配层。"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Literal

@dataclass(frozen=True)
class ToolCall:
    request_id: str
    call_id: str
    name: str
    arguments_json: str

@dataclass(frozen=True)
class ToolError:
    code: str
    message: str
    retryable: bool = False

@dataclass(frozen=True)
class ToolResult:
    call_id: str
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: ToolError | None = None
    truncated: bool = False

    @classmethod
    def fail(cls, call_id: str, code: str, message: str, retryable: bool = False):
        return cls(call_id, False, error=ToolError(code, message, retryable))

@dataclass(frozen=True)
class Event:
    schema_version: int
    session_id: str
    turn_id: str
    seq: int
    type: str
    payload: dict[str, Any]

@dataclass(frozen=True)
class ModelFrame:
    kind: Literal['text_delta', 'usage', 'response_finished']
    text: str = ''
    status: str = ''
    calls: tuple[ToolCall, ...] = ()
    provider_output: tuple[dict[str, Any], ...] = ()
    usage: dict[str, int] = field(default_factory=dict)

@dataclass(frozen=True)
class ModelRequest:
    instructions: str
    history: tuple[dict[str, Any], ...]
    tool_schemas: tuple[dict[str, Any], ...]
    max_output_tokens: int = 4096
    retry_on_connection_error: bool = True
