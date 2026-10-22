"""凭证过滤是日志边界的共同职责，不依赖模型遵守提示词。"""
from __future__ import annotations
import re
from typing import Any

class Redactor:
    def __init__(self, secrets: list[str] | tuple[str, ...] = ()):
        self.secrets = tuple(sorted((s for s in secrets if s), key=len, reverse=True))

    def text(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, '<REDACTED>')
        text = re.sub(r'(?i)(Bearer\s+)\S+', r'\1<REDACTED>', text)
        text = re.sub(r'(?i)((?:api[_-]?key|access[_-]?token|password|secret)\s*[=:]\s*)[^\s,;]+', r'\1<REDACTED>', text)
        return text

    def value(self, value: Any) -> Any:
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, list):
            return [self.value(v) for v in value]
        if isinstance(value, tuple):
            return tuple(self.value(v) for v in value)
        if isinstance(value, dict):
            return {k: '<REDACTED>' if re.fullmatch(r'(?i)(api_key|access_token|password|secret|authorization)', str(k)) else self.value(v) for k, v in value.items()}
        return value
