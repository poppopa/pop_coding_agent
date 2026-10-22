from typing import Protocol
from collections.abc import AsyncIterator
import asyncio
from coding_agent.domain import ModelRequest, ModelFrame

class ModelProvider(Protocol):
    def stream(self, request: ModelRequest, cancel_event: asyncio.Event) -> AsyncIterator[ModelFrame]: ...
