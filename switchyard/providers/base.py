"""Provider adapter interface."""
from __future__ import annotations

from abc import ABC, abstractmethod

from switchyard.config import ModelConfig
from switchyard.schemas import Message, ProviderResult


class ProviderError(Exception):
    """Raised when an upstream provider fails; triggers fallback."""


class Provider(ABC):
    name: str

    @abstractmethod
    async def complete(
        self,
        model: ModelConfig,
        messages: list[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> ProviderResult: ...


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) for providers that don't report usage."""
    return max(1, len(text) // 4)
