"""Deterministic mock provider so the gateway runs, tests and benchmarks offline.

Behaviour is controlled per-model with simple knobs so tests can simulate
outages and slow upstreams.
"""
from __future__ import annotations

import asyncio
import hashlib

from switchyard.config import ModelConfig
from switchyard.providers.base import Provider, ProviderError, estimate_tokens
from switchyard.schemas import Message, ProviderResult

# Simulated latency by tier, in seconds (bigger models are slower).
TIER_LATENCY = {1: 0.02, 2: 0.05, 3: 0.10}


class MockProvider(Provider):
    name = "mock"

    def __init__(self) -> None:
        self.down: set[str] = set()  # model names that should fail
        self.calls: dict[str, int] = {}

    async def complete(
        self,
        model: ModelConfig,
        messages: list[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> ProviderResult:
        self.calls[model.name] = self.calls.get(model.name, 0) + 1
        if model.name in self.down:
            raise ProviderError(f"{model.name} is unavailable (simulated outage)")
        await asyncio.sleep(TIER_LATENCY.get(model.tier, 0.05))
        prompt = "\n".join(m.content for m in messages)
        digest = hashlib.sha1(prompt.encode()).hexdigest()[:8]
        text = f"[{model.name}] answer #{digest} to: {messages[-1].content[:80]}"
        return ProviderResult(
            text=text,
            prompt_tokens=estimate_tokens(prompt),
            completion_tokens=estimate_tokens(text),
            model=model.name,
        )
