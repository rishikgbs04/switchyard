"""Anthropic Messages API upstream, translated to/from the OpenAI shape."""
from __future__ import annotations

import os

import httpx

from switchyard.config import ModelConfig
from switchyard.providers.base import Provider, ProviderError
from switchyard.schemas import Message, ProviderResult


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=60)

    async def complete(
        self,
        model: ModelConfig,
        messages: list[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> ProviderResult:
        system = "\n".join(m.content for m in messages if m.role == "system")
        convo = [{"role": m.role, "content": m.content} for m in messages if m.role != "system"]
        payload = {
            "model": model.upstream_name or model.name,
            "messages": convo,
            "max_tokens": max_tokens or 1024,
            "temperature": temperature,
        }
        if system:
            payload["system"] = system
        try:
            r = await self.client.post(
                "https://api.anthropic.com/v1/messages",
                json=payload,
                headers={
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                },
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"anthropic transport error: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"anthropic {r.status_code}: {r.text[:200]}")
        data = r.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage", {})
        return ProviderResult(
            text=text,
            prompt_tokens=usage.get("input_tokens", 0),
            completion_tokens=usage.get("output_tokens", 0),
            model=model.name,
        )
