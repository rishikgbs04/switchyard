"""OpenAI-compatible upstream (works for OpenAI and any server speaking its API,
e.g. vLLM, Ollama, Together, Groq) via plain HTTP."""
from __future__ import annotations

import os

import httpx

from switchyard.config import ModelConfig
from switchyard.providers.base import Provider, ProviderError
from switchyard.schemas import Message, ProviderResult


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self, base_url: str | None = None, api_key: str | None = None) -> None:
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.client = httpx.AsyncClient(timeout=60)

    async def complete(
        self,
        model: ModelConfig,
        messages: list[Message],
        temperature: float,
        max_tokens: int | None,
    ) -> ProviderResult:
        payload = {
            "model": model.upstream_name or model.name,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        try:
            r = await self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
        except httpx.HTTPError as e:
            raise ProviderError(f"openai transport error: {e}") from e
        if r.status_code >= 400:
            raise ProviderError(f"openai {r.status_code}: {r.text[:200]}")
        data = r.json()
        usage = data.get("usage", {})
        return ProviderResult(
            text=data["choices"][0]["message"]["content"] or "",
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            model=model.name,
        )
