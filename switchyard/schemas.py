"""OpenAI-compatible request/response schemas (the subset Switchyard supports)."""
from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ChatRequest(BaseModel):
    model: str = Field(
        default="auto",
        description="A concrete model name, or a routing policy: 'auto', 'cheapest', 'best'.",
    )
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int | None = None
    # Switchyard extensions (ignored by plain OpenAI clients)
    cache: bool = True


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int = 0
    message: Message
    finish_reason: str = "stop"


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:24]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage


class ProviderResult(BaseModel):
    """What a provider adapter returns before the gateway wraps it."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
