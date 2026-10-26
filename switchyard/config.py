"""Gateway configuration, loaded from YAML."""
from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    name: str
    provider: str  # "mock" | "openai" | "anthropic"
    tier: int = Field(ge=1, le=3, description="1 = small/cheap, 3 = frontier")
    input_cost_per_mtok: float  # USD per 1M input tokens
    output_cost_per_mtok: float  # USD per 1M output tokens
    upstream_name: str | None = None  # name the provider expects, if different


class CacheConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.85
    semantic_max_words: int = 40
    ttl_seconds: int = 3600
    max_entries: int = 10_000


class KeyConfig(BaseModel):
    key: str
    requests_per_minute: int = 60
    monthly_budget_usd: float = 10.0


class Settings(BaseModel):
    models: list[ModelConfig]
    cache: CacheConfig = CacheConfig()
    api_keys: list[KeyConfig] = []
    require_auth: bool = True
    db_path: str = "switchyard.db"
    breaker_failure_threshold: int = 3
    breaker_cooldown_seconds: float = 30.0

    def model(self, name: str) -> ModelConfig | None:
        return next((m for m in self.models if m.name == name), None)


def load_settings(path: str | None = None) -> Settings:
    path = path or os.environ.get("SWITCHYARD_CONFIG", "config.yaml")
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / "config.example.yaml"
    raw = yaml.safe_load(p.read_text())
    return Settings(**raw)
