"""Core request pipeline: budget -> cache -> route -> call with fallback -> log."""
from __future__ import annotations

import time
from dataclasses import dataclass

from switchyard.breaker import CircuitBreaker
from switchyard.cache import SemanticCache
from switchyard.config import ModelConfig, Settings
from switchyard.metrics import Metrics
from switchyard.providers import Provider, ProviderError
from switchyard.router import plan_route
from switchyard.schemas import ChatRequest, ProviderResult
from switchyard.store import Store


class AllProvidersFailed(Exception):
    pass


class BudgetExceeded(Exception):
    pass


def cost_of(m: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * m.input_cost_per_mtok + completion_tokens * m.output_cost_per_mtok) / 1e6


@dataclass
class GatewayResult:
    result: ProviderResult
    cache: str
    attempts: int
    difficulty: float | None
    cost_usd: float
    saved_usd: float


class Gateway:
    def __init__(self, settings: Settings, providers: dict[str, Provider], cache: SemanticCache,
                 breaker: CircuitBreaker, store: Store, metrics: Metrics) -> None:
        self.s = settings
        self.providers = providers
        self.cache = cache
        self.breaker = breaker
        self.store = store
        self.metrics = metrics
        self.baseline = max(settings.models, key=lambda m: (m.tier, m.output_cost_per_mtok))

    async def handle(self, req: ChatRequest, api_key: str, budget: float | None) -> GatewayResult:
        t0 = time.perf_counter()
        if budget is not None and self.store.month_spend(api_key) >= budget:
            raise BudgetExceeded(f"monthly budget of ${budget:.2f} reached")

        plan = plan_route(self.s, req.model, req.messages)
        difficulty = plan.difficulty.score if plan.difficulty else None
        use_cache = self.s.cache.enabled and req.cache and SemanticCache.cacheable(req.temperature)

        if use_cache and (hit := self.cache.get(req.model, req.messages)):
            r = hit.result
            saved = cost_of(self.baseline, r.prompt_tokens, r.completion_tokens)
            return self._finish(t0, api_key, req.model, r, hit.kind, 0, difficulty, 0.0, saved)

        attempts = 0
        errors: list[str] = []
        for model in plan.candidates:
            if not self.breaker.allow(model.name):
                continue
            provider = self.providers.get(model.provider)
            if provider is None:
                continue
            attempts += 1
            try:
                r = await provider.complete(model, req.messages, req.temperature, req.max_tokens)
            except ProviderError as e:
                self.breaker.record_failure(model.name)
                self.metrics.fallback(model.name)
                errors.append(f"{model.name}: {e}")
                continue
            self.breaker.record_success(model.name)
            if use_cache:
                self.cache.put(req.model, req.messages, r)
            cost = cost_of(model, r.prompt_tokens, r.completion_tokens)
            base = cost_of(self.baseline, r.prompt_tokens, r.completion_tokens)
            return self._finish(t0, api_key, req.model, r, "miss", attempts, difficulty,
                                cost, max(base - cost, 0.0))

        latency = (time.perf_counter() - t0) * 1000
        self.store.log(api_key=api_key, policy=req.model, model=None, difficulty=difficulty,
                       cache="miss", attempts=attempts, latency_ms=latency, status="error")
        self.metrics.observe("none", "miss", "error", latency, 0.0)
        raise AllProvidersFailed("; ".join(errors) or "no healthy model available")

    def _finish(self, t0, api_key, policy, r, cache, attempts, difficulty, cost, saved):
        latency = (time.perf_counter() - t0) * 1000
        self.store.log(api_key=api_key, policy=policy, model=r.model, difficulty=difficulty,
                       cache=cache, attempts=attempts, prompt_tokens=r.prompt_tokens,
                       completion_tokens=r.completion_tokens, cost_usd=cost, saved_usd=saved,
                       latency_ms=latency, status="ok")
        self.metrics.observe(r.model, cache, "ok", latency, cost)
        return GatewayResult(r, cache, attempts, difficulty, cost, saved)
