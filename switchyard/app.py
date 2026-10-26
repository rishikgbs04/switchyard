"""FastAPI app: OpenAI-compatible endpoint plus ops endpoints."""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from switchyard import __version__
from switchyard.breaker import CircuitBreaker
from switchyard.cache import SemanticCache
from switchyard.config import Settings, load_settings
from switchyard.embeddings import build_embedder
from switchyard.gateway import AllProvidersFailed, BudgetExceeded, Gateway
from switchyard.metrics import Metrics
from switchyard.providers import Provider, build_providers
from switchyard.ratelimit import RateLimiter
from switchyard.router import POLICIES, score_difficulty
from switchyard.schemas import ChatRequest, ChatResponse, Choice, Message, Usage
from switchyard.store import Store

# Declared as a security scheme (not a plain header) so the /docs page shows an
# Authorize button; Swagger UI never sends a header parameter named Authorization.
bearer = HTTPBearer(auto_error=False)
BearerCreds = Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)]


def create_app(settings: Settings | None = None,
               providers: dict[str, Provider] | None = None) -> FastAPI:
    settings = settings or load_settings()
    providers = providers or build_providers()
    metrics = Metrics()
    breaker = CircuitBreaker(settings.breaker_failure_threshold, settings.breaker_cooldown_seconds)
    cache = SemanticCache(build_embedder(), settings.cache.similarity_threshold,
                          settings.cache.ttl_seconds, settings.cache.max_entries,
                          settings.cache.semantic_max_words)
    store = Store(settings.db_path)
    gw = Gateway(settings, providers, cache, breaker, store, metrics)
    limiter = RateLimiter()
    keys = {k.key: k for k in settings.api_keys}

    app = FastAPI(title="Switchyard", version=__version__,
                  description="Self-hosted, OpenAI-compatible LLM gateway.")
    app.state.gateway = gw

    def authenticate(authorization: str | None):
        token = (authorization or "").removeprefix("Bearer ").strip()
        if token in keys:
            return token, keys[token]
        if not settings.require_auth:
            return token or "anonymous", None
        raise HTTPException(401, "invalid or missing API key")

    @app.post("/v1/chat/completions")
    async def chat(req: ChatRequest,
                   creds: BearerCreds):
        key, kcfg = authenticate(creds.credentials if creds else None)
        if kcfg:
            ok, wait = limiter.allow(key, kcfg.requests_per_minute)
            if not ok:
                raise HTTPException(429, "rate limit exceeded",
                                    headers={"Retry-After": f"{wait:.1f}"})
        if req.model not in POLICIES and settings.model(req.model) is None:
            raise HTTPException(404, f"unknown model '{req.model}'")
        try:
            out = await gw.handle(req, key, kcfg.monthly_budget_usd if kcfg else None)
        except BudgetExceeded as e:
            raise HTTPException(402, str(e)) from e
        except AllProvidersFailed as e:
            raise HTTPException(502, f"all upstreams failed: {e}") from e

        r = out.result
        body = ChatResponse(
            model=r.model,
            choices=[Choice(message=Message(role="assistant", content=r.text))],
            usage=Usage(prompt_tokens=r.prompt_tokens, completion_tokens=r.completion_tokens,
                        total_tokens=r.prompt_tokens + r.completion_tokens),
        )
        headers = {
            "x-switchyard-model": r.model,
            "x-switchyard-cache": out.cache,
            "x-switchyard-attempts": str(out.attempts),
            "x-switchyard-cost-usd": f"{out.cost_usd:.8f}",
        }
        if out.difficulty is not None:
            headers["x-switchyard-difficulty"] = str(out.difficulty)
        return JSONResponse(body.model_dump(), headers=headers)

    @app.post("/v1/route/explain")
    async def explain(req: ChatRequest):
        """Dry run: show how a request would be routed, without calling a model."""
        from switchyard.router import plan_route

        plan = plan_route(settings, req.model, req.messages)
        d = plan.difficulty or score_difficulty(req.messages)
        return {"policy": plan.policy, "difficulty": d.score, "target_tier": d.tier,
                "features": d.features, "candidates": [m.name for m in plan.candidates]}

    @app.get("/v1/models")
    async def models():
        data = [{"id": m.name, "object": "model", "owned_by": m.provider, "tier": m.tier}
                for m in settings.models]
        data += [{"id": p, "object": "model", "owned_by": "switchyard-router"} for p in sorted(POLICIES)]
        return {"object": "list", "data": data}

    @app.get("/v1/stats")
    async def stats():
        return {**store.stats(), "cache_entries": len(cache), "breakers": breaker.status()}

    @app.get("/metrics", response_class=PlainTextResponse)
    async def prom():
        return metrics.render()

    @app.get("/healthz")
    async def health():
        return {"status": "ok", "version": __version__}

    return app