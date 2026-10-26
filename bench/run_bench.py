"""Benchmark Switchyard on a synthetic, labelled workload (offline, mock models).

Measures, against an "always send to the top model" baseline:
  - cost reduction from routing + caching
  - cache hit rate (exact vs semantic)
  - router accuracy vs hand-labelled target tiers
  - availability under a simulated outage of one model

Usage:  python -m bench.run_bench [--n 2000] [--seed 7]
The numbers are only as realistic as the workload; re-run against a sample
of real traffic before quoting them anywhere.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import statistics
import tempfile
import time
from pathlib import Path

from switchyard.breaker import CircuitBreaker
from switchyard.cache import SemanticCache
from switchyard.config import load_settings
from switchyard.embeddings import HashEmbedder
from switchyard.gateway import Gateway, cost_of
from switchyard.metrics import Metrics
from switchyard.providers import MockProvider
from switchyard.router import plan_route
from switchyard.schemas import ChatRequest, Message
from switchyard.store import Store

# Templated intents: (target_tier, [paraphrase templates], slot values).
# Slots create a long tail of distinct questions; paraphrases of the same
# question exercise the semantic cache; Zipf popularity creates repeats.
COUNTRIES = ["France", "Japan", "Brazil", "Kenya", "Canada", "India", "Peru", "Norway", "Egypt",
             "Chile", "Vietnam", "Ghana", "Poland", "Spain", "Nepal", "Mexico", "Italy", "Iran"]
WORDS = ["happy", "fast", "big", "smart", "angry", "calm", "brave", "tired", "rich", "quiet"]
NUMS = [str(n) for n in range(1, 60)]
LANGS = ["Python", "JavaScript", "Go", "Rust", "Java"]
TOPICS = ["REST vs GraphQL", "SQL vs NoSQL", "threads vs async", "monolith vs microservices",
          "Kafka vs RabbitMQ", "Redis vs Memcached", "TCP vs UDP", "B-trees vs LSM trees"]
SYSTEMS = ["a URL shortener", "a rate limiter", "a chat service", "a news feed", "a job scheduler",
           "a payment ledger", "a search autocomplete", "a ride matching service"]

INTENTS = [
    (1, ["What is the capital of {x}?", "whats the capital of {x}", "capital of {x}?"], COUNTRIES),
    (1, ["How do I reverse a list in {x}?", "how to reverse a {x} list", "reverse a list in {x}"], LANGS),
    (1, ["Convert {x} miles to kilometers", "{x} miles in km", "how many kilometers is {x} miles"], NUMS),
    (1, ["Give me a synonym for {x}", "synonym of {x}", "another word for {x}"], WORDS),
    (1, ["What is the population of {x}?", "population of {x}", "how many people live in {x}"], COUNTRIES),
    (2, ["Compare {x} and explain why you would pick one for a startup",
         "Explain why a startup would choose between {x}, compare them"], TOPICS),
    (2, ["Write a SQL query: SELECT the top {x} customers by revenue from orders joined with customers",
         "SELECT top {x} customers by total revenue, joining orders and customers tables"], NUMS),
    (2, ["Refactor this {x} function to be more readable:\n```def f(a):\n  return [x*2 for x in a if x>0]```"], LANGS),
    (3, ["Design {x}, analyze the trade-offs of the storage options, derive the capacity estimates "
         "step by step, and optimize the hot path:\n```class Service:\n  pass```"], SYSTEMS),
    (3, ["Prove step by step that the algorithm in {x} runs in O(n log n), analyze the trade-offs "
         "versus quicksort, and debug:\n```def merge(a,b):\n  return sorted(a+b)```"], LANGS),
]


CONTEXT_VOCAB = ["billing", "onboarding", "dashboard", "migration", "invoice", "analytics", "checkout", "inventory", "warehouse", "shipping", "reporting", "compliance", "payroll", "scheduling", "marketing", "support", "telemetry", "pricing", "loyalty", "fraud", "recruiting", "logistics", "forecasting", "auditing"]


def build_workload(n: int, rng: random.Random, unique_frac: float = 0.5):
    """Zipf popularity over (intent, slot) pairs, like real chat traffic, with
    `unique_frac` of requests wrapped in a random one-off project context so
    they are textually unique. Each request carries a ground-truth key
    (intent, slot): a cache hit is correct only if it returns an answer
    generated for the same key."""
    items = [(tier, phr, x, (i, x)) for i, (tier, phr, slots) in enumerate(INTENTS) for x in slots]
    rng.shuffle(items)
    weights = [1 / (i + 1) ** 0.6 for i in range(len(items))]
    out = []
    for _ in range(n):
        tier, phr, x, key = rng.choices(items, weights=weights)[0]
        text = rng.choice(phr).format(x=x)
        if rng.random() < unique_frac:
            ctx = " ".join(rng.sample(CONTEXT_VOCAB, 4))
            text = f"For our {ctx} project (ticket {rng.randint(1000, 99999)}): {text}"
        out.append((tier, text, key))
    return out


async def run(n: int, seed: int, outage: str | None, use_cache: bool = True) -> dict:
    rng = random.Random(seed)
    settings = load_settings(str(Path(__file__).resolve().parent.parent / "config.example.yaml"))
    tmp = tempfile.mkdtemp()
    mock = MockProvider()
    if outage:
        mock.down.add(outage)
    gw = Gateway(settings, {"mock": mock},
                 SemanticCache(HashEmbedder(), settings.cache.similarity_threshold,
                               semantic_max_words=settings.cache.semantic_max_words),
                 CircuitBreaker(3, 30), Store(f"{tmp}/bench.db"), Metrics())
    baseline_model = gw.baseline

    work = build_workload(n, rng)
    cost = base_cost = 0.0
    cache = {"miss": 0, "exact": 0, "semantic": 0}
    correct = errors = wrong_hits = 0
    answered_for: dict[str, tuple] = {}  # response text -> ground-truth key
    lat = []
    for tier, text, key in work:
        msgs = [Message(role="user", content=text)]
        plan = plan_route(settings, "auto", msgs)
        correct += plan.difficulty.tier == tier
        t0 = time.perf_counter()
        try:
            out = await gw.handle(ChatRequest(model="auto", messages=msgs, cache=use_cache), "bench", None)
        except Exception:
            errors += 1
            continue
        lat.append((time.perf_counter() - t0) * 1000)
        cache[out.cache] += 1
        if out.cache == "miss":
            answered_for[out.result.text] = key
        elif answered_for.get(out.result.text) != key:
            wrong_hits += 1
        cost += out.cost_usd
        r = out.result
        base_cost += cost_of(baseline_model, r.prompt_tokens, r.completion_tokens)

    lat.sort()
    return {
        "requests": n,
        "baseline_cost": base_cost,
        "switchyard_cost": cost,
        "cost_reduction_pct": 100 * (1 - cost / base_cost) if base_cost else 0,
        "cache_hit_pct": 100 * (cache["exact"] + cache["semantic"]) / n,
        "semantic_hit_pct": 100 * cache["semantic"] / n,
        "router_accuracy_pct": 100 * correct / n,
        "wrong_hit_pct": 100 * wrong_hits / max(1, cache["exact"] + cache["semantic"]),
        "success_pct": 100 * (n - errors) / n,
        "p50_ms": statistics.median(lat),
        "p95_ms": lat[int(0.95 * len(lat))],
        "upstream_calls": sum(mock.calls.values()),
    }


def fmt(name: str, r: dict) -> str:
    return (f"| {name} | {r['requests']} | {r['cost_reduction_pct']:.1f}% | {r['cache_hit_pct']:.1f}% "
            f"({r['semantic_hit_pct']:.1f}% semantic) | {r['wrong_hit_pct']:.2f}% | {r['router_accuracy_pct']:.1f}% | "
            f"{r['success_pct']:.1f}% | {r['p50_ms']:.2f} / {r['p95_ms']:.1f} | {r['upstream_calls']} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    routing = asyncio.run(run(a.n, a.seed, None, use_cache=False))
    normal = asyncio.run(run(a.n, a.seed, None))
    outage = asyncio.run(run(a.n, a.seed, "mock-small"))
    table = "\n".join([
        "| Scenario | Requests | Cost reduction vs top model | Cache hit rate | Wrong cache hits | Router accuracy | Success | p50 / p95 ms | Upstream calls |",
        "|---|---|---|---|---|---|---|---|---|",
        fmt("Routing only (cache off)", routing),
        fmt("Routing + cache", normal),
        fmt("Routing + cache, small model down", outage),
    ])
    print(table)
    Path(__file__).with_name("results.md").write_text(
        f"# Benchmark results\n\nSynthetic labelled workload, mock models, seed {a.seed}. "
        f"Generated by `python -m bench.run_bench`.\n\n{table}\n")


if __name__ == "__main__":
    main()
