"""Two-level response cache.

L1: exact match on the normalised request (O(1) dict lookup).
L2: semantic match. The last user message is embedded and compared by cosine
    similarity against cached entries that share the same "context key"
    (system prompt + earlier turns + routing policy), so a cached answer is
    never reused across different conversations or instructions.

Guard: embeddings are poor at distinguishing numbers ("top 5" vs "top 50"
scores ~0.92), so a semantic hit additionally requires the set of numbers in
both prompts to match exactly.

Semantic matching is also limited to short, code-free prompts. Long prompts
that share a template but differ in one key term ("design a URL shortener"
vs "design a rate limiter") embed too closely to be told apart safely; those
only get exact-match caching.

Only low-temperature requests are cached: at high temperature the caller is
asking for variety, so replaying an old answer would be wrong.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from switchyard.schemas import Message, ProviderResult

MAX_CACHEABLE_TEMPERATURE = 0.2
_NUMBER = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> frozenset[str]:
    return frozenset(_NUMBER.findall(text))


def semantic_eligible(text: str, max_words: int) -> bool:
    return "```" not in text and len(text.split()) <= max_words


@dataclass
class CacheHit:
    result: ProviderResult
    kind: str  # "exact" | "semantic"
    similarity: float


@dataclass
class _Entry:
    context_key: str
    vector: np.ndarray
    numbers: frozenset[str]
    result: ProviderResult
    created: float


def _context_key(policy: str, messages: list[Message]) -> str:
    ctx = [m.model_dump() for m in messages[:-1]]
    return hashlib.sha256(json.dumps([policy, ctx], sort_keys=True).encode()).hexdigest()


def _exact_key(policy: str, messages: list[Message]) -> str:
    norm = [{"r": m.role, "c": " ".join(m.content.lower().split())} for m in messages]
    return hashlib.sha256(json.dumps([policy, norm]).encode()).hexdigest()


class SemanticCache:
    def __init__(self, embedder, threshold: float = 0.85, ttl: float = 3600,
                 max_entries: int = 10_000, semantic_max_words: int = 40,
                 clock=time.monotonic) -> None:
        self.embedder = embedder
        self.semantic_max_words = semantic_max_words
        self.threshold = threshold
        self.ttl = ttl
        self.max_entries = max_entries
        self.clock = clock
        self._exact: OrderedDict[str, _Entry] = OrderedDict()
        self._by_ctx: dict[str, list[_Entry]] = {}

    @staticmethod
    def cacheable(temperature: float) -> bool:
        return temperature <= MAX_CACHEABLE_TEMPERATURE

    def _alive(self, e: _Entry) -> bool:
        return self.clock() - e.created < self.ttl

    def get(self, policy: str, messages: list[Message]) -> CacheHit | None:
        ek = _exact_key(policy, messages)
        e = self._exact.get(ek)
        if e and self._alive(e):
            self._exact.move_to_end(ek)
            return CacheHit(e.result, "exact", 1.0)

        if not semantic_eligible(messages[-1].content, self.semantic_max_words):
            return None
        nums = _numbers(messages[-1].content)
        bucket = [x for x in self._by_ctx.get(_context_key(policy, messages), [])
                  if self._alive(x) and x.numbers == nums]
        if not bucket:
            return None
        q = self.embedder.embed(messages[-1].content)
        mat = np.stack([x.vector for x in bucket])
        sims = mat @ q
        i = int(np.argmax(sims))
        if sims[i] >= self.threshold:
            return CacheHit(bucket[i].result, "semantic", float(sims[i]))
        return None

    def put(self, policy: str, messages: list[Message], result: ProviderResult) -> None:
        entry = _Entry(
            context_key=_context_key(policy, messages),
            vector=self.embedder.embed(messages[-1].content),
            numbers=_numbers(messages[-1].content),
            result=result,
            created=self.clock(),
        )
        self._exact[_exact_key(policy, messages)] = entry
        self._by_ctx.setdefault(entry.context_key, []).append(entry)
        while len(self._exact) > self.max_entries:
            _, old = self._exact.popitem(last=False)
            lst = self._by_ctx.get(old.context_key, [])
            if old in lst:
                lst.remove(old)

    def __len__(self) -> int:
        return len(self._exact)
