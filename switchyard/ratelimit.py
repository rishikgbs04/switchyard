"""Token-bucket rate limiter, one bucket per API key."""
from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class _Bucket:
    tokens: float
    updated: float


class RateLimiter:
    def __init__(self, clock=time.monotonic) -> None:
        self.clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str, per_minute: int) -> tuple[bool, float]:
        """Returns (allowed, seconds_until_next_token)."""
        now = self.clock()
        rate = per_minute / 60.0
        b = self._buckets.get(key)
        if b is None:
            b = self._buckets[key] = _Bucket(tokens=float(per_minute), updated=now)
        b.tokens = min(float(per_minute), b.tokens + (now - b.updated) * rate)
        b.updated = now
        if b.tokens >= 1.0:
            b.tokens -= 1.0
            return True, 0.0
        return False, (1.0 - b.tokens) / rate
