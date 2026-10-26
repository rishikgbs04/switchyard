"""Per-model circuit breaker: stop sending traffic to a failing upstream,
then let a single probe through after a cooldown."""
from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class _State:
    failures: int = 0
    opened_at: float | None = None


@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    cooldown_seconds: float = 30.0
    clock: callable = time.monotonic
    _states: dict[str, _State] = field(default_factory=dict)

    def _s(self, key: str) -> _State:
        return self._states.setdefault(key, _State())

    def allow(self, key: str) -> bool:
        s = self._s(key)
        # Closed, or open long enough that a half-open probe is allowed.
        return s.opened_at is None or self.clock() - s.opened_at >= self.cooldown_seconds

    def record_success(self, key: str) -> None:
        s = self._s(key)
        s.failures = 0
        s.opened_at = None

    def record_failure(self, key: str) -> None:
        s = self._s(key)
        s.failures += 1
        if s.failures >= self.failure_threshold:
            s.opened_at = self.clock()

    def status(self) -> dict[str, str]:
        return {k: ("open" if not self.allow(k) else "closed") for k in self._states}
