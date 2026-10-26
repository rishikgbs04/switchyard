"""Minimal Prometheus exposition (no client library needed)."""
from __future__ import annotations

import threading
from collections import defaultdict

LATENCY_BUCKETS_MS = (25, 50, 100, 250, 500, 1000, 2500, 5000, 10000)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests = defaultdict(int)       # (model, cache, status) -> n
        self.fallbacks = defaultdict(int)      # model -> n
        self.cost = defaultdict(float)         # model -> usd
        self.hist = defaultdict(lambda: [0] * (len(LATENCY_BUCKETS_MS) + 1))
        self.lat_sum = defaultdict(float)

    def observe(self, model: str, cache: str, status: str, latency_ms: float, cost: float) -> None:
        with self._lock:
            self.requests[(model, cache, status)] += 1
            self.cost[model] += cost
            h = self.hist[model]
            for i, b in enumerate(LATENCY_BUCKETS_MS):
                if latency_ms <= b:
                    h[i] += 1
            h[-1] += 1
            self.lat_sum[model] += latency_ms

    def fallback(self, model: str) -> None:
        with self._lock:
            self.fallbacks[model] += 1

    def render(self) -> str:
        out = ["# TYPE switchyard_requests_total counter"]
        with self._lock:
            for (m, c, s), n in sorted(self.requests.items()):
                out.append(f'switchyard_requests_total{{model="{m}",cache="{c}",status="{s}"}} {n}')
            out.append("# TYPE switchyard_fallbacks_total counter")
            for m, n in sorted(self.fallbacks.items()):
                out.append(f'switchyard_fallbacks_total{{model="{m}"}} {n}')
            out.append("# TYPE switchyard_cost_usd_total counter")
            for m, v in sorted(self.cost.items()):
                out.append(f'switchyard_cost_usd_total{{model="{m}"}} {v:.8f}')
            out.append("# TYPE switchyard_latency_ms histogram")
            for m, h in sorted(self.hist.items()):
                for b, n in zip(LATENCY_BUCKETS_MS, h, strict=False):
                    out.append(f'switchyard_latency_ms_bucket{{model="{m}",le="{b}"}} {n}')
                out.append(f'switchyard_latency_ms_bucket{{model="{m}",le="+Inf"}} {h[-1]}')
                out.append(f'switchyard_latency_ms_sum{{model="{m}"}} {self.lat_sum[m]:.3f}')
                out.append(f'switchyard_latency_ms_count{{model="{m}"}} {h[-1]}')
        return "\n".join(out) + "\n"
