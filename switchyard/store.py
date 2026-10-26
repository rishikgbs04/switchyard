"""SQLite request log: every request's routing decision, cost and latency.
Powers budgets, /v1/stats and offline analysis (e.g. training a better router)."""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import UTC, datetime

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            REAL NOT NULL,
    api_key       TEXT NOT NULL,
    policy        TEXT NOT NULL,
    model         TEXT,
    difficulty    REAL,
    cache         TEXT,          -- miss | exact | semantic
    attempts      INTEGER NOT NULL,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    cost_usd      REAL NOT NULL DEFAULT 0,
    saved_usd     REAL NOT NULL DEFAULT 0,
    latency_ms    REAL NOT NULL,
    status        TEXT NOT NULL  -- ok | error
);
CREATE INDEX IF NOT EXISTS idx_requests_key_ts ON requests(api_key, ts);
"""


class Store:
    def __init__(self, path: str) -> None:
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._lock = threading.Lock()

    def log(self, **row) -> None:
        row.setdefault("ts", time.time())
        cols = ",".join(row)
        qs = ",".join("?" * len(row))
        with self._lock:
            self._db.execute(f"INSERT INTO requests ({cols}) VALUES ({qs})", tuple(row.values()))
            self._db.commit()

    def month_spend(self, api_key: str) -> float:
        now = datetime.now(UTC)
        start = datetime(now.year, now.month, 1, tzinfo=UTC).timestamp()
        with self._lock:
            (v,) = self._db.execute(
                "SELECT COALESCE(SUM(cost_usd),0) FROM requests WHERE api_key=? AND ts>=?",
                (api_key, start),
            ).fetchone()
        return float(v)

    def stats(self) -> dict:
        with self._lock:
            q = self._db.execute
            total, cost, saved = q(
                "SELECT COUNT(*), COALESCE(SUM(cost_usd),0), COALESCE(SUM(saved_usd),0) FROM requests"
            ).fetchone()
            by_model = q(
                "SELECT model, COUNT(*), ROUND(AVG(latency_ms),1), ROUND(SUM(cost_usd),6) "
                "FROM requests WHERE status='ok' GROUP BY model ORDER BY 2 DESC"
            ).fetchall()
            by_cache = dict(q("SELECT cache, COUNT(*) FROM requests GROUP BY cache").fetchall())
            p = [r[0] for r in q("SELECT latency_ms FROM requests ORDER BY latency_ms").fetchall()]

        def pct(x: float) -> float | None:
            return round(p[min(len(p) - 1, int(len(p) * x))], 1) if p else None

        return {
            "requests": total,
            "cost_usd": round(cost, 6),
            "saved_usd": round(saved, 6),
            "cache": by_cache,
            "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "p99": pct(0.99)},
            "models": [
                {"model": m, "requests": n, "avg_latency_ms": lat, "cost_usd": c}
                for m, n, lat, c in by_model
            ],
        }
