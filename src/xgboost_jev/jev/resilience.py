"""Caching, circuit breaking and graceful degradation around any JevClient.

When Jev is down the wrapper returns NaN probabilities instead of raising.
XGBoost routes NaN down each split's learned default branch, so the downstream
model keeps scoring on its metadata and embedding features.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict

from .client import BaseJevClient, JevClient
from .schema import JevQuery, JevResult


class TTLCache:
    def __init__(self, max_items: int = 10_000, ttl_s: float = 300.0) -> None:
        self.max_items = max_items
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, JevResult]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> JevResult | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                return None
            ts, value = item
            if time.monotonic() - ts > self.ttl_s:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return value

    def put(self, key: str, value: JevResult) -> None:
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.max_items:
                self._data.popitem(last=False)

    def __len__(self) -> int:
        return len(self._data)


class CircuitBreaker:
    """Opens after ``failure_threshold`` consecutive failures, half-opens after ``reset_s``."""

    def __init__(self, failure_threshold: int = 5, reset_s: float = 10.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_s = reset_s
        self._failures = 0
        self._opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            if time.monotonic() - self._opened_at >= self.reset_s:
                return "half_open"
            return "open"

    def allow(self) -> bool:
        return self.state != "open"

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            if self._failures >= self.failure_threshold:
                self._opened_at = time.monotonic()


class ResilientJevClient(BaseJevClient):
    def __init__(
        self,
        primary: JevClient,
        fallback: JevClient | None = None,
        cache: TTLCache | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.cache = cache if cache is not None else TTLCache()
        self.breaker = breaker if breaker is not None else CircuitBreaker()

    def _degrade(self, query: JevQuery, error: str) -> JevResult:
        if self.fallback is not None:
            try:
                r = self.fallback.ask(query)
                return JevResult(r.probabilities, r.latency_ms, "fallback", error)
            except Exception as exc:  # noqa: BLE001 - fallback must never raise
                error = f"{error}; fallback: {exc}"
        return JevResult.unavailable(query.question_ids, error)

    def ask(self, query: JevQuery) -> JevResult:
        key = query.cache_key()
        cached = self.cache.get(key)
        if cached is not None:
            return JevResult(cached.probabilities, 0.0, "cache")
        if not self.breaker.allow():
            return self._degrade(query, "circuit open")
        try:
            result = self.primary.ask(query)
        except Exception as exc:  # noqa: BLE001
            self.breaker.record_failure()
            return self._degrade(query, str(exc))
        self.breaker.record_success()
        self.cache.put(key, result)
        return result
