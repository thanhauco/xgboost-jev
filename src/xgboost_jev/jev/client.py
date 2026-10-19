"""HTTP client for Jev with deadlines, bounded retries and concurrent fan-out."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Protocol, runtime_checkable

import httpx

from .schema import JevQuery, JevResult, parse_probabilities


class JevError(RuntimeError):
    pass


@runtime_checkable
class JevClient(Protocol):
    def ask(self, query: JevQuery) -> JevResult: ...

    def ask_many(self, queries: Sequence[JevQuery]) -> list[JevResult]: ...


class BaseJevClient:
    """Implements ``ask_many`` as a thread-pool fan-out over ``ask``."""

    max_concurrency: int = 16

    def ask(self, query: JevQuery) -> JevResult:  # pragma: no cover - abstract
        raise NotImplementedError

    def ask_many(self, queries: Sequence[JevQuery]) -> list[JevResult]:
        if len(queries) <= 1:
            return [self.ask(q) for q in queries]
        workers = min(self.max_concurrency, len(queries))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.ask, queries))


class HttpJevClient(BaseJevClient):
    """POSTs ``{"state", "questions"}`` to a Jev endpoint.

    Configure via arguments or ``JEV_API_URL`` / ``JEV_API_KEY``. Raises
    :class:`JevError` on failure; wrap in ``ResilientJevClient`` for fallbacks.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        timeout_s: float = 0.25,
        retries: int = 1,
        backoff_s: float = 0.02,
        transport: httpx.BaseTransport | None = None,
        max_concurrency: int = 16,
    ) -> None:
        self.url = url or os.environ.get("JEV_API_URL")
        if not self.url:
            raise ValueError("Jev URL missing: pass url= or set JEV_API_URL")
        key = api_key or os.environ.get("JEV_API_KEY")
        headers = {"Content-Type": "application/json"}
        if key:
            headers["Authorization"] = f"Bearer {key}"
        self.timeout_s = timeout_s
        self.retries = retries
        self.backoff_s = backoff_s
        self.max_concurrency = max_concurrency
        self._http = httpx.Client(headers=headers, timeout=timeout_s, transport=transport)

    def ask(self, query: JevQuery) -> JevResult:
        start = time.perf_counter()
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                resp = self._http.post(self.url, json=query.to_payload())
                if resp.status_code >= 500:
                    raise JevError(f"Jev returned HTTP {resp.status_code}")
                resp.raise_for_status()
                probs = parse_probabilities(resp.json(), query.question_ids)
                return JevResult(probs, (time.perf_counter() - start) * 1000, "jev")
            except (httpx.HTTPError, JevError, ValueError) as exc:
                last_error = exc
                retryable = not (
                    isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500
                )
                if attempt < self.retries and retryable:
                    time.sleep(self.backoff_s * (2**attempt))
                    continue
                break
        raise JevError(f"Jev request failed: {last_error}") from last_error

    def close(self) -> None:
        self._http.close()
