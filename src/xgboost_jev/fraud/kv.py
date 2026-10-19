"""Minimal key-value backend so online state can live outside the process.

State objects (decayed counters, profiles) are small dicts of floats. The
in-memory backend is the default; ``RedisKV`` stores each key as a Redis hash
so several scorer replicas can share state. Both satisfy :class:`KVBackend`.
"""

from __future__ import annotations

import threading
from typing import Protocol


class KVBackend(Protocol):
    def get(self, key: str) -> dict[str, float] | None: ...

    def put(self, key: str, value: dict[str, float]) -> None: ...


class InMemoryKV:
    def __init__(self) -> None:
        self._d: dict[str, dict[str, float]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, float] | None:
        with self._lock:
            v = self._d.get(key)
            return dict(v) if v is not None else None

    def put(self, key: str, value: dict[str, float]) -> None:
        with self._lock:
            self._d[key] = dict(value)

    def __len__(self) -> int:
        return len(self._d)


class RedisKV:
    """Redis hash per key. ``client`` is a ``redis.Redis``-compatible object."""

    def __init__(self, client, prefix: str = "xgbjev:", ttl_s: int | None = 90 * 86400) -> None:
        self.r = client
        self.prefix = prefix
        self.ttl_s = ttl_s

    def get(self, key: str) -> dict[str, float] | None:
        raw = self.r.hgetall(self.prefix + key)
        if not raw:
            return None
        return {(k.decode() if isinstance(k, bytes) else k): float(v) for k, v in raw.items()}

    def put(self, key: str, value: dict[str, float]) -> None:
        k = self.prefix + key
        pipe = self.r.pipeline()
        pipe.hset(k, mapping={f: repr(float(v)) for f, v in value.items()})
        if self.ttl_s:
            pipe.expire(k, self.ttl_s)
        pipe.execute()


def partition_for(key: str, n_partitions: int) -> int:
    """Stable partition for a routing key (e.g. card_id) - same on every host."""
    import hashlib

    return (
        int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), "big") % n_partitions
    )
