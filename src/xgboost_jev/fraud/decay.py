"""Exponentially decayed counters: O(1) memory and O(1) reads per entity.

Instead of keeping every event in a deque, each entity stores, per half-life h,
a decayed count and a decayed sum::

    value <- value * 2^(-dt / h) + x

A 30-day baseline costs the same three floats as a 1-hour one, and the state is
a flat dict, so it maps directly onto a Redis hash (see ``kv.RedisKV``).
"""

from __future__ import annotations

import math

from .kv import InMemoryKV, KVBackend

HALF_LIVES_S = {"1h": 3600.0, "1d": 86400.0, "7d": 7 * 86400.0, "30d": 30 * 86400.0}


class DecayedCounterStore:
    def __init__(
        self, kv: KVBackend | None = None, half_lives: dict[str, float] | None = None
    ) -> None:
        self.kv = kv or InMemoryKV()
        self.half_lives = half_lives or HALF_LIVES_S

    def read(self, key: str, now: float) -> dict[str, float]:
        """Decayed ``cnt_<h>`` / ``sum_<h>`` as of ``now`` (zeros for unseen keys)."""
        state = self.kv.get(key)
        out = {}
        for name, hl in self.half_lives.items():
            if state is None:
                out[f"cnt_{name}"] = out[f"sum_{name}"] = 0.0
                continue
            f = 2.0 ** (-max(now - state["t"], 0.0) / hl)
            out[f"cnt_{name}"] = state[f"cnt_{name}"] * f
            out[f"sum_{name}"] = state[f"sum_{name}"] * f
        return out

    def add(self, key: str, now: float, value: float = 1.0) -> None:
        cur = self.read(key, now)
        new = {"t": now}
        for name in self.half_lives:
            new[f"cnt_{name}"] = cur[f"cnt_{name}"] + 1.0
            new[f"sum_{name}"] = cur[f"sum_{name}"] + value
        self.kv.put(key, new)


class DecayedMeanVar:
    """Exponentially weighted mean / variance (e.g. per peer group)."""

    def __init__(self, kv: KVBackend | None = None, half_life_s: float = 30 * 86400.0) -> None:
        self.kv = kv or InMemoryKV()
        self.hl = half_life_s

    def read(self, key: str, now: float) -> tuple[float, float, float]:
        s = self.kv.get(key)
        if s is None:
            return 0.0, math.nan, math.nan
        f = 2.0 ** (-max(now - s["t"], 0.0) / self.hl)
        w = s["w"] * f
        if w < 1e-9:
            return 0.0, math.nan, math.nan
        mean = s["m1"] / s["w"]
        var = max(s["m2"] / s["w"] - mean * mean, 0.0)
        return w, mean, var

    def add(self, key: str, now: float, x: float) -> None:
        s = self.kv.get(key)
        f = 1.0 if s is None else 2.0 ** (-max(now - s["t"], 0.0) / self.hl)
        s = s or {"w": 0.0, "m1": 0.0, "m2": 0.0}
        self.kv.put(
            key, {"t": now, "w": s["w"] * f + 1, "m1": s["m1"] * f + x, "m2": s["m2"] * f + x * x}
        )
