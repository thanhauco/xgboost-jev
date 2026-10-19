"""Thread-safe latency recorder with percentile summaries."""

from __future__ import annotations

import threading
from collections import defaultdict, deque

import numpy as np


class LatencyRecorder:
    def __init__(self, window: int = 10_000) -> None:
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=window))
        self._counts: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def record(self, name: str, ms: float) -> None:
        with self._lock:
            self._samples[name].append(ms)
            self._counts[name] += 1

    def summary(self) -> dict[str, dict[str, float]]:
        with self._lock:
            items = {k: np.array(v) for k, v in self._samples.items()}
            counts = dict(self._counts)
        return {k: summarize(v) | {"count": counts[k]} for k, v in items.items() if len(v)}


def summarize(ms: np.ndarray) -> dict[str, float]:
    ms = np.asarray(ms, dtype=float)
    return {
        "mean_ms": round(float(ms.mean()), 3),
        "p50_ms": round(float(np.percentile(ms, 50)), 3),
        "p95_ms": round(float(np.percentile(ms, 95)), 3),
        "p99_ms": round(float(np.percentile(ms, 99)), 3),
    }
