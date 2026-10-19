"""Population Stability Index drift monitoring for features and scores."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """PSI with quantile bins from ``expected``; NaNs get their own bucket.

    Rule of thumb: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 major shift.
    """
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    e_fin, a_fin = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = (
        np.unique(np.quantile(e_fin, np.linspace(0, 1, bins + 1)))
        if len(e_fin)
        else np.array([0.0])
    )
    edges[0], edges[-1] = -np.inf, np.inf

    def dist(x_fin, x_all):
        counts = np.histogram(x_fin, edges)[0] if len(edges) > 1 else np.array([len(x_fin)])
        counts = np.append(counts, np.isnan(x_all).sum())
        return np.clip(counts / max(len(x_all), 1), eps, None)

    e, a = dist(e_fin, expected), dist(a_fin, actual)
    return float(np.sum((a - e) * np.log(a / e)))


class DriftMonitor:
    def __init__(
        self, reference: Mapping[str, np.ndarray], window: int = 5000, alert_at: float = 0.25
    ):
        self.reference = {k: np.asarray(v, dtype=float) for k, v in reference.items()}
        self.recent = {k: deque(maxlen=window) for k in self.reference}
        self.alert_at = alert_at

    def update(self, row: Mapping[str, float]) -> None:
        for k, q in self.recent.items():
            if k in row:
                q.append(row[k])

    def report(self, min_samples: int = 200) -> dict[str, dict]:
        out = {}
        for k, q in self.recent.items():
            if len(q) < min_samples:
                continue
            v = psi(self.reference[k], np.array(q))
            out[k] = {"psi": round(v, 4), "alert": v >= self.alert_at, "n": len(q)}
        return out
