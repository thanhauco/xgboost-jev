"""Probability calibration and per-segment decision thresholds.

``scale_pos_weight`` makes boosters rank well but inflates raw scores, so the
"expected loss = p x amount" arithmetic is wrong until scores are calibrated.
Isotonic regression is fitted on a held-out, time-later slice and stored as a
piecewise-linear map (JSON-serialisable, applied with ``np.interp``).

A $15 false decline is not the same cost for every customer: segments
(channel x tenure) get their own cost-optimal thresholds, falling back to the
global policy when a segment has too few positives to tune on.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.isotonic import IsotonicRegression

from .scorer import DecisionPolicy, choose_decline_threshold


@dataclass
class IsotonicCalibrator:
    x: list[float] = field(default_factory=lambda: [0.0, 1.0])
    y: list[float] = field(default_factory=lambda: [0.0, 1.0])

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray) -> IsotonicCalibrator:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(scores, labels)
        return cls([float(v) for v in iso.X_thresholds_], [float(v) for v in iso.y_thresholds_])

    def __call__(self, scores):
        return np.interp(scores, self.x, self.y)

    def to_dict(self) -> dict:
        return asdict(self)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    """ECE with equal-mass bins (robust when scores pile up near 0)."""
    order = np.argsort(p)
    err = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            err += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(err / max(len(p), 1))


def tenure_bucket(account_age_days: float) -> str:
    return (
        "new"
        if (account_age_days is None or math.isnan(account_age_days) or account_age_days < 30)
        else "est"
    )


def segment_of(channel: str, account_age_days: float) -> str:
    return f"{channel}|{tenure_bucket(account_age_days)}"


DEFAULT_FALSE_DECLINE_COST = {"est": 25.0, "new": 8.0}


@dataclass
class SegmentedPolicy:
    default: DecisionPolicy
    segments: dict[str, DecisionPolicy] = field(default_factory=dict)

    def for_segment(self, seg: str) -> DecisionPolicy:
        return self.segments.get(seg, self.default)

    def to_dict(self) -> dict:
        return {
            "default": asdict(self.default),
            "segments": {k: asdict(v) for k, v in self.segments.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> SegmentedPolicy:
        return cls(
            DecisionPolicy(**d["default"]),
            {k: DecisionPolicy(**v) for k, v in d["segments"].items()},
        )

    @classmethod
    def fit(
        cls,
        y: np.ndarray,
        p: np.ndarray,
        amounts: np.ndarray,
        segments: list[str],
        cost_for: Callable[[str], float] | None = None,
        min_pos: int = 5,
        min_rows: int = 200,
    ) -> SegmentedPolicy:
        cost_for = cost_for or (lambda s: DEFAULT_FALSE_DECLINE_COST[s.split("|")[1]])
        t, _ = choose_decline_threshold(y, p, amounts, 15.0)
        out = cls(_ladder(t))
        segs = np.array(segments)
        for s in sorted(set(segments)):
            m = segs == s
            if m.sum() >= min_rows and y[m].sum() >= min_pos:
                ts, _ = choose_decline_threshold(y[m], p[m], amounts[m], cost_for(s))
                out.segments[s] = _ladder(ts)
        return out

    def expected_cost(self, y, p, amounts, segments, cost_for=None) -> float:
        cost_for = cost_for or (lambda s: DEFAULT_FALSE_DECLINE_COST[s.split("|")[1]])
        total = 0.0
        for yi, pi, ai, si in zip(y, p, amounts, segments, strict=True):
            declined = pi >= self.for_segment(si).decline_at
            total += (
                ai
                if (yi == 1 and not declined)
                else (cost_for(si) if (yi == 0 and declined) else 0.0)
            )
        return float(total)


def _ladder(decline_at: float) -> DecisionPolicy:
    return DecisionPolicy(decline_at / 3, decline_at * 0.6, decline_at)
