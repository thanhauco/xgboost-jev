"""Pattern 3 - multi-stage latency routing.

Stage 1: one Jev ``is_malicious`` noul question. Extreme probabilities short
         circuit (allow < 0.05, block > 0.95).
Stage 2: anything ambiguous, Jev failures, and texts carrying counterfeit-policy
         markers go to the deep check (embeddings + XGBoost, or the ensemble).

The counterfeit guard closes the obvious bypass: an attacker who tricks Stage 1
into a near-zero score still cannot earn a fast-path ALLOW.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Protocol

from ..jev.client import JevClient
from ..jev.schema import JevQuery
from .decision import Action, GuardDecision
from .features import counterfeit_policy_features
from .questions import PREFILTER_QUESTION


class Scorer(Protocol):
    def score(self, text: str) -> float: ...


@dataclass
class RouterConfig:
    allow_below: float = 0.05
    block_above: float = 0.95
    block_threshold: float = 0.5
    review_threshold: float = 0.3
    counterfeit_trigger: float = 0.3


@dataclass
class RouterStats:
    total: int = 0
    stage1_allow: int = 0
    stage1_block: int = 0
    stage2: int = 0
    counterfeit_escalations: int = 0
    jev_failures: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def bump(self, name: str) -> None:
        with self._lock:
            setattr(self, name, getattr(self, name) + 1)

    @property
    def short_circuit_rate(self) -> float:
        return (self.stage1_allow + self.stage1_block) / max(self.total, 1)


class LatencyRouter:
    def __init__(self, jev: JevClient, deep: Scorer, config: RouterConfig | None = None) -> None:
        self.jev = jev
        self.deep = deep
        self.config = config or RouterConfig()
        self.stats = RouterStats()

    def check(self, text: str) -> GuardDecision:
        c = self.config
        start = time.perf_counter()
        self.stats.bump("total")
        try:
            p1 = self.jev.ask(JevQuery(text, (PREFILTER_QUESTION,))).get(PREFILTER_QUESTION.id)
        except Exception:  # noqa: BLE001 - any Jev failure routes to stage 2
            p1 = math.nan
        cf = counterfeit_policy_features(text)["cf_score"]
        reasons: list[str] = []

        if math.isnan(p1):
            self.stats.bump("jev_failures")
            reasons.append("jev_unavailable")
        elif p1 >= c.block_above:
            self.stats.bump("stage1_block")
            return GuardDecision(
                Action.BLOCK,
                p1,
                ["stage1_block"],
                {"p_stage1": p1, "cf_score": cf},
                "stage1",
                (time.perf_counter() - start) * 1000,
            )
        elif p1 <= c.allow_below and cf < c.counterfeit_trigger:
            self.stats.bump("stage1_allow")
            return GuardDecision(
                Action.ALLOW,
                p1,
                ["stage1_allow"],
                {"p_stage1": p1, "cf_score": cf},
                "stage1",
                (time.perf_counter() - start) * 1000,
            )
        elif p1 <= c.allow_below:
            self.stats.bump("counterfeit_escalations")
            reasons.append("counterfeit_policy_suspected")

        self.stats.bump("stage2")
        p2 = self.deep.score(text)
        action = (
            Action.BLOCK
            if p2 >= c.block_threshold
            else Action.REVIEW
            if p2 >= c.review_threshold
            else Action.ALLOW
        )
        return GuardDecision(
            action,
            p2,
            reasons + ["stage2"],
            {"p_stage1": p1, "p_stage2": p2, "cf_score": cf},
            "stage2",
            (time.perf_counter() - start) * 1000,
        )

    def score(self, text: str) -> float:
        return self.check(text).score
