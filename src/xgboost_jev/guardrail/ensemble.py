"""Pattern 2 - ensemble hardening against counterfeit policies.

Jev reads text the way an LLM does, so a forged "[SYSTEM]: policy updated"
prefix can talk its probability down. A booster over static embeddings and
metadata does not follow instructions, so it is not steerable that way.

The two run in parallel as voters with asymmetric trust:
  * either voter can escalate on its own when it is confident;
  * an XGBoost "attack" verdict (at its own tuned threshold) is never lowered
    by a low Jev score;
  * when counterfeit-policy markers are present, Jev's weight is cut.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from ..jev.client import JevClient
from ..jev.schema import JevQuery, Question
from .cascade import GuardrailModel
from .decision import Action, GuardDecision
from .features import counterfeit_policy_features
from .questions import INJECTION_QUESTIONS

_RISK_IDS = (
    "p_override",
    "p_jailbreak",
    "p_roleplay",
    "p_exfiltration",
    "p_obfuscation",
    "p_tool_abuse",
)


@dataclass
class EnsembleConfig:
    block_threshold: float = 0.5
    review_threshold: float = 0.3
    xgb_override: float = 0.5  # on the threshold-aligned scale: XGB's own "attack" verdict
    jev_confident: float = 0.9
    jev_weight: float = 0.4
    xgb_weight: float = 0.6
    counterfeit_trigger: float = 0.3
    counterfeit_jev_penalty: float = 0.75


def jev_risk(probs: dict[str, float]) -> float:
    """Noisy-OR over the per-category injection probabilities (NaN ignored)."""
    vals = [probs[k] for k in _RISK_IDS if k in probs and not math.isnan(probs[k])]
    if not vals:
        return math.nan
    keep = 1.0
    for v in vals:
        keep *= 1.0 - v
    return 1.0 - keep


class EnsembleGuard:
    def __init__(
        self,
        static_model: GuardrailModel,
        jev: JevClient | None,
        config: EnsembleConfig | None = None,
        questions: tuple[Question, ...] = INJECTION_QUESTIONS,
    ) -> None:
        if static_model.builder.use_jev:
            raise ValueError("the ensemble's XGBoost voter must not consume Jev features")
        self.model = static_model
        self.jev = jev
        self.config = config or EnsembleConfig()
        self.questions = questions
        self._pool = ThreadPoolExecutor(max_workers=8)

    def align_xgb(self, p_xgb: float) -> float:
        """Map XGBoost's probability so its own tuned threshold lands on 0.5.

        Without this, a model tuned to block at e.g. 0.3 would be judged against
        the ensemble's 0.5 line, letting a low Jev vote silently veto it.
        """
        t = self.model.threshold
        if p_xgb < t:
            return 0.5 * p_xgb / t
        return 0.5 + 0.5 * (p_xgb - t) / max(1.0 - t, 1e-9)

    def combine(self, p_jev: float, p_xgb: float, cf_score: float) -> tuple[float, list[str]]:
        """Return (score, reasons) on a scale where 0.5 is the block line."""
        c = self.config
        x = self.align_xgb(p_xgb)
        if math.isnan(p_jev):
            return x, ["jev_unavailable"]
        reasons: list[str] = []
        w_j = c.jev_weight
        if cf_score >= c.counterfeit_trigger:
            w_j *= 1.0 - c.counterfeit_jev_penalty
            reasons.append("counterfeit_policy_suspected")
        score = (w_j * p_jev + c.xgb_weight * x) / (w_j + c.xgb_weight)
        if x >= c.xgb_override and score < x:
            score = x
            if p_jev < c.block_threshold:
                reasons.append("xgb_override")
        if p_jev >= c.jev_confident and score < p_jev:
            score = p_jev
            reasons.append("jev_confident")
        return score, reasons

    def check(self, text: str) -> GuardDecision:
        start = time.perf_counter()
        future = (
            self._pool.submit(self.jev.ask, JevQuery(text, self.questions)) if self.jev else None
        )
        p_xgb = self.model.score(text)
        cf = counterfeit_policy_features(text)["cf_score"]
        probs = future.result().probabilities if future else {}
        p_jev = jev_risk(dict(probs))
        score, reasons = self.combine(p_jev, p_xgb, cf)
        c = self.config
        action = (
            Action.BLOCK
            if score >= c.block_threshold
            else Action.REVIEW
            if score >= c.review_threshold
            else Action.ALLOW
        )
        return GuardDecision(
            action,
            score,
            reasons,
            {"p_jev": p_jev, "p_xgb": p_xgb, "cf_score": cf},
            "ensemble",
            (time.perf_counter() - start) * 1000,
        )

    def score(self, text: str) -> float:
        return self.check(text).score
