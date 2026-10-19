"""Real-time fraud scorer with tiered Jev escalation.

    txn -> online features (sub-ms) -> tabular XGBoost -> p_tab
             |  escalate only when text exists AND (p_tab is ambiguous OR channel is P2P)
             +-> Jev semantic questions -> hybrid XGBoost(tabular + Jev) -> p
    p -> that model's cost-based policy -> APPROVE / STEP_UP / REVIEW / DECLINE (+ reason codes)

The fast path never waits on the network. Jev is spent only where tabular
signals are weak, which keeps p99 latency and Jev cost bounded.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from ..jev.client import JevClient
from .explain import reason_codes
from .features import FEATURE_NAMES, TransactionFeaturizer
from .mock_data import LabeledStream
from .schema import Transaction
from .semantic import SEMANTIC_NAMES, FraudSemanticExtractor, build_state

HYBRID_NAMES = FEATURE_NAMES + SEMANTIC_NAMES

PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 5,
    "eta": 0.08,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 1,
    "tree_method": "hist",
    "nthread": 1,
}


class FraudAction(str, Enum):
    APPROVE = "approve"
    STEP_UP = "step_up"  # e.g. 3-D Secure / OTP challenge
    REVIEW = "review"
    DECLINE = "decline"


@dataclass
class DecisionPolicy:
    step_up_at: float = 0.2
    review_at: float = 0.5
    decline_at: float = 0.8

    def decide(self, p: float) -> FraudAction:
        if p >= self.decline_at:
            return FraudAction.DECLINE
        if p >= self.review_at:
            return FraudAction.REVIEW
        if p >= self.step_up_at:
            return FraudAction.STEP_UP
        return FraudAction.APPROVE


@dataclass
class EscalationPolicy:
    band: tuple[float, float] = (0.02, 0.9)
    always_channels: frozenset = frozenset({"p2p"})
    min_amount: float = 20.0

    def should_escalate(self, t: Transaction, p_tab: float) -> bool:
        if build_state(t) is None or t.amount < self.min_amount:
            return False
        return t.channel in self.always_channels or self.band[0] <= p_tab <= self.band[1]


@dataclass
class FraudDecision:
    txn_id: str
    score: float
    action: FraudAction
    model: str
    p_tabular: float
    jev_used: bool
    reasons: list[dict] = field(default_factory=list)
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


def choose_decline_threshold(
    y: np.ndarray, scores: np.ndarray, amounts: np.ndarray, false_decline_cost: float = 15.0
) -> tuple[float, float]:
    """Pick the threshold minimising missed-fraud dollars + false-decline cost."""
    best_t, best_cost = 0.5, math.inf
    for t in np.linspace(0.05, 0.95, 91):
        flagged = scores >= t
        cost = amounts[(y == 1) & ~flagged].sum() + false_decline_cost * ((y == 0) & flagged).sum()
        if cost < best_cost:
            best_t, best_cost = float(t), float(cost)
    return best_t, best_cost


@dataclass
class FraudModels:
    """Two boosters, each with its own cost-tuned policy.

    The models' score distributions differ (the hybrid sees extra evidence), so a
    threshold tuned on one must never be applied to the other.
    """

    tabular: xgb.Booster
    hybrid: xgb.Booster
    tabular_policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    hybrid_policy: DecisionPolicy = field(default_factory=DecisionPolicy)
    metrics: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tabular.save_model(path / "tabular.json")
        self.hybrid.save_model(path / "hybrid.json")
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "tabular_policy": asdict(self.tabular_policy),
                    "hybrid_policy": asdict(self.hybrid_policy),
                    "metrics": self.metrics,
                    "tabular_features": FEATURE_NAMES,
                    "hybrid_features": HYBRID_NAMES,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> FraudModels:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        tab, hyb = xgb.Booster(), xgb.Booster()
        tab.load_model(path / "tabular.json")
        hyb.load_model(path / "hybrid.json")
        return cls(
            tab,
            hyb,
            DecisionPolicy(**meta["tabular_policy"]),
            DecisionPolicy(**meta["hybrid_policy"]),
            meta["metrics"],
        )


@dataclass
class ReplayedFeatures:
    tabular: np.ndarray
    semantic: np.ndarray
    labels: np.ndarray
    amounts: np.ndarray
    scenarios: list[str]
    timestamps: np.ndarray


def replay(stream: LabeledStream, jev: JevClient | None) -> ReplayedFeatures:
    """Point-in-time feature replay: compute, then observe, in timestamp order."""
    fz = TransactionFeaturizer()
    rows = []
    for t in stream.transactions:
        rows.append(fz.vector(fz.compute(t)))
        fz.observe(t)
    sem = (
        np.array(FraudSemanticExtractor(jev).extract_many(stream.transactions), dtype=np.float32)
        if jev
        else np.full((len(rows), len(SEMANTIC_NAMES)), np.nan, dtype=np.float32)
    )
    txns = stream.transactions
    return ReplayedFeatures(
        np.vstack(rows),
        sem,
        np.array([stream.labels[t.txn_id] for t in txns]),
        np.array([t.amount for t in txns]),
        [stream.scenarios[t.txn_id] for t in txns],
        np.array([t.timestamp for t in txns]),
    )


def _policy_from_cost(y, scores, amounts) -> DecisionPolicy:
    decline_at, _ = choose_decline_threshold(y, scores, amounts)
    return DecisionPolicy(decline_at / 3, decline_at * 0.6, decline_at)


def _fit(x, y, names, x_va, y_va, rounds, seed):
    p = {**PARAMS, "seed": seed, "scale_pos_weight": float((y == 0).sum() / max((y == 1).sum(), 1))}
    dtr = xgb.DMatrix(x, label=y, feature_names=names)
    dva = xgb.DMatrix(x_va, label=y_va, feature_names=names)
    b = xgb.train(
        p, dtr, rounds, evals=[(dva, "valid")], early_stopping_rounds=40, verbose_eval=False
    )
    return b[: b.best_iteration + 1]


def train_fraud_models(
    stream: LabeledStream,
    jev: JevClient,
    train_frac: float = 0.7,
    jev_dropout: float = 0.1,
    rounds: int = 400,
    seed: int = 0,
) -> tuple[FraudModels, ReplayedFeatures, np.ndarray]:
    """Time-ordered split (train on the past, validate on the future)."""
    f = replay(stream, jev)
    cut = int(len(f.labels) * train_frac)
    tr, va = slice(0, cut), slice(cut, None)
    xh = np.hstack([f.tabular, f.semantic])
    xh_tr = xh[tr].copy()
    rng = np.random.default_rng(seed)
    xh_tr[rng.random(cut) < jev_dropout, len(FEATURE_NAMES) :] = np.nan
    tab = _fit(
        f.tabular[tr], f.labels[tr], FEATURE_NAMES, f.tabular[va], f.labels[va], rounds, seed
    )
    hyb = _fit(xh_tr, f.labels[tr], HYBRID_NAMES, xh[va], f.labels[va], rounds, seed)
    p_tab = tab.inplace_predict(f.tabular[va])
    p_hyb = hyb.inplace_predict(xh[va])
    y, amt = f.labels[va], f.amounts[va]
    policies = [_policy_from_cost(y, p, amt) for p in (p_tab, p_hyb)]
    metrics = {
        "tabular": {
            "roc_auc": float(roc_auc_score(y, p_tab)),
            "pr_auc": float(average_precision_score(y, p_tab)),
        },
        "hybrid": {
            "roc_auc": float(roc_auc_score(y, p_hyb)),
            "pr_auc": float(average_precision_score(y, p_hyb)),
        },
        "n_train": cut,
        "n_valid": int(len(y)),
        "valid_fraud": int(y.sum()),
    }
    return FraudModels(tab, hyb, *policies, metrics), f, np.arange(cut, len(f.labels))


class RealTimeFraudScorer:
    def __init__(
        self,
        models: FraudModels,
        jev: JevClient | None,
        featurizer: TransactionFeaturizer | None = None,
        escalation: EscalationPolicy | None = None,
        explain: bool = True,
    ) -> None:
        self.models = models
        self.featurizer = featurizer or TransactionFeaturizer()
        self.extractor = FraudSemanticExtractor(jev) if jev else None
        self.escalation = escalation or EscalationPolicy()
        self.explain = explain

    def _finish(self, t, x_tab, p_tab, sem, start) -> FraudDecision:
        if sem is not None:
            x = np.concatenate([x_tab, np.asarray(sem, dtype=np.float32)])
            booster, names, model = self.models.hybrid, HYBRID_NAMES, "hybrid"
            policy = self.models.hybrid_policy
            p = float(booster.inplace_predict(x.reshape(1, -1))[0])
        else:
            x, booster, names, model, p = (
                x_tab,
                self.models.tabular,
                FEATURE_NAMES,
                "tabular",
                p_tab,
            )
            policy = self.models.tabular_policy
        action = policy.decide(p)
        reasons = (
            reason_codes(booster, x, names)
            if self.explain and action != FraudAction.APPROVE
            else []
        )
        return FraudDecision(
            t.txn_id,
            p,
            action,
            model,
            p_tab,
            sem is not None,
            reasons,
            (time.perf_counter() - start) * 1000,
        )

    def score(self, t: Transaction) -> FraudDecision:
        start = time.perf_counter()
        x_tab = self.featurizer.vector(self.featurizer.compute(t))
        self.featurizer.observe(t)
        p_tab = float(self.models.tabular.inplace_predict(x_tab.reshape(1, -1))[0])
        sem = None
        if self.extractor and self.escalation.should_escalate(t, p_tab):
            sem = self.extractor.extract(t)
        return self._finish(t, x_tab, p_tab, sem, start)

    def score_batch(self, txns: Sequence[Transaction]) -> list[FraudDecision]:
        """State updates stay in order; escalated Jev calls fan out concurrently."""
        start = time.perf_counter()
        xs = []
        for t in txns:
            xs.append(self.featurizer.vector(self.featurizer.compute(t)))
            self.featurizer.observe(t)
        x_tab = np.vstack(xs) if xs else np.empty((0, len(FEATURE_NAMES)), np.float32)
        p_tab = self.models.tabular.inplace_predict(x_tab) if xs else np.array([])
        esc = [
            i
            for i, t in enumerate(txns)
            if self.extractor and self.escalation.should_escalate(t, float(p_tab[i]))
        ]
        sems = (
            dict(zip(esc, self.extractor.extract_many([txns[i] for i in esc]), strict=True))
            if esc
            else {}
        )
        return [
            self._finish(t, x_tab[i], float(p_tab[i]), sems.get(i), start)
            for i, t in enumerate(txns)
        ]
