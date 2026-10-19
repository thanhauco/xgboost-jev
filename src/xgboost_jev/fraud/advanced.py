"""Advanced fraud models: as-of training, calibration, segment policies, online scorer.

Training only sees what was known at training time:
labels come from ``LabelStore.training_labels(as_of)`` (late chargebacks are
missing, young transactions excluded until they mature), and the calibration /
threshold slice is strictly later in time than the training slice.

The online scorer layers, in order:
  rules -> tabular XGBoost (calibrated) -> budget-aware Jev escalation
        -> hybrid XGBoost (calibrated) -> segment policy -> reason codes
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score

from ..jev.client import JevClient
from .calibration import IsotonicCalibrator, SegmentedPolicy, segment_of
from .explain import reason_codes
from .labels import LabelStore, propagate_entity_labels
from .pipeline import ADVANCED_FEATURE_NAMES, AdvancedFeaturizer
from .rules import RulesEngine
from .schema import AccountEvent, LabelEvent, Transaction
from .scorer import PARAMS, EscalationPolicy, FraudAction, FraudDecision
from .semantic import ADVANCED_FRAUD_QUESTIONS, ADVANCED_SEMANTIC_NAMES, FraudSemanticExtractor

# More evidence of fraud must never LOWER the score. Without this, label-derived
# features (merchant risk, confirmed-fraud flags) are learned as narrow bands from
# training rows where few labels had arrived yet, and the model then mis-scores the
# higher values it meets online once labels are in.
MONOTONE_UP = (
    "r_merchant",
    "r_mcc",
    "g_device_fraud",
    "g_ip_fraud",
    "g_card_fraud",
    "g_comp_fraud",
    "g_2hop_fraud",
    "g_payee_fraud",
    "g_payee_comp_fraud",
    "desc_brand_lookalike",
    "desc_suspicious_tokens",
    "desc_leet",
    "dev_emulator",
    "dev_rooted",
    "seq_reset_then_payee_48h",
    "high_risk_mcc",
)
MONOTONE_DOWN = ("cop_similarity",)


def monotone_constraints(names: list[str]) -> str:
    def c(n: str) -> int:
        if n in MONOTONE_UP or n.startswith("jev_"):
            return 1
        return -1 if n in MONOTONE_DOWN else 0

    return "(" + ",".join(str(c(n)) for n in names) + ")"


def fit_booster(
    x, y, names, x_es, y_es, rounds=400, seed=0, xgb_model=None, params=None, monotone=True
) -> xgb.Booster:
    p = {
        **PARAMS,
        **(params or {}),
        "seed": seed,
        "scale_pos_weight": float((y == 0).sum() / max((y == 1).sum(), 1)),
    }
    if monotone:
        p["monotone_constraints"] = monotone_constraints(names)
    dtr = xgb.DMatrix(x, label=y, feature_names=names)
    evals = []
    if x_es is not None and len(y_es) and 0 < y_es.sum() < len(y_es):
        evals = [(xgb.DMatrix(x_es, label=y_es, feature_names=names), "es")]
    b = xgb.train(
        p,
        dtr,
        rounds,
        evals=evals,
        early_stopping_rounds=40 if evals else None,
        verbose_eval=False,
        xgb_model=xgb_model,
    )
    if evals and b.best_iteration is not None:
        b = b[: b.best_iteration + 1]
    return b


@dataclass
class AdvancedModels:
    tabular: xgb.Booster
    hybrid: xgb.Booster
    tabular_names: list[str]
    hybrid_names: list[str]
    tabular_cal: IsotonicCalibrator = field(default_factory=IsotonicCalibrator)
    hybrid_cal: IsotonicCalibrator = field(default_factory=IsotonicCalibrator)
    tabular_policy: SegmentedPolicy | None = None
    hybrid_policy: SegmentedPolicy | None = None
    metrics: dict = field(default_factory=dict)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.tabular.save_model(path / "tabular.json")
        self.hybrid.save_model(path / "hybrid.json")
        (path / "meta.json").write_text(
            json.dumps(
                {
                    "tabular_names": self.tabular_names,
                    "hybrid_names": self.hybrid_names,
                    "tabular_cal": self.tabular_cal.to_dict(),
                    "hybrid_cal": self.hybrid_cal.to_dict(),
                    "tabular_policy": self.tabular_policy.to_dict(),
                    "hybrid_policy": self.hybrid_policy.to_dict(),
                    "metrics": self.metrics,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> AdvancedModels:
        path = Path(path)
        m = json.loads((path / "meta.json").read_text())
        tab, hyb = xgb.Booster(), xgb.Booster()
        tab.load_model(path / "tabular.json")
        hyb.load_model(path / "hybrid.json")
        return cls(
            tab,
            hyb,
            m["tabular_names"],
            m["hybrid_names"],
            IsotonicCalibrator(**m["tabular_cal"]),
            IsotonicCalibrator(**m["hybrid_cal"]),
            SegmentedPolicy.from_dict(m["tabular_policy"]),
            SegmentedPolicy.from_dict(m["hybrid_policy"]),
            m["metrics"],
        )

    def predict_hybrid(
        self, x_full: np.ndarray, sem: np.ndarray, calibrated: bool = True
    ) -> np.ndarray:
        cols = _cols(self.hybrid_names[: len(self.hybrid_names) - sem.shape[1]])
        p = self.hybrid.inplace_predict(np.hstack([x_full[:, cols], sem]))
        return self.hybrid_cal(p) if calibrated else p

    def predict_tabular(self, x_full: np.ndarray, calibrated: bool = True) -> np.ndarray:
        p = self.tabular.inplace_predict(x_full[:, _cols(self.tabular_names)])
        return self.tabular_cal(p) if calibrated else p


_COL = {n: i for i, n in enumerate(ADVANCED_FEATURE_NAMES)}


def _cols(names: list[str]) -> list[int]:
    return [_COL[n] for n in names]


def segments_for(txns: list[Transaction], x_full: np.ndarray) -> list[str]:
    age = x_full[:, _COL["account_age_days"]]
    return [segment_of(t.channel, float(a)) for t, a in zip(txns, age, strict=True)]


def train_advanced(
    x_full: np.ndarray,
    sem: np.ndarray,
    txns: list[Transaction],
    labels: LabelStore,
    as_of: float,
    tabular_names: list[str] | None = None,
    data_until: float | None = None,
    window_days: float | None = None,
    maturity_days: float = 7.0,
    calib_frac: float = 0.2,
    jev_dropout: float = 0.1,
    rounds: int = 400,
    seed: int = 0,
    init: AdvancedModels | None = None,
    update_rounds: int = 120,
    propagate: bool = True,
    monotone: bool = True,
) -> AdvancedModels:
    """Train tabular + hybrid boosters using only labels known at ``as_of``.

    ``data_until`` (default ``as_of``) excludes later rows, e.g. a shadow-evaluation
    slice. ``init`` warm-starts both boosters from a previous champion.
    ``propagate`` adds weak positives by case linking (see ``propagate_entity_labels``).
    """
    tabular_names = tabular_names or ADVANCED_FEATURE_NAMES
    ts = np.array([t.timestamp for t in txns])
    ids = [t.txn_id for t in txns]
    y, usable = labels.training_labels(ids, ts, as_of, maturity_days)
    flipped = 0
    if propagate:
        past = ts < as_of
        y_past, flipped = propagate_entity_labels(
            [t for t, k in zip(txns, past, strict=True) if k],
            y[past],
            labels.fraud_types_as_of(as_of),
        )
        y = y.copy()
        y[past] = y_past
    usable &= ts < (data_until or as_of)
    if window_days:
        usable &= ts >= as_of - window_days * 86400
    idx = np.flatnonzero(usable)  # rows are in time order
    cut = int(len(idx) * (1 - calib_frac))
    tr, ca = idx[:cut], idx[cut:]
    xt = x_full[:, _cols(tabular_names)]
    xh = np.hstack([xt, sem])
    hyb_names = tabular_names + ADVANCED_SEMANTIC_NAMES
    xh_tr = xh[tr].copy()
    rng = np.random.default_rng(seed)
    xh_tr[rng.random(len(tr)) < jev_dropout, len(tabular_names) :] = np.nan
    n_rounds = update_rounds if init else rounds
    tab = fit_booster(
        xt[tr],
        y[tr],
        tabular_names,
        xt[ca],
        y[ca],
        n_rounds,
        seed,
        init.tabular if init else None,
        monotone=monotone,
    )
    hyb = fit_booster(
        xh_tr,
        y[tr],
        hyb_names,
        xh[ca],
        y[ca],
        n_rounds,
        seed,
        init.hybrid if init else None,
        monotone=monotone,
    )
    p_tab, p_hyb = tab.inplace_predict(xt[ca]), hyb.inplace_predict(xh[ca])
    cal_t, cal_h = IsotonicCalibrator.fit(p_tab, y[ca]), IsotonicCalibrator.fit(p_hyb, y[ca])
    seg = segments_for([txns[i] for i in ca], x_full[ca])
    amt = np.array([txns[i].amount for i in ca])
    models = AdvancedModels(
        tab,
        hyb,
        list(tabular_names),
        hyb_names,
        cal_t,
        cal_h,
        SegmentedPolicy.fit(y[ca], cal_t(p_tab), amt, seg),
        SegmentedPolicy.fit(y[ca], cal_h(p_hyb), amt, seg),
        {
            "as_of": as_of,
            "n_train": int(len(tr)),
            "n_calib": int(len(ca)),
            "train_fraud": int(y[tr].sum()),
            "propagated": flipped,
            "calib_pr_auc_hybrid": float(average_precision_score(y[ca], p_hyb))
            if y[ca].sum()
            else math.nan,
        },
    )
    return models


class AdvancedFraudScorer:
    """Online scorer. Feed it account events and labels as they happen, too."""

    def __init__(
        self,
        models: AdvancedModels,
        jev: JevClient | None = None,
        featurizer: AdvancedFeaturizer | None = None,
        rules: RulesEngine | None = None,
        escalation: EscalationPolicy | None = None,
        budget_ms: float | None = None,
        explain: bool = True,
    ) -> None:
        self.models = models
        self.featurizer = featurizer or AdvancedFeaturizer()
        self.extractor = FraudSemanticExtractor(jev, ADVANCED_FRAUD_QUESTIONS) if jev else None
        self.rules = rules or RulesEngine()
        self.escalation = escalation or EscalationPolicy()
        self.budget_ms = budget_ms
        self.explain = explain
        self.jev_latency_est_ms = 0.0
        self._pool = ThreadPoolExecutor(max_workers=8)
        self._tab_idx = np.array(_cols(models.tabular_names))
        self._n_tab_hyb = len(models.hybrid_names) - len(ADVANCED_SEMANTIC_NAMES)
        self._hyb_idx = np.array(_cols(models.hybrid_names[: self._n_tab_hyb]))

    def on_account_event(self, ev: AccountEvent) -> None:
        self.featurizer.on_account_event(ev)

    def on_label(self, ev: LabelEvent) -> None:
        self.featurizer.on_label(ev)

    def _jev(self, t: Transaction, remaining_ms: float | None):
        start = time.perf_counter()
        fut = self._pool.submit(self.extractor.extract, t)
        try:
            sem = fut.result(timeout=None if remaining_ms is None else max(remaining_ms, 0) / 1000)
        except FutureTimeout:
            # Learn from the timeout so later calls skip Jev on budget instead of waiting it out.
            self.jev_latency_est_ms = max(
                self.jev_latency_est_ms, (time.perf_counter() - start) * 1000
            )
            return None, "timeout"
        obs = (time.perf_counter() - start) * 1000
        self.jev_latency_est_ms = (
            obs if self.jev_latency_est_ms == 0 else 0.8 * self.jev_latency_est_ms + 0.2 * obs
        )
        return sem, None

    def score(self, t: Transaction, budget_ms: float | None = None) -> FraudDecision:
        start = time.perf_counter()
        budget = budget_ms if budget_ms is not None else self.budget_ms
        fz, m = self.featurizer, self.models
        feats = fz.compute(t)
        x_full = fz.vector(feats)
        seg = segment_of(t.channel, feats["account_age_days"])
        rule = self.rules.evaluate(t, feats, now=t.timestamp)
        if rule is not None:
            fz.observe(t, feats)
            return FraudDecision(
                t.txn_id,
                1.0 if rule.action != FraudAction.APPROVE else 0.0,
                rule.action,
                "rule",
                math.nan,
                False,
                [{"feature": "rule", "reason": rule.reason or rule.id}],
                (time.perf_counter() - start) * 1000,
                seg,
                rule.id,
            )
        x_tab = x_full[self._tab_idx]
        p_tab = float(m.tabular_cal(m.tabular.inplace_predict(x_tab.reshape(1, -1))[0]))
        sem, skipped = None, None
        if self.extractor and self.escalation.should_escalate(t, p_tab):
            elapsed = (time.perf_counter() - start) * 1000
            if budget is not None and elapsed + self.jev_latency_est_ms * 1.2 > budget:
                skipped = "budget"
            else:
                sem, skipped = self._jev(t, None if budget is None else budget - elapsed)
        if sem is not None:
            x = np.concatenate([x_full[self._hyb_idx], np.asarray(sem, dtype=np.float32)])
            booster, names, model = m.hybrid, m.hybrid_names, "hybrid"
            p = float(m.hybrid_cal(booster.inplace_predict(x.reshape(1, -1))[0]))
            policy = m.hybrid_policy.for_segment(seg)
        else:
            x, booster, names, model, p = x_tab, m.tabular, m.tabular_names, "tabular", p_tab
            policy = m.tabular_policy.for_segment(seg)
        action = policy.decide(p)
        reasons = (
            reason_codes(booster, x, names)
            if self.explain and action != FraudAction.APPROVE
            else []
        )
        fz.observe(t, feats)
        return FraudDecision(
            t.txn_id,
            p,
            action,
            model,
            p_tab,
            sem is not None,
            reasons,
            (time.perf_counter() - start) * 1000,
            seg,
            None,
            skipped,
        )
