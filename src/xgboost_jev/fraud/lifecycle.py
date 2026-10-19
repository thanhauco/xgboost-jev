"""Model lifecycle: continual retraining on delayed labels, shadow promotion,
champion/challenger routing and per-fraud-type recall monitoring.

Fraudsters adapt in days; chargebacks arrive in weeks. The defence is a loop:

    every N days:  labels known now -> warm-start challenger from champion
                   -> shadow-compare on the most recent (held-out) days
                   -> promote only if it is at least as good

Graph reputation (``EntityGraph.mark_fraud``) updates on every label without
any retrain; retraining teaches the model how much that reputation matters.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import average_precision_score

from .advanced import AdvancedModels, train_advanced
from .labels import LabelStore
from .schema import LabelEvent, Transaction

DAY = 86400.0


@dataclass
class RetrainRecord:
    as_of_day: float
    promoted: bool
    champion_pr_auc: float
    challenger_pr_auc: float
    eval_rows: int
    eval_fraud: int
    p_challenger_worse: float = math.nan


def prob_challenger_worse(
    y: np.ndarray, p_champ: np.ndarray, p_chal: np.ndarray, n_boot: int = 200, seed: int = 0
) -> float:
    """Bootstrap P(challenger PR-AUC < champion PR-AUC) on the shadow window.

    Shadow windows hold only the few frauds already reported, so a raw
    "challenger >= champion" test is mostly noise and vetoes good retrains.
    """
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    worse = 0
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        worse += average_precision_score(y[idx], p_chal[idx]) < average_precision_score(
            y[idx], p_champ[idx]
        )
    return worse / n_boot


@dataclass
class ContinualResult:
    scores: (
        np.ndarray
    )  # hybrid score of the model that was live when each row arrived (NaN before start)
    history: list[RetrainRecord] = field(default_factory=list)
    final: AdvancedModels | None = None


def continual_backtest(
    x_full: np.ndarray,
    sem: np.ndarray,
    txns: list[Transaction],
    labels: LabelStore,
    start_ts: float,
    first_train_ts: float,
    every_days: float = 7.0,
    eval_days: float = 5.0,
    block_confidence: float = 0.9,
    warm_start: bool = True,
    seed: int = 0,
    **train_kw,
) -> ContinualResult:
    """Replay deployment: train at ``first_train_ts``, then retrain every ``every_days``.

    Each period is scored by whichever model was live at the time, so the result
    is what production would have seen. Shadow evaluation uses labels known at
    retrain time on the last ``eval_days`` (excluded from challenger training);
    the challenger is promoted unless it is worse with ``block_confidence``
    (bootstrap). ``block_confidence=0`` reproduces a naive "must beat" gate,
    ``block_confidence>1`` always promotes.
    """
    ts = np.array([t.timestamp for t in txns])
    ids = [t.txn_id for t in txns]
    scores = np.full(len(txns), np.nan)
    champion = train_advanced(x_full, sem, txns, labels, first_train_ts, seed=seed, **train_kw)
    res = ContinualResult(scores)
    t_now = first_train_ts
    end = ts.max() + 1
    while t_now < end:
        t_next = t_now + every_days * DAY
        live = (ts >= t_now) & (ts < t_next)
        if live.any():
            scores[live] = champion.predict_hybrid(x_full[live], sem[live], calibrated=False)
        if t_next >= end:
            break
        challenger = train_advanced(
            x_full,
            sem,
            txns,
            labels,
            t_next,
            data_until=t_next - eval_days * DAY,
            seed=seed,
            init=champion if warm_start else None,
            **train_kw,
        )
        known = labels.known_as_of(t_next)
        ev = (ts >= t_next - eval_days * DAY) & (ts < t_next)
        y_ev = np.array([known.get(ids[i], 0) for i in np.flatnonzero(ev)])
        pc = pn = p_worse = math.nan
        if ev.any() and 0 < y_ev.sum() < len(y_ev):
            s_c = champion.predict_hybrid(x_full[ev], sem[ev], False)
            s_n = challenger.predict_hybrid(x_full[ev], sem[ev], False)
            pc, pn = average_precision_score(y_ev, s_c), average_precision_score(y_ev, s_n)
            p_worse = prob_challenger_worse(y_ev, s_c, s_n, seed=seed)
        if block_confidence <= 0:
            promote = not math.isnan(pn) and pn >= pc
        else:
            promote = math.isnan(p_worse) or p_worse < block_confidence
        res.history.append(
            RetrainRecord(
                (t_next - start_ts) / DAY, promote, pc, pn, int(ev.sum()), int(y_ev.sum()), p_worse
            )
        )
        if promote:
            champion = challenger
        t_now = t_next
    res.final = champion
    return res


class ChampionChallenger:
    """Score every transaction with both models; act on the challenger for a stable slice of traffic."""

    def __init__(
        self, champion: AdvancedModels, challenger: AdvancedModels, challenger_share: float = 0.05
    ):
        self.champion, self.challenger, self.share = champion, challenger, challenger_share
        self.log: list[tuple[str, float, float, bool]] = []

    def routed_to_challenger(self, txn_id: str) -> bool:
        h = int.from_bytes(hashlib.blake2b(txn_id.encode(), digest_size=4).digest(), "big")
        return h / 2**32 < self.share

    def score(self, txn_ids: list[str], x_full: np.ndarray, sem: np.ndarray) -> np.ndarray:
        pc = self.champion.predict_hybrid(x_full, sem, calibrated=True)
        pn = self.challenger.predict_hybrid(x_full, sem, calibrated=True)
        out = pc.copy()
        for i, tid in enumerate(txn_ids):
            routed = self.routed_to_challenger(tid)
            if routed:
                out[i] = pn[i]
            self.log.append((tid, float(pc[i]), float(pn[i]), routed))
        return out

    def report(self, labels: dict[str, int], review_capacity: float = 0.015) -> dict:
        tids = [r[0] for r in self.log]
        y = np.array([labels.get(t, 0) for t in tids])
        out = {}
        for name, col in (("champion", 1), ("challenger", 2)):
            p = np.array([r[col] for r in self.log])
            k = max(1, int(review_capacity * len(p)))
            top = np.argsort(-p)[:k]
            out[name] = {
                "pr_auc": float(average_precision_score(y, p))
                if 0 < y.sum() < len(y)
                else math.nan,
                "precision_at_capacity": float(y[top].mean()),
                "recall_at_capacity": float(y[top].sum() / max(y.sum(), 1)),
            }
        out["challenger_traffic"] = float(np.mean([r[3] for r in self.log]))
        return out


class RecallMonitor:
    """Rolling recall per fraud type, computed when labels arrive.

    A drop versus the type's baseline is usually the first sign attackers have
    adapted - or that a new pattern has appeared (``unseen`` types alert at once
    when recall is low).
    """

    def __init__(
        self, window: int = 50, min_labels: int = 10, drop_alert: float = 0.25, floor: float = 0.5
    ):
        self.window, self.min_labels, self.drop_alert, self.floor = (
            window,
            min_labels,
            drop_alert,
            floor,
        )
        self._flagged: dict[str, bool] = {}
        self._recent: dict[str, deque] = defaultdict(lambda: deque(maxlen=window))
        self._baseline: dict[str, float] = {}

    def record_decision(self, txn_id: str, flagged: bool) -> None:
        self._flagged[txn_id] = flagged

    def on_label(self, ev: LabelEvent) -> None:
        if ev.label != 1 or ev.txn_id not in self._flagged:
            return
        q = self._recent[ev.fraud_type or "unknown"]
        q.append(self._flagged[ev.txn_id])

    def freeze_baseline(self) -> None:
        for k, q in self._recent.items():
            if len(q) >= self.min_labels:
                self._baseline[k] = float(np.mean(q))

    def report(self) -> dict[str, dict]:
        out = {}
        for k, q in self._recent.items():
            if len(q) < self.min_labels:
                continue
            r = float(np.mean(q))
            base = self._baseline.get(k)
            alert = (base is not None and r < base - self.drop_alert) or (
                base is None and r < self.floor
            )
            out[k] = {"recall": round(r, 3), "baseline": base, "n": len(q), "alert": bool(alert)}
        return out
