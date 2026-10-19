"""Advanced streaming featurizer and the time-ordered event replay that feeds it.

One object owns every piece of online state and consumes a single stream of
three event types, in timestamp order:

* ``LabelEvent``    -> fraud flags in the entity graph, merchant risk tables
* ``AccountEvent``  -> sequence store, graph edges (logins)
* ``Transaction``   -> features are READ, then the transaction is OBSERVED

Training replay and online scoring both drive this exact object, so features
are point-in-time correct by construction.
"""

from __future__ import annotations

import heapq
import math
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .decay import DecayedCounterStore, DecayedMeanVar
from .features import FEATURE_NAMES, TransactionFeaturizer
from .graph import EntityGraph
from .kv import KVBackend
from .risk import RiskTables
from .schema import AccountEvent, LabelEvent, Transaction
from .sequence import RISKY, SequenceStore
from .store import OnlineFeatureStore
from .textsig import descriptor_signals, name_similarity

DECAY_NAMES = [
    "dc_card_cnt_1h",
    "dc_card_amt_1d",
    "dc_card_amt_30d",
    "dc_card_burst",
    "dc_user_amt_7d",
    "dc_device_cnt_1d",
    "dc_payee_in_cnt_1d",
    "dc_payee_in_amt_1d",
]
GRAPH_NAMES = [
    "g_device_users",
    "g_device_fraud",
    "g_ip_fraud",
    "g_card_fraud",
    "g_comp_size",
    "g_comp_fraud",
    "g_2hop_fraud",
    "payee_fan_in_7d",
    "payee_age_days",
    "payee_new_for_user",
    "g_payee_fraud",
    "g_payee_comp_size",
    "g_payee_comp_fraud",
]
RISK_NAMES = ["r_merchant", "r_merchant_vol", "r_mcc", "peer_amount_z"]
SEQ_NAMES = [f"seq_h_since_{k}" for k in RISKY] + [
    "seq_risky_24h",
    "seq_markov_surprise",
    "seq_reset_then_payee_48h",
]
TEXT_NAMES = ["cop_similarity", "desc_brand_lookalike", "desc_suspicious_tokens", "desc_leet"]
DEVICE_NAMES = ["dev_emulator", "dev_rooted"]
ADV_NAMES = DECAY_NAMES + GRAPH_NAMES + RISK_NAMES + SEQ_NAMES + TEXT_NAMES + DEVICE_NAMES
ADVANCED_FEATURE_NAMES = FEATURE_NAMES + ADV_NAMES


def _flag(v: bool | None) -> float:
    return math.nan if v is None else float(v)


class AdvancedFeaturizer:
    feature_names = ADVANCED_FEATURE_NAMES

    def __init__(self, kv: KVBackend | None = None, txn_index_cap: int = 2_000_000) -> None:
        self.base = TransactionFeaturizer(OnlineFeatureStore())
        self.decay = DecayedCounterStore(kv)
        self.peer = DecayedMeanVar(kv)
        self.graph = EntityGraph()
        self.risk = RiskTables(kv)
        self.seq = SequenceStore()
        self._index: OrderedDict[str, tuple] = OrderedDict()
        self._cap = txn_index_cap

    # -- events ----------------------------------------------------------------
    def on_account_event(self, ev: AccountEvent) -> None:
        self.seq.observe(ev.user_id, ev.timestamp, ev.kind)
        if ev.kind.startswith("login"):
            self.graph.observe_login(ev.user_id, ev.device_id, ev.ip)

    def on_label(self, ev: LabelEvent) -> None:
        if ev.label != 1:
            return
        ent = self._index.get(ev.txn_id)
        if ent is None:
            return
        card, device, ip, payee, merchant, mcc = ent
        self.graph.mark_fraud(f"card:{card}")
        if device:
            self.graph.mark_fraud(f"device:{device}")
        if ip:
            self.graph.mark_fraud(f"ip:{ip}")
        if payee:
            self.graph.mark_fraud(f"payee:{payee}")
        self.risk.label_fraud([f"m:{merchant}", f"mcc:{mcc}"], ev.timestamp)

    # -- transactions ----------------------------------------------------------
    def compute(self, t: Transaction) -> dict[str, float]:
        now = t.timestamp
        f = self.base.compute(t)
        card = self.decay.read(f"card:{t.card_id}", now)
        f["dc_card_cnt_1h"] = card["cnt_1h"]
        f["dc_card_amt_1d"] = card["sum_1d"]
        f["dc_card_amt_30d"] = card["sum_30d"]
        f["dc_card_burst"] = card["sum_1d"] / (card["sum_30d"] / 30.0 + 1.0)
        f["dc_user_amt_7d"] = self.decay.read(f"user:{t.user_id}", now)["sum_7d"]
        f["dc_device_cnt_1d"] = (
            self.decay.read(f"device:{t.device_id}", now)["cnt_1d"] if t.device_id else math.nan
        )

        g = self.graph
        dev = f"device:{t.device_id}" if t.device_id else None
        f["g_device_users"] = float(g.degree(dev, "user:")) if dev else math.nan
        f["g_device_fraud"] = float(g.is_fraud(dev)) if dev else math.nan
        f["g_ip_fraud"] = float(g.is_fraud(f"ip:{t.ip}")) if t.ip else math.nan
        f["g_card_fraud"] = float(g.is_fraud(f"card:{t.card_id}"))
        size, fraud = g.components([f"user:{t.user_id}"] + ([dev] if dev else []))
        f["g_comp_size"], f["g_comp_fraud"] = float(size), float(fraud)
        f["g_2hop_fraud"] = float(g.two_hop_fraud(dev)) if dev else math.nan
        if t.payee_id:
            f.update(g.payee_stats(t.payee_id, t.user_id, now))
            p = f"payee:{t.payee_id}"
            f["g_payee_fraud"] = float(g.is_fraud(p))
            size, fr = g.component(p)
            f["g_payee_comp_size"], f["g_payee_comp_fraud"] = float(size), float(fr)
            pin = self.decay.read(f"payee:{t.payee_id}", now)
            f["dc_payee_in_cnt_1d"], f["dc_payee_in_amt_1d"] = pin["cnt_1d"], pin["sum_1d"]
        else:
            for k in (
                "payee_fan_in_7d",
                "payee_age_days",
                "payee_new_for_user",
                "g_payee_fraud",
                "g_payee_comp_size",
                "g_payee_comp_fraud",
                "dc_payee_in_cnt_1d",
                "dc_payee_in_amt_1d",
            ):
                f[k] = math.nan

        prior = self.risk.prior(now)
        f["r_merchant"], f["r_merchant_vol"] = self.risk.rate(f"m:{t.merchant_id}", now, prior)
        f["r_mcc"], _ = self.risk.rate(f"mcc:{t.mcc}", now, prior)
        w, mean, var = self.peer.read(self._segment(t, f), now)
        f["peer_amount_z"] = (
            (f["log_amount"] - mean) / math.sqrt(var) if w > 20 and var > 1e-6 else math.nan
        )

        f.update(self.seq.features(t.user_id, now))
        f["cop_similarity"] = (
            name_similarity(t.payee_name, t.payee_account_name) if t.payee_id else math.nan
        )
        f.update(descriptor_signals(t.merchant_descriptor if t.channel != "p2p" else ""))
        f["dev_emulator"], f["dev_rooted"] = _flag(t.is_emulator), _flag(t.is_rooted)
        return f

    @staticmethod
    def _segment(t: Transaction, f: dict) -> str:
        age = f.get("account_age_days", math.nan)
        tenure = "new" if (math.isnan(age) or age < 30) else "est"
        return f"peer:{t.channel}|{tenure}"

    def vector(self, f: dict[str, float]) -> np.ndarray:
        return np.array([f[n] for n in self.feature_names], dtype=np.float32)

    def observe(self, t: Transaction, feats: dict[str, float] | None = None) -> None:
        now = t.timestamp
        self.base.observe(t)
        self.decay.add(f"card:{t.card_id}", now, t.amount)
        self.decay.add(f"user:{t.user_id}", now, t.amount)
        if t.device_id:
            self.decay.add(f"device:{t.device_id}", now, t.amount)
        if t.payee_id:
            self.decay.add(f"payee:{t.payee_id}", now, t.amount)
        self.graph.observe_txn(now, t.user_id, t.card_id, t.device_id, t.ip, t.payee_id)
        self.risk.observe([f"m:{t.merchant_id}", f"mcc:{t.mcc}"], now)
        seg_f = feats if feats is not None else {"account_age_days": math.nan}
        self.peer.add(self._segment(t, seg_f), now, math.log1p(t.amount))
        self.seq.observe(t.user_id, now, "txn")
        self._index[t.txn_id] = (t.card_id, t.device_id, t.ip, t.payee_id, t.merchant_id, t.mcc)
        if len(self._index) > self._cap:
            self._index.popitem(last=False)

    def process(self, t: Transaction) -> dict[str, float]:
        f = self.compute(t)
        self.observe(t, f)
        return f


def merged_events(
    transactions: Iterable[Transaction],
    account_events: Iterable[AccountEvent] = (),
    label_events: Iterable[LabelEvent] = (),
):
    """Yield events in time order; at equal timestamps labels, then account events, then payments."""
    streams = [
        (
            (e.timestamp, 0, i, e)
            for i, e in enumerate(sorted(label_events, key=lambda e: e.timestamp))
        ),
        (
            (e.timestamp, 1, i, e)
            for i, e in enumerate(sorted(account_events, key=lambda e: e.timestamp))
        ),
        (
            (e.timestamp, 2, i, e)
            for i, e in enumerate(sorted(transactions, key=lambda e: e.timestamp))
        ),
    ]
    for _, _, _, e in heapq.merge(*streams):
        yield e


@dataclass
class AdvancedReplay:
    x: np.ndarray
    txns: list[Transaction]
    featurizer: AdvancedFeaturizer


def replay_advanced(
    transactions,
    account_events=(),
    label_events=(),
    featurizer: AdvancedFeaturizer | None = None,
    stop_at: float | None = None,
) -> AdvancedReplay:
    fz = featurizer or AdvancedFeaturizer()
    rows, txns = [], []
    for e in merged_events(transactions, account_events, label_events):
        if stop_at is not None and e.timestamp >= stop_at:
            break
        if isinstance(e, Transaction):
            rows.append(fz.vector(fz.process(e)))
            txns.append(e)
        elif isinstance(e, AccountEvent):
            fz.on_account_event(e)
        else:
            fz.on_label(e)
    x = np.vstack(rows) if rows else np.empty((0, len(fz.feature_names)), np.float32)
    return AdvancedReplay(x, txns, fz)
