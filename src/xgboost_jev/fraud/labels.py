"""Delayed-label store: what was KNOWN, and when.

Chargebacks arrive weeks after a card payment, victim reports in days, analyst
decisions in hours. Training must only use labels available at training time,
and very recent transactions are not yet "clean" just because nobody reported
them - they are excluded until they mature.
"""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Iterable

import numpy as np

from .schema import LabelEvent


class LabelStore:
    def __init__(self) -> None:
        self._events: list[LabelEvent] = []
        self._times: list[float] = []

    def add(self, ev: LabelEvent) -> None:
        i = bisect.bisect_right(self._times, ev.timestamp)
        self._times.insert(i, ev.timestamp)
        self._events.insert(i, ev)

    def extend(self, evs: Iterable[LabelEvent]) -> None:
        for e in evs:
            self.add(e)

    def fraud_types_as_of(self, t: float) -> dict[str, str]:
        """txn_id -> fraud type for confirmed fraud known at ``t``."""
        out: dict[str, str] = {}
        for ev in self._events[: bisect.bisect_right(self._times, t)]:
            if ev.label == 1:
                out[ev.txn_id] = ev.fraud_type or "unknown"
            else:
                out.pop(ev.txn_id, None)
        return out

    def known_as_of(self, t: float) -> dict[str, int]:
        """txn_id -> label for every label available at time ``t`` (latest wins)."""
        out: dict[str, int] = {}
        for ev in self._events[: bisect.bisect_right(self._times, t)]:
            out[ev.txn_id] = ev.label
        return out

    def training_labels(
        self, txn_ids: list[str], txn_ts: np.ndarray, as_of: float, maturity_days: float = 7.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """(labels, usable_mask) as a model trained at ``as_of`` would see them.

        Only transactions older than ``maturity_days`` are usable - fraud AND
        legit alike. Keeping recent known fraud while dropping recent unlabelled
        rows makes the newest data almost all positive, and the model then learns
        "recent = fraud" from any feature that drifts with time (account age,
        history length). Within the usable rows, fraud whose label has not
        arrived by ``as_of`` is (realistically) treated as legit.
        """
        known = self.known_as_of(as_of)
        y = np.array([known.get(t, 0) for t in txn_ids], dtype=np.int32)
        usable = txn_ts < as_of - maturity_days * 86400
        return y, usable


# Which entities to link, by fraud type (production: chargeback / report reason codes).
# Attacker infrastructure links by device/IP; victim-side fraud happens on the victim's
# OWN device, so it must link by the receiving side instead.
LINK_KEYS: dict[str, tuple[str, ...]] = {
    "card_testing": ("device_id", "ip"),
    "account_takeover": ("device_id", "ip"),
    "ato_warmup": ("device_id", "ip", "payee_id"),
    "low_and_slow": ("device_id", "ip"),
    "app_scam": ("payee_id",),
    "mule_ring": ("payee_id",),
    "descriptor_spoof": ("merchant_id",),
}
_HUB_KEYS = ("device_id", "ip")


def propagate_entity_labels(
    txns: list,
    y: np.ndarray,
    fraud_types: dict[str, str],
    window_days: float = 30.0,
    link_keys: dict[str, tuple[str, ...]] | None = None,
    max_users_per_entity: int = 12,
) -> tuple[np.ndarray, int]:
    """Weak positives from case linking: rows sharing an attacker-side entity with
    confirmed fraud (within ``window_days``) are labelled fraud too.

    Fixes label poisoning: without it, a fraud ring's older transactions - whose
    chargebacks have not arrived - teach the model that the ring is legit.
    ``fraud_types`` maps confirmed-fraud txn ids to their type; the type picks the
    linking keys (``LINK_KEYS``). Unknown types are not propagated. Devices/IPs
    used by many accounts (hubs) are skipped. Returns (labels, rows flipped).
    """
    link_keys = LINK_KEYS if link_keys is None else link_keys
    users: dict[tuple[str, str], set] = defaultdict(set)
    rows: dict[tuple[str, str], list[int]] = defaultdict(list)
    all_keys = {k for ks in link_keys.values() for k in ks}
    for i, t in enumerate(txns):
        for k in all_keys:
            v = getattr(t, k, None)
            if v:
                users[(k, v)].add(t.user_id)
                rows[(k, v)].append(i)
    out = y.copy()
    w = window_days * 86400
    for t in txns:
        ftype = fraud_types.get(t.txn_id)
        if ftype is None:
            continue
        for k in link_keys.get(ftype, ()):
            v = getattr(t, k, None)
            if not v or (k in _HUB_KEYS and len(users[(k, v)]) > max_users_per_entity):
                continue
            for j in rows[(k, v)]:
                if abs(txns[j].timestamp - t.timestamp) <= w:
                    out[j] = 1
    return out, int((out != y).sum())
