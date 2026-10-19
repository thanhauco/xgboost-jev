"""In-process online feature store with sliding windows and entity profiles.

Point-in-time correctness: features for a transaction are always read BEFORE
the transaction is written, both during offline replay (training) and online
scoring, so training features match what the scorer sees in production.
Swap in a Redis/Aerospike implementation of the same methods to scale out.
"""

from __future__ import annotations

import math
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, field

from .schema import Transaction

WINDOWS_S = {"1m": 60, "10m": 600, "1h": 3600, "24h": 86400}
_HORIZON = max(WINDOWS_S.values())


@dataclass
class _Event:
    ts: float
    amount: float
    merchant_id: str
    card_id: str


@dataclass
class EntityProfile:
    first_seen: float
    n: int = 0
    mean_log_amt: float = 0.0
    m2_log_amt: float = 0.0
    last_ts: float | None = None
    last_lat: float | None = None
    last_lon: float | None = None
    devices: OrderedDict = field(default_factory=OrderedDict)
    ips: OrderedDict = field(default_factory=OrderedDict)
    merchants: OrderedDict = field(default_factory=OrderedDict)

    @property
    def std_log_amt(self) -> float:
        return math.sqrt(self.m2_log_amt / (self.n - 1)) if self.n > 1 else math.nan


def _remember(od: OrderedDict, key: str | None, cap: int = 64) -> None:
    if key is None:
        return
    od[key] = True
    od.move_to_end(key)
    while len(od) > cap:
        od.popitem(last=False)


class OnlineFeatureStore:
    def __init__(self, max_events: int = 2000) -> None:
        self.max_events = max_events
        self._events: dict[str, deque[_Event]] = {}
        self._profiles: dict[str, EntityProfile] = {}
        self._lock = threading.RLock()

    def _evict(self, q: deque[_Event], now: float) -> None:
        while q and now - q[0].ts > _HORIZON:
            q.popleft()

    def window(self, key: str, now: float, seconds: float) -> list[_Event]:
        with self._lock:
            q = self._events.get(key)
            if not q:
                return []
            self._evict(q, now)
            return [e for e in q if now - e.ts <= seconds]

    def profile(self, key: str) -> EntityProfile | None:
        with self._lock:
            return self._profiles.get(key)

    def observe(self, txn: Transaction) -> None:
        ev = _Event(txn.timestamp, txn.amount, txn.merchant_id, txn.card_id)
        keys = [f"card:{txn.card_id}", f"user:{txn.user_id}"]
        if txn.device_id:
            keys.append(f"device:{txn.device_id}")
        if txn.ip:
            keys.append(f"ip:{txn.ip}")
        with self._lock:
            for k in keys:
                q = self._events.setdefault(k, deque(maxlen=self.max_events))
                q.append(ev)
                self._evict(q, txn.timestamp)
            prof = self._profiles.setdefault(f"user:{txn.user_id}", EntityProfile(txn.timestamp))
            x = math.log1p(txn.amount)
            prof.n += 1
            delta = x - prof.mean_log_amt
            prof.mean_log_amt += delta / prof.n
            prof.m2_log_amt += delta * (x - prof.mean_log_amt)
            prof.last_ts = txn.timestamp
            if txn.lat is not None and txn.lon is not None:
                prof.last_lat, prof.last_lon = txn.lat, txn.lon
            _remember(prof.devices, txn.device_id)
            _remember(prof.ips, txn.ip)
            _remember(prof.merchants, txn.merchant_id, cap=256)
