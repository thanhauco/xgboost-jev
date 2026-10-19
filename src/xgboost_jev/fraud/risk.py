"""Time-decayed, Bayesian-smoothed fraud rates per merchant / MCC / payee.

    rate(key) = (fraud_decayed + a * prior) / (count_decayed + a)

Counts grow when a transaction is observed; fraud counts grow only when its
label ARRIVES. Reads therefore only use labels known at that moment - the
point-in-time property that prevents target leakage, the classic way merchant
risk features inflate offline metrics.
"""

from __future__ import annotations

import math

from .decay import DecayedCounterStore
from .kv import KVBackend

_HL = {"30d": 30 * 86400.0}


class RiskTables:
    def __init__(
        self, kv: KVBackend | None = None, prior_strength: float = 20.0, prior_rate: float = 0.005
    ):
        self.counts = DecayedCounterStore(kv, _HL)
        self.frauds = DecayedCounterStore(kv, _HL)
        self.a = prior_strength
        self.prior_rate = prior_rate

    def _prior(self, now: float) -> float:
        n = self.counts.read("risk:__all__", now)["cnt_30d"]
        f = self.frauds.read("riskf:__all__", now)["cnt_30d"]
        return (f + self.a * self.prior_rate) / (n + self.a)

    def rate(self, key: str, now: float, prior: float | None = None) -> tuple[float, float]:
        """(smoothed fraud rate, log1p(decayed volume)) for ``key``.

        Pass ``prior`` (from :meth:`prior`) when looking up several keys at once.
        """
        n = self.counts.read(f"risk:{key}", now)["cnt_30d"]
        f = self.frauds.read(f"riskf:{key}", now)["cnt_30d"]
        p = self._prior(now) if prior is None else prior
        return (f + self.a * p) / (n + self.a), math.log1p(n)

    def prior(self, now: float) -> float:
        return self._prior(now)

    def observe(self, keys: list[str], now: float) -> None:
        self.counts.add("risk:__all__", now)
        for k in keys:
            self.counts.add(f"risk:{k}", now)

    def label_fraud(self, keys: list[str], now: float) -> None:
        self.frauds.add("riskf:__all__", now)
        for k in keys:
            self.frauds.add(f"riskf:{k}", now)
