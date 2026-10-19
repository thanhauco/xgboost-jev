"""Account-event sequence features.

Account takeovers follow a script - new-device login, password reset, new
payee, then a transfer - where each step alone is common. Two complementary
signals:

* recency features: hours since the last risky event of each kind, and
* Markov surprise: mean -log P(next | prev) of the user's recent event chain
  under a transition model learned online from all users (point-in-time).

``SequenceEncoder`` is the hook for a learned encoder (GRU/transformer
embedding) whose output columns can be appended the same way.
"""

from __future__ import annotations

import math
import threading
from collections import defaultdict, deque
from typing import Protocol

from .schema import EVENT_KINDS

RISKY = ("login_new_device", "password_reset", "mfa_change", "new_payee", "profile_change")


class SequenceEncoder(Protocol):
    names: list[str]

    def encode(self, kinds: list[str], gaps_s: list[float]) -> list[float]: ...


class SequenceStore:
    def __init__(self, history: int = 20, surprise_steps: int = 5) -> None:
        self.history = history
        self.k = surprise_steps
        self._events: dict[str, deque] = defaultdict(lambda: deque(maxlen=history))
        self._last: dict[str, dict[str, float]] = defaultdict(dict)
        self._trans: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        self._lock = threading.Lock()

    def _p(self, a: str, b: str) -> float:
        row = self._trans.get(a)
        total = sum(row.values()) if row else 0.0
        n_kinds = len(EVENT_KINDS)
        return ((row.get(b, 0.0) if row else 0.0) + 0.5) / (total + 0.5 * n_kinds)

    def features(self, user: str, now: float, next_kind: str = "txn") -> dict[str, float]:
        with self._lock:
            ev = list(self._events.get(user, ()))
            last = self._last.get(user, {})
            out = {}
            for kind in RISKY:
                ts = last.get(kind)
                out[f"seq_h_since_{kind}"] = (now - ts) / 3600 if ts is not None else math.nan
            out["seq_risky_24h"] = float(sum(1 for ts, k in ev if k in RISKY and now - ts <= 86400))
            chain = [k for _, k in ev][-self.k :] + [next_kind]
            if len(chain) >= 2:
                nll = [-math.log(self._p(a, b)) for a, b in zip(chain, chain[1:], strict=False)]
                out["seq_markov_surprise"] = sum(nll) / len(nll)
            else:
                out["seq_markov_surprise"] = math.nan
            r, p = last.get("password_reset"), last.get("new_payee")
            out["seq_reset_then_payee_48h"] = float(
                r is not None and p is not None and p >= r and now - r <= 48 * 3600
            )
            return out

    def observe(self, user: str, ts: float, kind: str) -> None:
        with self._lock:
            q = self._events[user]
            if q:
                self._trans[q[-1][1]][kind] += 1.0
            q.append((ts, kind))
            self._last[user][kind] = ts
