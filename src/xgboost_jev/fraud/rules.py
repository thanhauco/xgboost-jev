"""Hot-swappable rules evaluated before the model.

During an attack you need to act in minutes, not after a retrain: block a
merchant, force review on a BIN, allowlist a known-good payee. Rules are plain
data (no ``eval``), so they can be added over the API and audited.

    {"id": "block-dk", "action": "decline", "priority": 10, "reason": "keys reseller under attack",
     "conditions": [{"field": "merchant_id", "op": "startswith", "value": "dk_"},
                    {"field": "amount", "op": "gt", "value": 5}]}

``field`` is looked up on the transaction first, then in the computed features.
"""

from __future__ import annotations

import operator
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .scorer import FraudAction

_OPS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "ge": operator.ge,
    "lt": operator.lt,
    "le": operator.le,
    "in": lambda a, b: a in b,
    "not_in": lambda a, b: a not in b,
    "contains": lambda a, b: a is not None and str(b).lower() in str(a).lower(),
    "startswith": lambda a, b: a is not None and str(a).startswith(str(b)),
}


@dataclass(frozen=True)
class Condition:
    field: str
    op: str
    value: Any

    def __post_init__(self) -> None:
        if self.op not in _OPS:
            raise ValueError(f"unknown op {self.op!r}; use one of {sorted(_OPS)}")


@dataclass
class Rule:
    id: str
    action: FraudAction
    conditions: list[Condition]
    priority: int = 100  # lower runs first
    reason: str = ""
    expires_at: float | None = None
    hits: int = field(default=0, compare=False)

    @classmethod
    def from_dict(cls, d: dict) -> Rule:
        return cls(
            id=d["id"],
            action=FraudAction(d["action"]),
            priority=int(d.get("priority", 100)),
            reason=d.get("reason", ""),
            expires_at=d.get("expires_at"),
            conditions=[Condition(c["field"], c["op"], c["value"]) for c in d["conditions"]],
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d


class RulesEngine:
    def __init__(self) -> None:
        self._rules: dict[str, Rule] = {}
        self._lock = threading.Lock()

    def add(self, rule: Rule | dict) -> Rule:
        r = Rule.from_dict(rule) if isinstance(rule, dict) else rule
        with self._lock:
            self._rules[r.id] = r
        return r

    def remove(self, rule_id: str) -> bool:
        with self._lock:
            return self._rules.pop(rule_id, None) is not None

    def list(self) -> list[Rule]:
        with self._lock:
            return sorted(self._rules.values(), key=lambda r: (r.priority, r.id))

    def evaluate(self, txn, features: dict | None = None, now: float | None = None) -> Rule | None:
        now = time.time() if now is None else now
        features = features or {}
        for r in self.list():
            if r.expires_at is not None and now >= r.expires_at:
                continue
            if all(_match(c, txn, features) for c in r.conditions):
                r.hits += 1
                return r
        return None


def _match(c: Condition, txn, features: dict) -> bool:
    value = getattr(txn, c.field) if hasattr(txn, c.field) else features.get(c.field)
    try:
        return bool(_OPS[c.op](value, c.value))
    except TypeError:
        return False
