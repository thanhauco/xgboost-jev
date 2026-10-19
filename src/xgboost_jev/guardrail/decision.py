from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum


class Action(str, Enum):
    ALLOW = "allow"
    REVIEW = "review"
    BLOCK = "block"


@dataclass
class GuardDecision:
    action: Action
    score: float
    reasons: list[str] = field(default_factory=list)
    components: dict[str, float] = field(default_factory=dict)
    stage: str = ""
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["action"] = self.action.value
        return d
