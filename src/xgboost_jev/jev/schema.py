"""Request/response types for Jev parallel question blocks.

A Jev query pairs one untrusted ``state`` (the text) with several questions that
Jev answers in a single round trip. ``noul`` questions return a probability in
[0, 1]; each answer becomes one numeric feature column for XGBoost.

The response format is not taken from published Jev docs, so
:func:`parse_probabilities` accepts the shapes a probability API commonly uses.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

NOUL = "noul"


@dataclass(frozen=True)
class Question:
    id: str
    body: str
    type: str = NOUL

    def to_dict(self) -> dict:
        return {"id": self.id, "type": self.type, "body": self.body}


@dataclass(frozen=True)
class JevQuery:
    state: str
    questions: tuple[Question, ...]

    def __post_init__(self) -> None:
        ids = [q.id for q in self.questions]
        if not ids:
            raise ValueError("a Jev query needs at least one question")
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate question ids: {ids}")

    @property
    def question_ids(self) -> tuple[str, ...]:
        return tuple(q.id for q in self.questions)

    def to_payload(self) -> dict:
        return {"state": self.state, "questions": [q.to_dict() for q in self.questions]}

    def cache_key(self) -> str:
        raw = json.dumps(self.to_payload(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()


@dataclass(frozen=True)
class JevResult:
    """Probabilities keyed by question id. Missing answers are NaN, never 0."""

    probabilities: Mapping[str, float]
    latency_ms: float = 0.0
    source: str = "jev"  # jev | cache | mock | fallback | unavailable
    error: str | None = field(default=None, compare=False)

    def get(self, qid: str) -> float:
        return float(self.probabilities.get(qid, math.nan))

    @property
    def ok(self) -> bool:
        return self.error is None

    @classmethod
    def unavailable(cls, question_ids: Iterable[str], error: str, latency_ms: float = 0.0):
        return cls({q: math.nan for q in question_ids}, latency_ms, "unavailable", error)


def _clamp(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return math.nan
    if math.isnan(v):
        return math.nan
    return min(1.0, max(0.0, v))


def parse_probabilities(payload: Mapping, question_ids: Iterable[str]) -> dict[str, float]:
    """Extract ``{question_id: probability}`` from a Jev response body.

    Accepted shapes::

        {"answers": {"p_override": 0.93, ...}}
        {"answers": [{"id": "p_override", "value": 0.93}, ...]}   # or probability / p
        {"p_override": 0.93, ...}
    """
    ids = list(question_ids)
    answers = payload.get("answers", payload.get("results", payload))
    found: dict[str, float] = {}
    if isinstance(answers, Mapping):
        for qid in ids:
            if qid in answers:
                raw = answers[qid]
                if isinstance(raw, Mapping):
                    raw = raw.get("value", raw.get("probability", raw.get("p")))
                found[qid] = _clamp(raw)
    elif isinstance(answers, list):
        for item in answers:
            if isinstance(item, Mapping) and item.get("id") in ids:
                raw = item.get("value", item.get("probability", item.get("p")))
                found[item["id"]] = _clamp(raw)
    return {qid: found.get(qid, math.nan) for qid in ids}
