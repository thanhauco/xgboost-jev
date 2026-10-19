"""Analyst assistant: case files for the review queue, Jev triage, fast labels.

Jev answers yes/no questions; it does not write prose. The narrative here is
assembled deterministically from the decision, reason codes and the account
timeline, and Jev is asked triage questions ABOUT that narrative. Analyst
decisions come back as ``LabelEvent(source="analyst")`` - proxy labels that
arrive in hours instead of the weeks a chargeback takes, and that feed the
graph, the risk tables and the next retrain.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field

from ..jev.client import JevClient
from ..jev.schema import JevQuery, Question
from .pipeline import AdvancedFeaturizer
from .schema import LabelEvent, Transaction
from .scorer import FraudDecision
from .sequence import RISKY

TRIAGE_QUESTIONS: tuple[Question, ...] = (
    Question(
        "t_app_scam",
        "Does this case look like a customer being tricked into sending money to a scammer?",
    ),
    Question(
        "t_ato", "Does this case look like a criminal who has taken over the customer's account?"
    ),
    Question(
        "t_mule", "Does the receiving account look like a money mule collecting from many victims?"
    ),
)


@dataclass
class CaseFile:
    txn_id: str
    score: float
    action: str
    narrative: str
    timeline: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    triage: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _ago(seconds: float) -> str:
    seconds = max(seconds, 0)
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


class AnalystAssistant:
    def __init__(
        self, jev: JevClient | None = None, questions: tuple[Question, ...] = TRIAGE_QUESTIONS
    ) -> None:
        self.jev = jev
        self.questions = questions

    def build_case(
        self, t: Transaction, d: FraudDecision, featurizer: AdvancedFeaturizer, max_events: int = 8
    ) -> CaseFile:
        events = [
            (ts, k) for ts, k in featurizer.seq._events.get(t.user_id, ()) if ts <= t.timestamp
        ]
        timeline = [
            f"{_ago(t.timestamp - ts)}: {k.replace('_', ' ')}" for ts, k in events[-max_events:]
        ]
        reasons = [r["reason"] for r in d.reasons]
        target = (
            f"payee {t.payee_id} ('{t.payee_account_name or t.payee_name}')"
            if t.payee_id
            else t.merchant_descriptor
        )
        parts = [
            f"{d.action.value.upper()}: {t.channel} payment of ${t.amount:,.2f} to {target}.",
            f"Risk score {d.score:.2f} from the {d.model} model.",
        ]
        if reasons:
            parts.append("Main reasons: " + "; ".join(reasons) + ".")
        risky = [
            f"{k.replace('_', ' ')} {_ago(t.timestamp - ts)}"
            for ts, k in events
            if k in RISKY and t.timestamp - ts <= 48 * 3600
        ]
        if risky:
            parts.append("Risky account changes in the last 48h: " + "; ".join(risky) + ".")
        if timeline:
            parts.append("Recent account activity: " + "; ".join(timeline[-4:]) + ".")
        if t.payee_id and t.payee_name and t.payee_account_name:
            parts.append(
                f"Payer typed payee name '{t.payee_name}', account holder is '{t.payee_account_name}'."
            )
        if t.memo:
            parts.append(f"Memo: '{t.memo}'.")
        if t.session_text:
            parts.append(f"Customer said: '{t.session_text}'.")
        case = CaseFile(t.txn_id, d.score, d.action.value, " ".join(parts), timeline, reasons)
        if self.jev is not None:
            try:
                r = self.jev.ask(JevQuery(case.narrative, self.questions))
                case.triage = {q.id: r.get(q.id) for q in self.questions}
            except Exception:  # noqa: BLE001 - triage is advisory
                case.triage = {q.id: math.nan for q in self.questions}
        return case

    @staticmethod
    def record_decision(
        txn_id: str, is_fraud: bool, fraud_type: str = "", ts: float | None = None
    ) -> LabelEvent:
        """Turn an analyst verdict into a fast proxy label (feed it to the scorer and LabelStore)."""
        return LabelEvent(
            time.time() if ts is None else ts, txn_id, int(is_fraud), "analyst", fraud_type
        )
