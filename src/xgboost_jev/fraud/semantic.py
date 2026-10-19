"""Jev semantic signals over the free-text fields of a payment."""

from __future__ import annotations

import math
from collections.abc import Sequence

from ..jev.client import JevClient
from ..jev.schema import JevQuery, Question
from .schema import Transaction

FRAUD_QUESTIONS: tuple[Question, ...] = (
    Question(
        "f_scam_urgency",
        "Does this payment text show urgency, pressure or coercion typical of a scam?",
    ),
    Question(
        "f_impersonation",
        "Does the text claim to come from a bank, government agency or tech support?",
    ),
    Question(
        "f_investment_scam",
        "Does the text promise guaranteed returns, crypto profits or prize winnings?",
    ),
    Question(
        "f_payment_redirect", "Does it ask for payment via gift cards, crypto or a 'safe account'?"
    ),
    Question(
        "f_descriptor_spoof", "Does the merchant name imitate a well-known brand with misspellings?"
    ),
)
SEMANTIC_NAMES = [f"jev_{q.id}" for q in FRAUD_QUESTIONS]

# Advanced stack: also reads in-session text (support chat, payment-reason answers).
COACHING_QUESTION = Question(
    "f_coached",
    "Does the customer say someone (a 'bank', police, an agent) instructed them to move money?",
)
ADVANCED_FRAUD_QUESTIONS: tuple[Question, ...] = FRAUD_QUESTIONS + (COACHING_QUESTION,)
ADVANCED_SEMANTIC_NAMES = [f"jev_{q.id}" for q in ADVANCED_FRAUD_QUESTIONS]


def build_state(t: Transaction) -> str | None:
    parts = []
    if t.merchant_descriptor:
        parts.append(f"Merchant: {t.merchant_descriptor}")
    if t.memo:
        parts.append(f"Memo: {t.memo}")
    if getattr(t, "session_text", ""):
        parts.append(f"Customer said: {t.session_text}")
    return "\n".join(parts) if parts else None


class FraudSemanticExtractor:
    def __init__(self, jev: JevClient, questions: Sequence[Question] = FRAUD_QUESTIONS) -> None:
        self.jev = jev
        self.questions = tuple(questions)

    def extract_many(self, txns: Sequence[Transaction]) -> list[list[float]]:
        """One row per transaction; NaN where there is no text or Jev failed."""
        states = [build_state(t) for t in txns]
        idx = [i for i, s in enumerate(states) if s]
        out = [[math.nan] * len(self.questions) for _ in txns]
        if not idx:
            return out
        try:
            results = self.jev.ask_many([JevQuery(states[i], self.questions) for i in idx])
        except Exception:  # noqa: BLE001 - degrade to NaN, XGBoost handles missing
            return out
        for i, r in zip(idx, results, strict=True):
            out[i] = [r.get(q.id) for q in self.questions]
        return out

    def extract(self, t: Transaction) -> list[float]:
        return self.extract_many([t])[0]
