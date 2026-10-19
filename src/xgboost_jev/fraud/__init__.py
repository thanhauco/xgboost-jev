"""Real-time fraud detection: streaming features + Jev semantic signals + XGBoost."""

from .explain import reason_codes
from .features import FEATURE_NAMES, TransactionFeaturizer
from .mock_data import LabeledStream, generate_transactions
from .schema import Transaction
from .scorer import (
    DecisionPolicy,
    EscalationPolicy,
    FraudAction,
    FraudDecision,
    FraudModels,
    RealTimeFraudScorer,
    choose_decline_threshold,
    train_fraud_models,
)
from .semantic import FRAUD_QUESTIONS, FraudSemanticExtractor
from .store import OnlineFeatureStore

__all__ = [
    "FEATURE_NAMES",
    "FRAUD_QUESTIONS",
    "DecisionPolicy",
    "EscalationPolicy",
    "FraudAction",
    "FraudDecision",
    "FraudModels",
    "FraudSemanticExtractor",
    "LabeledStream",
    "OnlineFeatureStore",
    "RealTimeFraudScorer",
    "Transaction",
    "TransactionFeaturizer",
    "choose_decline_threshold",
    "generate_transactions",
    "reason_codes",
    "train_fraud_models",
]
