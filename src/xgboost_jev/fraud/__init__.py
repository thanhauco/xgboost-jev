"""Real-time fraud detection: streaming features + Jev semantic signals + XGBoost.

v1 (``RealTimeFraudScorer``): per-transaction velocity/behaviour/geo features.
Advanced (``AdvancedFraudScorer``): adds decayed counters, entity graph, risk
tables, sequence features, local text signals, delayed-label training,
calibration, segment policies, rules, latency budgets and model lifecycle.
"""

from .advanced import AdvancedFraudScorer, AdvancedModels, train_advanced
from .analyst import AnalystAssistant, CaseFile
from .calibration import IsotonicCalibrator, SegmentedPolicy, expected_calibration_error
from .explain import reason_codes
from .features import FEATURE_NAMES, TransactionFeaturizer
from .graph import EntityGraph
from .kv import InMemoryKV, RedisKV, partition_for
from .labels import LabelStore, propagate_entity_labels
from .lifecycle import ChampionChallenger, RecallMonitor, continual_backtest
from .mock_advanced import AdvancedStream, generate_advanced_stream
from .mock_data import LabeledStream, generate_transactions
from .pipeline import ADVANCED_FEATURE_NAMES, AdvancedFeaturizer, replay_advanced
from .rules import Rule, RulesEngine
from .scaling import PartitionedScorer
from .schema import AccountEvent, LabelEvent, Transaction
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
from .semantic import ADVANCED_FRAUD_QUESTIONS, FRAUD_QUESTIONS, FraudSemanticExtractor
from .store import OnlineFeatureStore

__all__ = [
    "ADVANCED_FEATURE_NAMES",
    "ADVANCED_FRAUD_QUESTIONS",
    "FEATURE_NAMES",
    "FRAUD_QUESTIONS",
    "AccountEvent",
    "AdvancedFeaturizer",
    "AdvancedFraudScorer",
    "AdvancedModels",
    "AdvancedStream",
    "AnalystAssistant",
    "CaseFile",
    "ChampionChallenger",
    "DecisionPolicy",
    "EntityGraph",
    "EscalationPolicy",
    "FraudAction",
    "FraudDecision",
    "FraudModels",
    "FraudSemanticExtractor",
    "InMemoryKV",
    "IsotonicCalibrator",
    "LabelEvent",
    "LabelStore",
    "LabeledStream",
    "OnlineFeatureStore",
    "PartitionedScorer",
    "RealTimeFraudScorer",
    "RecallMonitor",
    "RedisKV",
    "Rule",
    "RulesEngine",
    "SegmentedPolicy",
    "Transaction",
    "TransactionFeaturizer",
    "choose_decline_threshold",
    "continual_backtest",
    "expected_calibration_error",
    "generate_advanced_stream",
    "generate_transactions",
    "partition_for",
    "propagate_entity_labels",
    "reason_codes",
    "replay_advanced",
    "train_advanced",
    "train_fraud_models",
]
