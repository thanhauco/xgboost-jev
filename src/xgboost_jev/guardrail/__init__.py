"""Prompt-injection guardrails combining Jev semantic probabilities with XGBoost."""

from .cascade import GuardrailFeatureBuilder, GuardrailModel, train_guardrail
from .decision import Action, GuardDecision
from .embeddings import HashingSVDEmbedder
from .ensemble import EnsembleConfig, EnsembleGuard, jev_risk
from .features import counterfeit_policy_features, text_metadata_features
from .mock_data import generate_injection_dataset
from .questions import INJECTION_QUESTIONS, PREFILTER_QUESTION
from .router import LatencyRouter, RouterConfig

__all__ = [
    "INJECTION_QUESTIONS",
    "PREFILTER_QUESTION",
    "Action",
    "EnsembleConfig",
    "EnsembleGuard",
    "GuardDecision",
    "GuardrailFeatureBuilder",
    "GuardrailModel",
    "HashingSVDEmbedder",
    "LatencyRouter",
    "RouterConfig",
    "counterfeit_policy_features",
    "generate_injection_dataset",
    "jev_risk",
    "text_metadata_features",
    "train_guardrail",
]
