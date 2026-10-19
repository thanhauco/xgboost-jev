import pytest

from xgboost_jev.fraud import generate_transactions, train_fraud_models
from xgboost_jev.guardrail import generate_injection_dataset, train_guardrail
from xgboost_jev.jev import MockJevClient


@pytest.fixture(scope="session")
def jev():
    return MockJevClient()


@pytest.fixture(scope="session")
def injection_data():
    return (
        generate_injection_dataset(1200, split="train", seed=1),
        generate_injection_dataset(500, split="test", seed=2),
    )


@pytest.fixture(scope="session")
def static_model(injection_data):
    tr, _ = injection_data
    return train_guardrail(tr.text, tr.label, use_jev=False, num_boost_round=150)


@pytest.fixture(scope="session")
def cascade_model(injection_data, jev):
    tr, _ = injection_data
    return train_guardrail(tr.text, tr.label, jev=jev, use_jev=True, num_boost_round=150)


@pytest.fixture(scope="session")
def fraud_stream():
    return generate_transactions(n_users=150, days=10, seed=3)


@pytest.fixture(scope="session")
def fraud_trained(fraud_stream, jev):
    return train_fraud_models(fraud_stream, jev, rounds=200)
