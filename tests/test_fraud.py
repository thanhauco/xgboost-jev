import math

import numpy as np

from xgboost_jev.fraud import (
    FEATURE_NAMES,
    DecisionPolicy,
    EscalationPolicy,
    FraudAction,
    FraudModels,
    RealTimeFraudScorer,
    Transaction,
    TransactionFeaturizer,
    choose_decline_threshold,
)
from xgboost_jev.fraud.features import haversine_km
from xgboost_jev.fraud.semantic import FraudSemanticExtractor, build_state
from xgboost_jev.jev import MockJevClient

T0 = 1_788_000_000.0


def txn(i, ts, **kw):
    base = dict(
        txn_id=f"t{i}",
        timestamp=ts,
        user_id="u1",
        card_id="c1",
        amount=20.0,
        merchant_id="m1",
        device_id="d1",
        ip="10.0.0.1",
        lat=40.71,
        lon=-74.0,
    )
    base.update(kw)
    return Transaction(**base)


def test_haversine():
    assert math.isclose(haversine_km(40.71, -74.01, 51.51, -0.13), 5570, rel_tol=0.01)


def test_features_are_point_in_time():
    fz = TransactionFeaturizer()
    first = fz.compute(txn(1, T0))
    assert first["card_cnt_1h"] == 0 and first["user_history_len"] == 0
    fz.observe(txn(1, T0))
    f2 = fz.compute(txn(2, T0 + 30))
    assert f2["card_cnt_1m"] == 1 and f2["card_cnt_1h"] == 1
    fz.observe(txn(2, T0 + 30))
    later = fz.compute(txn(3, T0 + 2 * 86400))
    assert later["card_cnt_24h"] == 0  # evicted from the 24h window
    assert later["user_history_len"] == 2  # profile persists


def test_new_device_and_impossible_travel():
    fz = TransactionFeaturizer()
    for i in range(5):
        fz.observe(txn(i, T0 + i * 3600, amount=20 + i))
    f = fz.compute(
        txn(9, T0 + 5 * 3600 + 600, device_id="dX", ip="185.0.0.1", lat=51.5, lon=-0.13, amount=900)
    )
    assert f["is_new_device"] == 1 and f["is_new_ip"] == 1
    assert f["speed_kmh"] > 1000  # ~5,570 km in ~1 h
    assert f["amount_zscore"] > 3


def test_card_testing_velocity():
    fz = TransactionFeaturizer()
    for i in range(8):
        fz.observe(txn(i, T0 + i * 20, amount=1.0, merchant_id=f"m{i}"))
    f = fz.compute(txn(99, T0 + 200, amount=1.0))
    assert f["card_small_cnt_10m"] == 8 and f["card_distinct_merch_1h"] == 8


def test_vector_order_matches_feature_names():
    fz = TransactionFeaturizer()
    feats = fz.compute(txn(1, T0))
    v = fz.vector(feats)
    assert v.shape == (len(FEATURE_NAMES),)
    assert v[FEATURE_NAMES.index("amount")] == 20.0


def test_semantic_extractor_nan_without_text():
    ex = FraudSemanticExtractor(MockJevClient())
    rows = ex.extract_many(
        [txn(1, T0), txn(2, T0, memo="Bank security team: move funds to safe account immediately")]
    )
    assert all(math.isnan(v) for v in rows[0])
    assert max(rows[1]) > 0.9
    assert build_state(txn(3, T0)) is None


def test_models_beat_chance_and_hybrid_helps(fraud_trained):
    models, f, va = fraud_trained
    assert models.metrics["tabular"]["roc_auc"] > 0.95
    assert models.metrics["hybrid"]["pr_auc"] >= models.metrics["tabular"]["pr_auc"] - 0.02


def test_policy_and_cost_threshold():
    p = DecisionPolicy(0.2, 0.5, 0.8)
    assert [p.decide(x) for x in (0.1, 0.3, 0.6, 0.9)] == [
        FraudAction.APPROVE,
        FraudAction.STEP_UP,
        FraudAction.REVIEW,
        FraudAction.DECLINE,
    ]
    y = np.array([0, 0, 0, 1, 1])
    s = np.array([0.1, 0.2, 0.3, 0.7, 0.9])
    t, cost = choose_decline_threshold(y, s, np.array([10, 10, 10, 500, 500]))
    assert 0.3 < t <= 0.7 and cost == 0


def test_escalation_policy():
    e = EscalationPolicy()
    assert not e.should_escalate(txn(1, T0, amount=500), 0.5)  # no text
    assert e.should_escalate(txn(1, T0, amount=500, memo="rent", channel="p2p"), 0.0)
    assert not e.should_escalate(txn(1, T0, amount=5, memo="rent", channel="p2p"), 0.5)
    assert not e.should_escalate(txn(1, T0, amount=500, merchant_descriptor="SHOP"), 0.001)


def test_realtime_scorer_end_to_end(fraud_trained, fraud_stream, tmp_path):
    models, _, va = fraud_trained
    models.save(tmp_path / "f")
    models = FraudModels.load(tmp_path / "f")
    scorer = RealTimeFraudScorer(models, MockJevClient())
    cut = va[0]
    for t in fraud_stream.transactions[:cut]:
        scorer.featurizer.observe(t)
    window = fraud_stream.transactions[cut:]
    decisions = [scorer.score(t) for t in window]
    y = np.array([fraud_stream.labels[t.txn_id] for t in window])
    flagged = np.array([d.action != FraudAction.APPROVE for d in decisions])
    declined = np.array([d.action == FraudAction.DECLINE for d in decisions])
    assert flagged[y == 1].mean() > 0.7
    # Declines hit customers, so their rate on legit traffic must stay low. How many legit payments
    # are sent to REVIEW depends on thresholds tuned on a ~1,300-row validation window and on the
    # XGBoost version (10% on one, 41% on another), so it is deliberately not asserted here.
    assert declined[y == 0].mean() < 0.02
    hit = next(
        d
        for d, lab in zip(decisions, y, strict=True)
        if lab == 1 and d.action == FraudAction.DECLINE
    )
    assert hit.reasons and hit.reasons[0]["contribution"] > 0
    assert any(d.jev_used for d in decisions)


def test_score_batch_matches_sequential(fraud_trained, fraud_stream):
    models, _, va = fraud_trained
    txns = fraud_stream.transactions[: va[0] + 300]
    a = RealTimeFraudScorer(models, MockJevClient(), explain=False)
    b = RealTimeFraudScorer(models, MockJevClient(), explain=False)
    seq = [a.score(t).score for t in txns]
    bat = [d.score for i in range(0, len(txns), 64) for d in b.score_batch(txns[i : i + 64])]
    np.testing.assert_allclose(seq, bat, rtol=1e-6)
