import math

import numpy as np
import pytest

from xgboost_jev.fraud import (
    AccountEvent,
    AdvancedFeaturizer,
    AdvancedFraudScorer,
    AdvancedModels,
    AnalystAssistant,
    ChampionChallenger,
    EntityGraph,
    FraudAction,
    InMemoryKV,
    IsotonicCalibrator,
    LabelEvent,
    LabelStore,
    PartitionedScorer,
    RecallMonitor,
    RedisKV,
    RulesEngine,
    SegmentedPolicy,
    Transaction,
    continual_backtest,
    expected_calibration_error,
    generate_advanced_stream,
    partition_for,
    propagate_entity_labels,
    replay_advanced,
    train_advanced,
)
from xgboost_jev.fraud.decay import DecayedCounterStore, DecayedMeanVar
from xgboost_jev.fraud.lifecycle import prob_challenger_worse
from xgboost_jev.fraud.pipeline import merged_events
from xgboost_jev.fraud.risk import RiskTables
from xgboost_jev.fraud.semantic import ADVANCED_FRAUD_QUESTIONS, FraudSemanticExtractor
from xgboost_jev.fraud.sequence import SequenceStore
from xgboost_jev.fraud.textsig import descriptor_signals, name_similarity
from xgboost_jev.jev import MockJevClient

DAY = 86400.0
T0 = 1_788_000_000.0


# ----------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def adv_world():
    s = generate_advanced_stream(n_users=150, days=45, seed=5, mule_start_day=30, fraud_scale=2.0)
    r = replay_advanced(s.transactions, s.account_events, s.label_events)
    sem = np.array(
        FraudSemanticExtractor(MockJevClient(), ADVANCED_FRAUD_QUESTIONS).extract_many(r.txns),
        dtype=np.float32,
    )
    labels = LabelStore()
    labels.extend(s.label_events)
    cut = s.start_ts + 28 * DAY
    models = train_advanced(r.x, sem, r.txns, labels, cut, rounds=150)
    return s, r, sem, labels, cut, models


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
        lat=40.7,
        lon=-74.0,
    )
    base.update(kw)
    return Transaction(**base)


# ----------------------------------------------------------------------------- storage
class FakeRedis:
    def __init__(self):
        self.h, self.ttl = {}, {}

    def hgetall(self, k):
        return {f.encode(): v.encode() for f, v in self.h.get(k, {}).items()}

    def pipeline(self):
        return self

    def hset(self, k, mapping):
        self.h.setdefault(k, {}).update(mapping)

    def expire(self, k, s):
        self.ttl[k] = s

    def execute(self):
        pass


def test_kv_backends_roundtrip():
    for kv in (InMemoryKV(), RedisKV(FakeRedis())):
        assert kv.get("x") is None
        kv.put("x", {"a": 1.5})
        assert kv.get("x") == {"a": 1.5}


def test_partition_for_is_stable_and_spread():
    parts = [partition_for(f"c{i}", 8) for i in range(2000)]
    assert parts == [partition_for(f"c{i}", 8) for i in range(2000)]
    counts = np.bincount(parts, minlength=8)
    assert counts.min() > 150


def test_decayed_counter_halves_per_half_life():
    st = DecayedCounterStore(half_lives={"1h": 3600.0})
    assert st.read("k", T0) == {"cnt_1h": 0.0, "sum_1h": 0.0}
    st.add("k", T0, 100.0)
    r = st.read("k", T0 + 3600)
    assert math.isclose(r["cnt_1h"], 0.5) and math.isclose(r["sum_1h"], 50.0)


def test_decayed_counter_on_redis_backend():
    st = DecayedCounterStore(RedisKV(FakeRedis()))
    st.add("card:c1", T0, 10.0)
    st.add("card:c1", T0, 5.0)
    assert math.isclose(st.read("card:c1", T0)["sum_1d"], 15.0)


def test_decayed_mean_var():
    mv = DecayedMeanVar()
    for x in [1.0, 2.0, 3.0, 4.0]:
        mv.add("s", T0, x)
    w, mean, var = mv.read("s", T0)
    assert w == 4 and math.isclose(mean, 2.5) and math.isclose(var, 1.25)


# ----------------------------------------------------------------------------- graph
def test_graph_components_fraud_and_hubs():
    g = EntityGraph(hub_degree=3)
    g.observe_txn(T0, "u1", "c1", "dA", "ip1", None)
    g.observe_txn(T0, "u2", "c2", "dA", "ip2", None)
    assert g.component("user:u1") == g.component("user:u2")
    g.mark_fraud("device:dA")
    assert g.component("user:u2")[1] == 1
    for k in range(3, 9):  # shared NAT IP becomes a hub and stops merging
        g.observe_txn(T0, f"u{k}", f"c{k}", f"d{k}", "nat", None)
    assert g.component("user:u8")[0] < 8
    assert g.two_hop_fraud("user:u3") == 0


def test_graph_payee_fan_in_window():
    g = EntityGraph(fan_in_window_s=7 * DAY)
    for k in range(5):
        g.observe_txn(T0 + k, f"u{k}", f"c{k}", None, None, "mule")
    st = g.payee_stats("mule", "u0", T0 + 10)
    assert st["payee_fan_in_7d"] == 5 and st["payee_new_for_user"] == 0
    assert g.payee_stats("mule", "u0", T0 + 8 * DAY)["payee_fan_in_7d"] == 0
    assert g.payee_stats("mule", "u9", T0 + 10)["payee_new_for_user"] == 1


def test_mark_fraud_before_first_sighting_counts():
    g = EntityGraph()
    g.mark_fraud("device:dZ")
    g.observe_txn(T0, "u1", "c1", "dZ", None, None)
    assert g.component("user:u1")[1] == 1


# ----------------------------------------------------------------------------- risk / sequence / text
def test_risk_rate_is_point_in_time():
    r = RiskTables()
    for _ in range(50):
        r.observe(["m:bad"], T0)
    before, _ = r.rate("m:bad", T0)
    for _ in range(10):
        r.label_fraud(["m:bad"], T0 + 1)
    after, vol = r.rate("m:bad", T0 + 1)
    assert after > 5 * before and vol > 3


def test_sequence_features():
    sq = SequenceStore()
    for k in range(50):  # common legit chain teaches the transition model
        sq.observe("other", T0 + k, "login")
        sq.observe("other", T0 + k + 0.5, "txn")
    assert math.isnan(sq.features("u1", T0)["seq_h_since_password_reset"])
    sq.observe("u1", T0, "login_new_device")
    sq.observe("u1", T0 + 60, "password_reset")
    sq.observe("u1", T0 + 120, "new_payee")
    f = sq.features("u1", T0 + 3 * 3600)
    assert math.isclose(f["seq_h_since_password_reset"], (3 * 3600 - 60) / 3600)
    assert f["seq_reset_then_payee_48h"] == 1 and f["seq_risky_24h"] == 3
    assert f["seq_markov_surprise"] > sq.features("other", T0 + 60)["seq_markov_surprise"]


def test_name_similarity():
    assert name_similarity("J Smith", "John Smith") > 0.9
    assert name_similarity("john smith", "John Smith") == 1.0
    assert name_similarity("Bank Safe Account", "Mei Nguyen") < 0.5
    assert math.isnan(name_similarity("", "John Smith"))


def test_descriptor_lookalikes():
    assert descriptor_signals("AMAZ0N-PRIME REFUND-CENTER")["desc_brand_lookalike"] > 0.9
    assert descriptor_signals("AMAZON-SECURE-BILLING-HELP")["desc_brand_lookalike"] > 0.9
    assert descriptor_signals("AMAZON MKTPLACE PMTS")["desc_brand_lookalike"] == 0.0
    assert descriptor_signals("LOCAL DINER")["desc_brand_lookalike"] < 0.5
    assert math.isnan(descriptor_signals("")["desc_leet"])


# ----------------------------------------------------------------------------- labels
def test_label_store_as_of_and_symmetric_maturity():
    ls = LabelStore()
    ls.add(LabelEvent(T0 + 10 * DAY, "a", 1, "chargeback", "card_testing"))
    ls.add(LabelEvent(T0 + 2 * DAY, "b", 1, "report", "app_scam"))
    assert ls.known_as_of(T0 + 5 * DAY) == {"b": 1}
    assert ls.fraud_types_as_of(T0 + 11 * DAY) == {"b": "app_scam", "a": "card_testing"}
    ts = np.array([T0, T0 + 1 * DAY, T0 + 9 * DAY])
    y, usable = ls.training_labels(["a", "b", "c"], ts, T0 + 11 * DAY, maturity_days=7)
    assert y.tolist() == [1, 1, 0] and usable.tolist() == [True, True, False]


def test_propagation_depends_on_fraud_type():
    rows = [
        txn("ring1", T0, user_id="s1", device_id="emu", ip="ip-r"),
        txn("ring2", T0 + DAY, user_id="s2", device_id="emu", ip="ip-r2"),
        txn("vic", T0, user_id="v", device_id="vdev", ip="ip-v", payee_id="mule"),
        txn("vic_legit", T0 + DAY, user_id="v", device_id="vdev", ip="ip-v"),
        txn("other", T0 + 2 * DAY, user_id="w", device_id="wdev", ip="ip-w", payee_id="mule"),
    ]
    y = np.zeros(len(rows), dtype=np.int32)
    out, n = propagate_entity_labels(rows, y, {"tring1": "low_and_slow", "tvic": "mule_ring"})
    assert out.tolist() == [1, 1, 1, 0, 1] and n == 4  # victim's own-device purchase stays legit


# ----------------------------------------------------------------------------- calibration / rules
def test_isotonic_calibration_reduces_ece():
    rng = np.random.default_rng(0)
    p_true = rng.uniform(0, 0.2, 5000)
    y = (rng.random(5000) < p_true).astype(int)
    raw = np.sqrt(p_true)  # over-confident scores
    cal = IsotonicCalibrator.fit(raw, y)
    assert expected_calibration_error(y, cal(raw)) < expected_calibration_error(y, raw) / 3
    assert IsotonicCalibrator(**cal.to_dict())(0.3) == cal(0.3)


def test_segmented_policy_falls_back():
    rng = np.random.default_rng(1)
    n = 2000
    p = rng.random(n)
    y = (rng.random(n) < p * 0.1).astype(int)
    segs = ["p2p|est"] * 1000 + ["atm|new"] * 1000
    y[1000:] = 0
    pol = SegmentedPolicy.fit(y, p, np.full(n, 100.0), segs)
    assert "p2p|est" in pol.segments and "atm|new" not in pol.segments
    assert pol.for_segment("atm|new") is pol.default
    assert (
        SegmentedPolicy.from_dict(pol.to_dict()).for_segment("p2p|est").decline_at
        == pol.segments["p2p|est"].decline_at
    )


def test_rules_engine():
    eng = RulesEngine()
    eng.add(
        {
            "id": "allow-vip",
            "action": "approve",
            "priority": 1,
            "conditions": [{"field": "user_id", "op": "in", "value": ["vip"]}],
        }
    )
    eng.add(
        {
            "id": "blk",
            "action": "decline",
            "priority": 10,
            "reason": "attack",
            "conditions": [
                {"field": "merchant_id", "op": "startswith", "value": "dk_"},
                {"field": "card_cnt_1h", "op": "ge", "value": 2},
            ],
        }
    )
    eng.add(
        {
            "id": "old",
            "action": "review",
            "expires_at": T0 - 1,
            "conditions": [{"field": "amount", "op": "gt", "value": 0}],
        }
    )
    t = txn(1, T0, merchant_id="dk_1")
    assert eng.evaluate(t, {"card_cnt_1h": 3}, now=T0).id == "blk"
    assert eng.evaluate(t, {"card_cnt_1h": 1}, now=T0) is None  # expired rule ignored
    assert (
        eng.evaluate(txn(2, T0, user_id="vip", merchant_id="dk_1"), {"card_cnt_1h": 9}, now=T0).id
        == "allow-vip"
    )
    assert eng.remove("blk") and not eng.remove("blk")
    with pytest.raises(ValueError):
        eng.add(
            {
                "id": "x",
                "action": "decline",
                "conditions": [{"field": "a", "op": "regex", "value": 1}],
            }
        )


# ----------------------------------------------------------------------------- pipeline / models
def test_replay_is_point_in_time(adv_world):
    s, r, *_ = adv_world
    n = 3000
    cut_ts = r.txns[n].timestamp
    prefix = replay_advanced(s.transactions, s.account_events, s.label_events, stop_at=cut_ts)
    m = len(prefix.txns)
    np.testing.assert_array_equal(np.nan_to_num(prefix.x, nan=-9), np.nan_to_num(r.x[:m], nan=-9))


def test_label_events_update_graph_reputation():
    fz = AdvancedFeaturizer()
    fz.process(txn(1, T0, device_id="dX", user_id="a", card_id="ca"))
    before = fz.compute(txn(2, T0 + 60, device_id="dX", user_id="b", card_id="cb"))
    fz.on_label(LabelEvent(T0 + 30, "t1", 1, "chargeback", "card_testing"))
    after = fz.compute(txn(2, T0 + 60, device_id="dX", user_id="b", card_id="cb"))
    assert before["g_device_fraud"] == 0 and after["g_device_fraud"] == 1


def test_advanced_models_train_save_load(adv_world, tmp_path):
    _, r, sem, _, _, models = adv_world
    assert models.metrics["n_train"] > 1000 and models.metrics["train_fraud"] > 10
    models.save(tmp_path / "adv")
    loaded = AdvancedModels.load(tmp_path / "adv")
    np.testing.assert_allclose(
        models.predict_hybrid(r.x[:500], sem[:500]), loaded.predict_hybrid(r.x[:500], sem[:500])
    )


def test_advanced_beats_chance_on_future(adv_world):
    from sklearn.metrics import average_precision_score

    s, r, sem, _, cut, models = adv_world
    ts = np.array([t.timestamp for t in r.txns])
    y = np.array([s.labels[t.txn_id] for t in r.txns])
    va = ts >= cut
    ap = average_precision_score(y[va], models.predict_hybrid(r.x[va], sem[va], False))
    assert ap > 3 * y[va].mean()


def _online_run(adv_world, n_score=400, on_txn=None, **kw):
    """Warm the scorer on everything before the cut, then score live with ALL events interleaved."""
    s, r, _, _, cut, models = adv_world
    sc = AdvancedFraudScorer(models, kw.pop("jev", MockJevClient()), **kw)
    out = []
    for e in merged_events(s.transactions, s.account_events, s.label_events):
        if isinstance(e, Transaction):
            if e.timestamp < cut:
                sc.featurizer.process(e)
                continue
            if len(out) >= n_score:
                break
            t = on_txn(e) if on_txn else e
            out.append((t, sc.score(t)))
        elif isinstance(e, AccountEvent):
            sc.on_account_event(e)
        else:
            sc.on_label(e)
    return sc, out


def test_online_scorer_matches_offline(adv_world):
    s, r, _, _, cut, models = adv_world
    _, out = _online_run(adv_world, 600)
    idx = {t.txn_id: i for i, t in enumerate(r.txns)}
    p_off = models.predict_tabular(r.x, True)
    for t, d in out:
        assert abs(d.p_tabular - p_off[idx[t.txn_id]]) < 1e-5
        assert d.segment and d.action in FraudAction


def test_online_scorer_rules_and_budget(adv_world):
    _, out = _online_run(
        adv_world, 1500, jev=MockJevClient(latency_ms=60), budget_ms=20, explain=False
    )
    esc = [d for _, d in out if d.jev_skipped]
    assert (
        esc and esc[0].jev_skipped == "timeout" and all(d.jev_skipped == "budget" for d in esc[1:])
    )
    assert not any(d.jev_used for _, d in out)
    assert max(d.latency_ms for d in esc[1:]) < 40
    sc, _ = _online_run(adv_world, 1)
    sc.rules.add(
        {
            "id": "big",
            "action": "review",
            "conditions": [{"field": "amount", "op": "gt", "value": 1e6}],
        }
    )
    t0 = out[-1][0]
    d = sc.score(Transaction(**{**t0.to_dict(), "txn_id": "big1", "amount": 2e6}))
    assert d.model == "rule" and d.rule == "big" and d.action == FraudAction.REVIEW


# ----------------------------------------------------------------------------- lifecycle / analyst / scaling
def test_continual_backtest_scores_live_period(adv_world):
    s, r, sem, labels, cut, _ = adv_world
    res = continual_backtest(
        r.x, sem, r.txns, labels, s.start_ts, cut, every_days=7, rounds=80, update_rounds=40
    )
    ts = np.array([t.timestamp for t in r.txns])
    assert np.isnan(res.scores[ts < cut]).all() and not np.isnan(res.scores[ts >= cut]).any()
    assert len(res.history) >= 2 and res.final is not None


def test_prob_challenger_worse():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.05).astype(int)
    good = y + rng.normal(0, 0.3, 400)
    bad = rng.random(400)
    assert prob_challenger_worse(y, good, bad) > 0.9
    assert prob_challenger_worse(y, bad, good) < 0.1


def test_champion_challenger_routing(adv_world):
    _, r, sem, _, _, models = adv_world
    cc = ChampionChallenger(models, models, challenger_share=0.1)
    ids = [t.txn_id for t in r.txns[:3000]]
    cc.score(ids, r.x[:3000], sem[:3000])
    rep = cc.report({i: 0 for i in ids} | {ids[0]: 1})
    assert 0.05 < rep["challenger_traffic"] < 0.15
    assert cc.routed_to_challenger(ids[5]) == cc.routed_to_challenger(ids[5])


def test_recall_monitor_alerts_on_drop():
    mon = RecallMonitor(window=20, min_labels=5, drop_alert=0.3)
    for i in range(10):
        mon.record_decision(f"a{i}", True)
        mon.on_label(LabelEvent(T0, f"a{i}", 1, fraud_type="card_testing"))
    mon.freeze_baseline()
    for i in range(20):
        mon.record_decision(f"b{i}", False)
        mon.on_label(LabelEvent(T0, f"b{i}", 1, fraud_type="card_testing"))
        mon.record_decision(f"m{i}", False)
        mon.on_label(LabelEvent(T0, f"m{i}", 1, fraud_type="mule_ring"))
    rep = mon.report()
    assert (
        rep["card_testing"]["alert"]
        and rep["mule_ring"]["alert"]
        and rep["mule_ring"]["baseline"] is None
    )


def test_analyst_case_and_feedback(adv_world):
    s = adv_world[0]
    sc, out = _online_run(adv_world, 4000)
    t, d = next(
        (t, d)
        for t, d in out
        if s.labels[t.txn_id] and d.action != FraudAction.APPROVE and d.reasons
    )
    case = AnalystAssistant(MockJevClient()).build_case(t, d, sc.featurizer)
    assert (
        case.reasons
        and case.reasons[0] in case.narrative
        and set(case.triage) == {"t_app_scam", "t_ato", "t_mule"}
    )
    ev = AnalystAssistant.record_decision(t.txn_id, True, "mule_ring", ts=t.timestamp + 60)
    assert ev.source == "analyst" and ev.label == 1
    sc.on_label(ev)


def test_partitioned_scorer_routes_by_card():
    class Stub:
        def __init__(self, i):
            self.i, self.labels = i, 0

        def score(self, t):
            return self.i

        def on_label(self, ev):
            self.labels += 1

        def on_account_event(self, ev):
            pass

    ps = PartitionedScorer(4, Stub)
    t = txn(1, T0, card_id="card-42")
    assert ps.score(t) == ps.partition_of("card-42") == ps.score(txn(2, T0, card_id="card-42"))
    ps.broadcast_label(LabelEvent(T0, "x", 1))
    assert all(p.labels == 1 for p in ps.partitions)


def test_api_v2(adv_world, static_model, fraud_trained):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from xgboost_jev.fraud import RealTimeFraudScorer
    from xgboost_jev.guardrail import EnsembleGuard, LatencyRouter
    from xgboost_jev.serving.api import create_app

    jev = MockJevClient()
    sc, out = _online_run(adv_world, 2)
    future = [t for t, _ in out]
    router = LatencyRouter(jev, EnsembleGuard(static_model, jev))
    c = TestClient(create_app(router, RealTimeFraudScorer(fraud_trained[0], jev), advanced=sc))
    t = future[0].to_dict()
    r = c.post("/v2/fraud/score", json=t)
    assert r.status_code == 200 and "segment" in r.json()
    assert c.get(f"/v2/fraud/case/{t['txn_id']}").json()["narrative"]
    assert c.post(
        "/v2/fraud/events", json={"user_id": t["user_id"], "kind": "password_reset"}
    ).json() == {"ok": True}
    assert (
        c.post(
            "/v2/fraud/labels", json={"txn_id": t["txn_id"], "fraud_type": "app_scam"}
        ).status_code
        == 200
    )
    rule = {
        "id": "r1",
        "action": "decline",
        "conditions": [{"field": "amount", "op": "gt", "value": 0}],
    }
    assert c.post("/v2/fraud/rules", json=rule).status_code == 200
    assert [x["id"] for x in c.get("/v2/fraud/rules").json()] == ["r1"]
    nxt = future[1].to_dict()
    assert c.post("/v2/fraud/score", json=nxt).json()["rule"] == "r1"
    assert c.delete("/v2/fraud/rules/r1").status_code == 200
    assert c.post("/v2/fraud/rules", json={**rule, "action": "explode"}).status_code == 422
    bare = TestClient(create_app(router, RealTimeFraudScorer(fraud_trained[0], jev)))
    assert bare.post("/v2/fraud/score", json=t).status_code == 503
