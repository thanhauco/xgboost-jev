import json

from xgboost_jev.bench_advanced import _aggregate, bench_fraud_advanced, render_advanced


def test_aggregate_mean_and_sd():
    def run(a, b):
        return {
            "s": {
                "m": {"pr_auc": a, "recall": b, "fraud_usd_caught": 0.5, "per_scenario": {"x": a}}
            }
        }

    agg = _aggregate([run(0.2, 0.4), run(0.4, 0.6)])["s"]["m"]
    assert agg["pr_auc"] == 0.3 and agg["recall"] == 0.5 and agg["n_seeds"] == 2
    assert abs(agg["pr_auc_sd"] - 0.1414) < 1e-3 and agg["per_scenario"] == {"x": 0.3}


def test_advanced_benchmark_smoke_and_render():
    r = bench_fraud_advanced(n_users=60, days=40, cut_day=26.0, mule_start_day=28.0, seeds=(21, 22))
    json.dumps(r)  # must be JSON-serialisable
    assert r["config"]["seeds"] == [21, 22]
    for section in ("static", "ablations", "group_ablation", "continual"):
        assert r[section], section
    assert r["static"]["advanced hybrid (+Jev)"]["n_seeds"] == 2
    assert {"mule_vs_legit_p2p", "label_handling", "calibration", "online"} <= set(r)
    md = "\n".join(render_advanced(r))
    assert "mean ± standard deviation over 2 generator seeds" in md
    assert "Leave-one-feature-group-out" in md and "Alert queue" in md
