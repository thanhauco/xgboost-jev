"""End-to-end benchmark on mock data: accuracy, robustness and latency.

    python -m xgboost_jev.bench --out docs/benchmarks.md --json artifacts/bench.json

Jev is the offline ``MockJevClient`` with an artificial round-trip delay, so
latency numbers measure this repo's orchestration around a simulated network
call, and accuracy numbers measure the pipelines on synthetic data. Neither is
a statement about the real Jev service.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from .fraud import EscalationPolicy, RealTimeFraudScorer, generate_transactions, train_fraud_models
from .fraud.features import FEATURE_NAMES
from .guardrail import (
    EnsembleGuard,
    LatencyRouter,
    generate_injection_dataset,
    jev_risk,
    train_guardrail,
)
from .guardrail.cascade import classification_metrics
from .guardrail.questions import injection_query
from .jev import MockJevClient, ResilientJevClient
from .monitoring import psi, summarize


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def bench_guardrail(n_train: int, n_test: int, jev_latency_ms: float, seed: int) -> dict:
    tr = generate_injection_dataset(n_train, split="train", seed=seed)
    te = generate_injection_dataset(n_test, split="test", seed=seed + 1)
    y = te.label.to_numpy()
    jev = MockJevClient()
    tricked = MockJevClient(counterfeit_susceptibility=0.8)

    t0 = time.perf_counter()
    static = train_guardrail(tr.text, tr.label, use_jev=False, seed=seed)
    cascade = train_guardrail(tr.text, tr.label, jev=jev, use_jev=True, seed=seed)
    train_s = time.perf_counter() - t0

    texts = list(te.text)
    jev_scores = np.array(
        [jev_risk(dict(r.probabilities)) for r in jev.ask_many([injection_query(t) for t in texts])]
    )
    tricked_scores = np.array(
        [
            jev_risk(dict(r.probabilities))
            for r in tricked.ask_many([injection_query(t) for t in texts])
        ]
    )
    s_static = static.score_many(texts)
    s_cascade = cascade.score_many(texts)
    outage = cascade.builder.transform(texts)
    outage[cascade.builder.jev_columns] = np.nan
    s_outage = cascade.score_frame(outage)
    ens = EnsembleGuard(static, tricked)
    s_ens = np.array([ens.score(t) for t in texts])

    rows = {
        "Jev only (noisy-OR)": (jev_scores, 0.5),
        "Jev only, counterfeit-steered": (tricked_scores, 0.5),
        "XGBoost static (meta + embeddings)": (s_static, static.threshold),
        "Cascade (meta + emb + Jev)": (s_cascade, cascade.threshold),
        "Cascade during Jev outage (Jev=NaN)": (s_outage, cascade.threshold),
        "Ensemble (static XGB + steered Jev)": (s_ens, ens.config.block_threshold),
    }
    models = {
        k: {m: _r(v) for m, v in classification_metrics(y, s, t).items()}
        for k, (s, t) in rows.items()
    }
    per_kind = {}
    for k, (s, t) in rows.items():
        hit = (s >= t) == y
        per_kind[k] = {
            kind: _r(hit[(te.kind == kind).to_numpy()].mean()) for kind in sorted(te.kind.unique())
        }

    # Latency: router over a sample with simulated Jev round trip.
    slow = ResilientJevClient(MockJevClient(latency_ms=jev_latency_ms))
    slow_tricked = MockJevClient(latency_ms=jev_latency_ms, counterfeit_susceptibility=0.8)
    router = LatencyRouter(slow, EnsembleGuard(static, slow_tricked))
    sample = texts[: min(400, len(texts))]
    decisions = [router.check(t) for t in sample]
    y_s = y[: len(sample)]
    blocked = np.array([d.action.value == "block" for d in decisions])
    lat = {
        stage: summarize([d.latency_ms for d in decisions if d.stage == stage])
        for stage in ("stage1", "stage2")
        if any(d.stage == stage for d in decisions)
    }
    lat["all"] = summarize([d.latency_ms for d in decisions])
    st = router.stats

    t1 = time.perf_counter()
    for t in sample:
        static.score(t)
    static_ms = (time.perf_counter() - t1) * 1000 / len(sample)

    return {
        "n_train": n_train,
        "n_test": n_test,
        "train_seconds": _r(train_s, 2),
        "models": models,
        "per_kind_accuracy": per_kind,
        "router": {
            "sample": len(sample),
            "jev_latency_ms": jev_latency_ms,
            "short_circuit_rate": _r(st.short_circuit_rate),
            "stage1_allow": st.stage1_allow,
            "stage1_block": st.stage1_block,
            "stage2": st.stage2,
            "counterfeit_escalations": st.counterfeit_escalations,
            "recall": _r(blocked[y_s == 1].mean()),
            "fpr": _r(blocked[y_s == 0].mean()),
            "latency": lat,
        },
        "static_xgb_ms_per_text": _r(static_ms),
    }


def bench_fraud(n_users: int, days: int, jev_latency_ms: float, seed: int) -> dict:
    stream = generate_transactions(n_users=n_users, days=days, seed=seed)
    jev = MockJevClient()
    t0 = time.perf_counter()
    models, f, va = train_fraud_models(stream, jev, seed=seed)
    train_s = time.perf_counter() - t0
    xh = np.hstack([f.tabular, f.semantic])[va]
    p_tab = models.tabular.inplace_predict(f.tabular[va])
    p_hyb = models.hybrid.inplace_predict(xh)
    y = f.labels[va]
    kinds = np.array(f.scenarios)[va]
    t_tab, t_hyb = models.tabular_policy.decline_at, models.hybrid_policy.decline_at
    per = {}
    for k in sorted(set(kinds[y == 1])):
        m = (kinds == k) & (y == 1)
        per[k] = {
            "n": int(m.sum()),
            "tabular_recall": _r((p_tab[m] >= t_tab).mean()),
            "hybrid_recall": _r((p_hyb[m] >= t_hyb).mean()),
        }
    legit = y == 0
    fpr = {
        "tabular": _r((p_tab[legit] >= t_tab).mean(), 5),
        "hybrid": _r((p_hyb[legit] >= t_hyb).mean(), 5),
    }
    thr = {"tabular": _r(t_tab), "hybrid": _r(t_hyb)}

    # Online replay of the validation window through the real-time scorer.
    cut = va[0]
    online = RealTimeFraudScorer(
        models, MockJevClient(latency_ms=jev_latency_ms), escalation=EscalationPolicy()
    )
    for t in stream.transactions[:cut]:
        online.featurizer.observe(t)
    window = stream.transactions[cut : cut + 3000]
    decisions = [online.score(t) for t in window]
    by_model = {
        m: summarize([d.latency_ms for d in decisions if d.model == m])
        for m in ("tabular", "hybrid")
        if any(d.model == m for d in decisions)
    }
    by_model["all"] = summarize([d.latency_ms for d in decisions])
    actions = pd.Series([d.action.value for d in decisions]).value_counts().to_dict()

    batch = RealTimeFraudScorer(models, MockJevClient(latency_ms=jev_latency_ms), explain=False)
    for t in stream.transactions[:cut]:
        batch.featurizer.observe(t)
    t1 = time.perf_counter()
    for i in range(0, len(window), 250):
        batch.score_batch(window[i : i + 250])
    tps = len(window) / (time.perf_counter() - t1)

    d = xgb.DMatrix(f.tabular[va][:1], feature_names=FEATURE_NAMES)
    t2 = time.perf_counter()
    for _ in range(2000):
        models.tabular.inplace_predict(f.tabular[va][:1])
    predict_us = (time.perf_counter() - t2) / 2000 * 1e6
    del d

    mid = len(f.labels) // 2
    drift = {
        "amount_psi_same_period": _r(psi(f.tabular[:mid, 0], f.tabular[mid:, 0]), 4),
        "amount_psi_x1.6_inflation": _r(psi(f.tabular[:mid, 0], f.tabular[mid:, 0] * 1.6), 4),
    }

    return {
        "n_transactions": len(stream.transactions),
        "fraud_rate": _r(f.labels.mean(), 4),
        "train_seconds": _r(train_s, 2),
        "metrics": models.metrics,
        "decline_threshold": thr,
        "per_scenario": per,
        "false_positive_rate": fpr,
        "online": {
            "window": len(window),
            "jev_latency_ms": jev_latency_ms,
            "escalation_rate": _r(np.mean([d.jev_used for d in decisions]), 4),
            "latency": by_model,
            "actions": actions,
            "batch_throughput_tps": _r(tps, 0),
            "tabular_predict_us": _r(predict_us, 1),
        },
        "drift": drift,
    }


def render_markdown(res: dict) -> str:
    g, fr = res["guardrail"], res["fraud"]
    env = res["environment"]
    out = [
        "# Benchmark results",
        "",
        f"Generated by `python -m xgboost_jev.bench` on {env['date']} "
        f"(Python {env['python']}, xgboost {env['xgboost']}, {env['machine']}, single-threaded XGBoost).",
        "",
        "> **Read this first.** All data is synthetic (`mock_data.py`) and Jev is the offline "
        "`MockJevClient` lexicon stand-in with a simulated "
        f"{g['router']['jev_latency_ms']:.0f} ms round trip. The numbers show how the pipelines "
        "behave relative to each other; they are not a measurement of the real Jev service or of "
        "production traffic. Re-run against a real Jev endpoint and your own labelled data before "
        "drawing conclusions.",
        "",
        "## 1. Prompt-injection guardrails",
        "",
        f"Train: {g['n_train']} texts from training templates. Test: {g['n_test']} texts from "
        "**held-out templates** (phrasings never seen in training).",
        "",
        "| Model | ROC-AUC | PR-AUC | Precision | Recall | FPR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, m in g["models"].items():
        out.append(
            f"| {k} | {m['roc_auc']} | {m['pr_auc']} | {m['precision']} | {m['recall']} | {m['fpr']} |"
        )
    kinds = list(next(iter(g["per_kind_accuracy"].values())).keys())
    out += [
        "",
        "### Accuracy by traffic type",
        "",
        "| Model | " + " | ".join(kinds) + " |",
        "|---|" + "---:|" * len(kinds),
    ]
    for k, v in g["per_kind_accuracy"].items():
        out.append(f"| {k} | " + " | ".join(str(v[x]) for x in kinds) + " |")
    r = g["router"]
    out += [
        "",
        "### Latency routing (Pattern 3)",
        "",
        f"{r['sample']} test texts through `LatencyRouter` (Stage 1 Jev pre-filter, Stage 2 ensemble).",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Short-circuit rate (decided at Stage 1) | {r['short_circuit_rate']} |",
        f"| Stage 1 allow / block / Stage 2 | {r['stage1_allow']} / {r['stage1_block']} / {r['stage2']} |",
        f"| Counterfeit-policy escalations (blocked fast-path allow) | {r['counterfeit_escalations']} |",
        f"| Recall / FPR | {r['recall']} / {r['fpr']} |",
        "",
        "| Path | mean ms | p50 ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in r["latency"].items():
        out.append(f"| {k} | {v['mean_ms']} | {v['p50_ms']} | {v['p95_ms']} | {v['p99_ms']} |")
    out += [
        "",
        f"Static XGBoost alone (features + predict, no network): **{g['static_xgb_ms_per_text']} ms/text**.",
        "",
        "## 2. Real-time fraud detection",
        "",
        f"{fr['n_transactions']} transactions, fraud rate {fr['fraud_rate']}, time-ordered split "
        f"(train on the first 70%, validate on the last 30%). Each model's decline threshold is "
        "chosen separately by expected cost (missed fraud $ + $15 per false decline).",
        "",
        "| Model | ROC-AUC | PR-AUC | Decline threshold | FPR at threshold |",
        "|---|---:|---:|---:|---:|",
        f"| Tabular XGBoost (streaming features) | {_r(fr['metrics']['tabular']['roc_auc'], 4)} | "
        f"{_r(fr['metrics']['tabular']['pr_auc'], 4)} | {fr['decline_threshold']['tabular']} | "
        f"{fr['false_positive_rate']['tabular']} |",
        f"| Hybrid XGBoost (+ Jev semantic) | {_r(fr['metrics']['hybrid']['roc_auc'], 4)} | "
        f"{_r(fr['metrics']['hybrid']['pr_auc'], 4)} | {fr['decline_threshold']['hybrid']} | "
        f"{fr['false_positive_rate']['hybrid']} |",
        "",
        "### Recall by fraud scenario",
        "",
        "| Scenario | n | Tabular | Hybrid |",
        "|---|---:|---:|---:|",
    ]
    for k, v in fr["per_scenario"].items():
        out.append(f"| {k} | {v['n']} | {v['tabular_recall']} | {v['hybrid_recall']} |")
    o = fr["online"]
    out += [
        "",
        "### Online scoring",
        "",
        f"{o['window']} validation transactions replayed through `RealTimeFraudScorer` with a "
        f"{o['jev_latency_ms']:.0f} ms simulated Jev call. Escalation rate to Jev: **{o['escalation_rate']}**.",
        "",
        "| Path | mean ms | p50 ms | p95 ms | p99 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, v in o["latency"].items():
        out.append(f"| {k} | {v['mean_ms']} | {v['p50_ms']} | {v['p95_ms']} | {v['p99_ms']} |")
    out += [
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Batch throughput (`score_batch`, 250/batch) | {o['batch_throughput_tps']} txn/s |",
        f"| Tabular `inplace_predict`, single row | {o['tabular_predict_us']} µs |",
        "| Actions | " + ", ".join(f"{k}: {v}" for k, v in o["actions"].items()) + " |",
        f"| PSI amount, first vs second half | {fr['drift']['amount_psi_same_period']} |",
        f"| PSI amount, with 1.6x inflation | {fr['drift']['amount_psi_x1.6_inflation']} |",
        "",
    ]
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="docs/benchmarks.md")
    ap.add_argument("--json", default="docs/benchmarks.json")
    ap.add_argument("--jev-latency-ms", type=float, default=25.0)
    ap.add_argument("--quick", action="store_true", help="smaller datasets for CI smoke runs")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args(argv)
    gs = (1200, 600) if a.quick else (4000, 2000)
    fs = (150, 10) if a.quick else (500, 30)
    res = {
        "environment": {
            "date": time.strftime("%Y-%m-%d"),
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "machine": platform.machine(),
        },
        "guardrail": bench_guardrail(*gs, a.jev_latency_ms, a.seed),
        "fraud": bench_fraud(*fs, a.jev_latency_ms, a.seed),
    }
    Path(a.json).parent.mkdir(parents=True, exist_ok=True)
    Path(a.json).write_text(json.dumps(res, indent=2, default=str))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(render_markdown(res))
    print(render_markdown(res))


if __name__ == "__main__":
    main()
