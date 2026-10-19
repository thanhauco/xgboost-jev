"""Benchmark for the advanced fraud stack (``python -m xgboost_jev.bench`` runs it).

Every model is trained at the cut day using only labels KNOWN by then and is
evaluated on the following days against the true labels. Recall is measured at
a fixed alert budget (review capacity), so models compete for the same slots.

Comparisons are repeated over several generator seeds and reported as mean and
standard deviation: with a few hundred frauds in the evaluation window, a single
seed is easily misleading. Calibration, the online scorer and the gain table are
reported for the first seed only.
"""

from __future__ import annotations

import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss

from .fraud import (
    AccountEvent,
    AdvancedFraudScorer,
    LabelEvent,
    LabelStore,
    Transaction,
    continual_backtest,
    generate_advanced_stream,
    replay_advanced,
    train_advanced,
)
from .fraud.advanced import segments_for
from .fraud.calibration import SegmentedPolicy
from .fraud.features import FEATURE_NAMES
from .fraud.pipeline import (
    ADVANCED_FEATURE_NAMES,
    DECAY_NAMES,
    DEVICE_NAMES,
    GRAPH_NAMES,
    RISK_NAMES,
    SEQ_NAMES,
    TEXT_NAMES,
    merged_events,
)
from .fraud.semantic import ADVANCED_FRAUD_QUESTIONS, FraudSemanticExtractor
from .jev import MockJevClient
from .monitoring import summarize

DAY = 86400.0
GROUPS = {
    "v1 velocity/behaviour/geo": FEATURE_NAMES,
    "decayed counters": DECAY_NAMES,
    "entity graph / payee": GRAPH_NAMES,
    "merchant + peer risk": RISK_NAMES,
    "account-event sequence": SEQ_NAMES,
    "payee-name + descriptor text": TEXT_NAMES,
    "device signals": DEVICE_NAMES,
}
DEFAULT_SEEDS = (21, 22, 23)


def _r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


@dataclass
class _World:
    s: object
    x: np.ndarray
    sem: np.ndarray
    txns: list
    labels: LabelStore
    cut: float
    ts: np.ndarray
    y: np.ndarray
    kinds: np.ndarray
    amounts: np.ndarray
    va: np.ndarray
    alert_rate: float
    seed: int

    def evaluate(self, p: np.ndarray) -> dict:
        va, y, kinds = self.va, self.y, self.kinds
        pv, yv, kv = p[va], y[va], kinds[va]
        k = max(1, int(self.alert_rate * len(pv)))
        flag = np.zeros(len(pv), bool)
        flag[np.argsort(-pv)[:k]] = True
        fraud_usd = self.amounts[va][yv == 1].sum()
        return {
            "pr_auc": float(average_precision_score(yv, pv)),
            "recall": float(flag[yv == 1].mean()),
            "fraud_usd_caught": float(self.amounts[va][flag & (yv == 1)].sum() / fraud_usd),
            "per_scenario": {
                sc: float(flag[(kv == sc) & (yv == 1)].mean()) for sc in sorted(set(kv[yv == 1]))
            },
        }


def _build_world(seed, n_users, days, cut_day, mule_start_day, alert_rate) -> tuple[_World, dict]:
    t0 = time.perf_counter()
    s = generate_advanced_stream(
        n_users=n_users, days=days, seed=seed, mule_start_day=mule_start_day
    )
    gen_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    rep = replay_advanced(s.transactions, s.account_events, s.label_events)
    replay_us = (time.perf_counter() - t1) / max(len(rep.txns), 1) * 1e6
    txns = rep.txns
    sem = np.array(
        FraudSemanticExtractor(MockJevClient(), ADVANCED_FRAUD_QUESTIONS).extract_many(txns),
        dtype=np.float32,
    )
    labels = LabelStore()
    labels.extend(s.label_events)
    cut = s.start_ts + cut_day * DAY
    ts = np.array([t.timestamp for t in txns])
    w = _World(
        s,
        rep.x,
        sem,
        txns,
        labels,
        cut,
        ts,
        np.array([s.labels[t.txn_id] for t in txns]),
        np.array([s.scenarios[t.txn_id] for t in txns]),
        np.array([t.amount for t in txns]),
        ts >= cut,
        alert_rate,
        seed,
    )
    return w, {"generate_s": gen_s, "replay_us": replay_us}


def _experiments(w: _World, seed: int) -> tuple[dict, dict]:
    """Run every compared configuration; returns ({section: {row: metrics}}, extras)."""
    x, sem, txns, labels, cut = w.x, w.sem, w.txns, w.labels, w.cut
    out: dict[str, dict] = {"static": {}, "ablations": {}, "continual": {}, "group_ablation": {}}
    extras: dict = {}
    v1 = train_advanced(x, sem, txns, labels, cut, tabular_names=FEATURE_NAMES, seed=seed)
    adv = train_advanced(x, sem, txns, labels, cut, seed=seed)
    extras["adv"] = adv
    p_adv = adv.predict_hybrid(x, sem, False)
    out["static"] = {
        "v1 tabular": w.evaluate(v1.predict_tabular(x, False)),
        "v1 hybrid (+Jev)": w.evaluate(v1.predict_hybrid(x, sem, False)),
        "advanced tabular": w.evaluate(adv.predict_tabular(x, False)),
        "advanced hybrid (+Jev)": w.evaluate(p_adv),
    }
    oracle = LabelStore()
    oracle.extend(
        LabelEvent(t.timestamp, t.txn_id, 1, "oracle", w.s.scenarios[t.txn_id])
        for t in txns
        if w.s.labels[t.txn_id]
    )
    out["ablations"] = {
        "default (as-of labels, propagation, monotone)": out["static"]["advanced hybrid (+Jev)"],
        "without entity label propagation": w.evaluate(
            train_advanced(x, sem, txns, labels, cut, propagate=False, seed=seed).predict_hybrid(
                x, sem, False
            )
        ),
        "without monotone constraints": w.evaluate(
            train_advanced(x, sem, txns, labels, cut, monotone=False, seed=seed).predict_hybrid(
                x, sem, False
            )
        ),
        "oracle labels (upper bound, not achievable)": w.evaluate(
            train_advanced(x, sem, txns, oracle, cut, seed=seed).predict_hybrid(x, sem, False)
        ),
    }
    for g, names in GROUPS.items():
        keep = [n for n in ADVANCED_FEATURE_NAMES if n not in set(names)]
        m = train_advanced(x, sem, txns, labels, cut, tabular_names=keep, seed=seed)
        out["group_ablation"][f"minus {g}"] = w.evaluate(m.predict_hybrid(x, sem, False))
    history = {}
    out["continual"]["static (trained once at cut)"] = out["static"]["advanced hybrid (+Jev)"]
    for name, kw in [
        ("weekly, naive gate (must beat champion)", {"block_confidence": 0.0}),
        ("weekly, bootstrap gate", {}),
        ("weekly, cold retrain (no warm start)", {"warm_start": False}),
        ("weekly, v1 features, bootstrap gate", {"tabular_names": FEATURE_NAMES}),
    ]:
        res = continual_backtest(x, sem, txns, labels, w.s.start_ts, cut, seed=seed, **kw)
        out["continual"][name] = w.evaluate(res.scores) | {
            "promotions": float(sum(h.promoted for h in res.history)),
            "retrains": float(len(res.history)),
        }
        history[name] = [
            {
                "day": _r(h.as_of_day, 1),
                "promoted": h.promoted,
                "shadow_frauds": h.eval_fraud,
                "p_challenger_worse": _r(h.p_challenger_worse, 2),
            }
            for h in res.history
        ]
    extras["history"] = history
    return out, extras


def _aggregate(runs: list[dict]) -> dict:
    agg: dict = {}
    for section in runs[0]:
        agg[section] = {}
        for row in runs[0][section]:
            rows = [r[section][row] for r in runs]
            d: dict = {"n_seeds": len(rows)}
            for k in ("pr_auc", "recall", "fraud_usd_caught", "promotions", "retrains"):
                if k in rows[0]:
                    v = np.array([r[k] for r in rows])
                    d[k] = _r(v.mean())
                    d[k + "_sd"] = _r(v.std(ddof=1) if len(v) > 1 else 0.0)
            scen = defaultdict(list)
            for r in rows:
                for sc, v in r["per_scenario"].items():
                    scen[sc].append(v)
            d["per_scenario"] = {sc: _r(float(np.mean(v)), 2) for sc, v in sorted(scen.items())}
            agg[section][row] = d
    return agg


def _primary_extras(w: _World, adv, perf: dict, jev_latency_ms: float) -> dict:
    x, sem, txns, va, y = w.x, w.sem, w.txns, w.va, w.y
    # graph reputation: does a STATIC model benefit as mule labels reach the graph?
    p_static = adv.predict_hybrid(x, sem, False)
    thr = np.sort(p_static[va])[::-1][max(1, int(w.alert_rate * va.sum())) - 1]
    mule = va & (w.kinds == "mule_ring")
    rep_early = rep_late = None
    if mule.sum() >= 4:
        mts = w.ts[mule]
        mid = np.median(mts)
        rep_early = _r((p_static[mule][mts < mid] >= thr).mean(), 2)
        rep_late = _r((p_static[mule][mts >= mid] >= thr).mean(), 2)

    # calibration, judged where it matters: inside the alert queue
    raw = adv.predict_hybrid(x[va], sem[va], calibrated=False)
    cal = adv.hybrid_cal(raw)
    yv = y[va]
    seg = segments_for([t for t, v in zip(txns, va, strict=True) if v], x[va])
    amt = w.amounts[va]
    k = max(1, int(w.alert_rate * len(raw)))
    top = np.argsort(-cal)[:k]
    pol = adv.hybrid_policy
    calibration = {
        "eval_fraud_rate": _r(yv.mean(), 4),
        "train_fraud_rate": _r(adv.metrics["train_fraud"] / max(adv.metrics["n_train"], 1), 4),
        "queue_observed_fraud_rate": _r(yv[top].mean(), 3),
        "queue_mean_raw_score": _r(float(raw[top].mean()), 3),
        "queue_mean_calibrated_score": _r(float(cal[top].mean()), 3),
        "brier_raw": _r(brier_score_loss(yv, np.clip(raw, 0, 1)), 5),
        "brier_calibrated": _r(brier_score_loss(yv, np.clip(cal, 0, 1)), 5),
        "brier_constant_rate": _r(brier_score_loss(yv, np.full(len(yv), yv.mean())), 5),
        "cost_global_threshold": _r(
            SegmentedPolicy(pol.default).expected_cost(yv, cal, amt, seg), 0
        ),
        "cost_segment_thresholds": _r(pol.expected_cost(yv, cal, amt, seg), 0),
        "segments_tuned": {kk: _r(v.decline_at) for kk, v in sorted(pol.segments.items())},
        "default_threshold": _r(pol.default.decline_at),
    }
    # the deployed decision path: calibrated score -> segment policy -> any non-approve action
    flagged = np.array(
        [
            pol.for_segment(sg).decide(float(p)).value != "approve"
            for sg, p in zip(seg, cal, strict=True)
        ]
    )
    caught = int((flagged & (yv == 1)).sum())
    calibration.update(
        {
            "policy_flag_rate": _r(float(flagged.mean()), 4),
            "policy_recall": _r(caught / max(int(yv.sum()), 1), 3),
            "policy_precision": _r(caught / max(int(flagged.sum()), 1), 3),
            "raw_topk_recall": _r(float(yv[np.argsort(-raw)[:k]].sum() / max(int(yv.sum()), 1)), 3),
            "raw_topk_precision": _r(float(yv[np.argsort(-raw)[:k]].mean()), 3),
            "calibrated_distinct_values": int(len(np.unique(cal))),
            "calibrated_zero_share": _r(float((cal == 0).mean()), 3),
            "raw_distinct_values": int(len(np.unique(raw))),
        }
    )

    # online scorer: latency, escalation, budgets
    def online(budget_ms, n=3000):
        sc = AdvancedFraudScorer(
            adv, MockJevClient(latency_ms=jev_latency_ms), budget_ms=budget_ms, explain=False
        )
        out = []
        for e in merged_events(w.s.transactions, w.s.account_events, w.s.label_events):
            if isinstance(e, Transaction):
                if e.timestamp < w.cut:
                    sc.featurizer.process(e)
                elif len(out) < n:
                    out.append(sc.score(e))
                else:
                    break
            elif isinstance(e, AccountEvent):
                sc.on_account_event(e)
            else:
                sc.on_label(e)
        lat = defaultdict(list)
        for d in out:
            lat[d.model].append(d.latency_ms)
        return {
            "escalation_rate": _r(np.mean([d.jev_used for d in out]), 4),
            "skipped": dict(Counter(d.jev_skipped for d in out if d.jev_skipped)),
            "latency": {kk: summarize(v) for kk, v in lat.items()}
            | {"all": summarize([d.latency_ms for d in out])},
        }

    online_res = {
        "no budget": online(None),
        f"{int(jev_latency_ms * 0.6)} ms budget (Jev cannot fit)": online(jev_latency_ms * 0.6),
        f"{int(jev_latency_ms * 2)} ms budget (Jev fits)": online(jev_latency_ms * 2),
    }
    gain = adv.hybrid.get_score(importance_type="total_gain")
    total = sum(gain.values()) or 1.0
    groups = {g: _r(sum(gain.get(n, 0.0) for n in names) / total, 3) for g, names in GROUPS.items()}
    groups["Jev semantic"] = _r(
        sum(v for kk, v in gain.items() if kk.startswith("jev_")) / total, 3
    )
    # how separable the drifting mule ring is on the evaluation period (vs legitimate P2P)
    cols = [
        "payee_age_days",
        "payee_new_for_user",
        "payee_fan_in_7d",
        "g_payee_comp_fraud",
        "cop_similarity",
    ]
    leg = va & (w.kinds == "legit") & np.array([t.channel == "p2p" for t in txns])
    mule_vs_legit = {}
    for col in cols:
        j = ADVANCED_FEATURE_NAMES.index(col)
        mule_vs_legit[col] = {
            "mule_median": _r(float(np.nanmedian(x[mule, j])), 2) if mule.any() else None,
            "legit_p2p_median": _r(float(np.nanmedian(x[leg, j])), 2) if leg.any() else None,
        }
    label_handling = {
        "labels_known_at_cut": int(sum(1 for e in w.s.label_events if e.timestamp <= w.cut)),
        "fraud_before_cut": int(((w.ts < w.cut) & (y == 1)).sum()),
        "train_fraud_rows": adv.metrics["train_fraud"],
        "propagated_rows": adv.metrics["propagated"],
        "train_rows": adv.metrics["n_train"],
    }
    return {
        "mule_vs_legit_p2p": mule_vs_legit,
        "label_handling": label_handling,
        "graph_reputation": {
            "mule_recall_first_half": rep_early,
            "mule_recall_second_half": rep_late,
        },
        "calibration": calibration,
        "online": online_res,
        "feature_gain": groups,
        "replay_us_per_txn": _r(perf["replay_us"], 1),
    }


def bench_fraud_advanced(
    n_users=500,
    days=60,
    cut_day=35.0,
    mule_start_day=38.0,
    alert_rate=0.015,
    jev_latency_ms=25.0,
    seeds=DEFAULT_SEEDS,
) -> dict:
    t0 = time.perf_counter()
    runs, extras_first, data_first = [], None, None
    for i, seed in enumerate(seeds):
        w, perf = _build_world(seed, n_users, days, cut_day, mule_start_day, alert_rate)
        res, extras = _experiments(w, seed)
        runs.append(res)
        if i == 0:
            first = {"adv": extras.pop("adv"), "history": extras["history"]}
            extras_first = _primary_extras(w, first["adv"], perf, jev_latency_ms) | {
                "history": first["history"]
            }
            kinds = w.kinds
            data_first = {
                "transactions": len(w.txns),
                "account_events": len(w.s.account_events),
                "fraud": int(w.y.sum()),
                "eval_transactions": int(w.va.sum()),
                "eval_fraud": int(w.y[w.va].sum()),
                "eval_fraud_by_scenario": {
                    kk: int(((kinds == kk) & w.va & (w.y == 1)).sum())
                    for kk in sorted(set(kinds[w.va & (w.y == 1)]))
                },
                "labels_known_at_cut": int(
                    sum(1 for e in w.s.label_events if e.timestamp <= w.cut)
                ),
                "fraud_before_cut": int(((w.ts < w.cut) & (w.y == 1)).sum()),
            }
    agg = _aggregate(runs)
    return {
        "config": {
            "cut_day": cut_day,
            "days": days,
            "mule_start_day": mule_start_day,
            "alert_rate": alert_rate,
            "seeds": list(seeds),
            "n_users": n_users,
        },
        "first_seed_data": data_first,
        **agg,
        **extras_first,
        "total_seconds": _r(time.perf_counter() - t0, 0),
    }


def _pm(d: dict, k: str, pct: bool = False) -> str:
    v, sd = d.get(k), d.get(k + "_sd")
    if v is None:
        return ""
    return f"{v:.3f} ± {sd:.3f}" if sd else f"{v:.3f}"


def _table(rows: dict, scen: list[str], extra: tuple[str, ...] = ()) -> list[str]:
    head = (
        "| Model | PR-AUC | Recall | Fraud $ caught | "
        + " | ".join(scen)
        + "".join(f" | {e}" for e in extra)
        + " |"
    )
    out = [head, "|---|" + "---:|" * (3 + len(scen) + len(extra))]
    for k, v in rows.items():
        cells = [_pm(v, "pr_auc"), _pm(v, "recall"), _pm(v, "fraud_usd_caught")]
        cells += [str(v["per_scenario"].get(sc, "")) for sc in scen]
        cells += [("" if v.get(e) is None else f"{v[e]:g}") for e in extra]
        out.append(f"| {k} | " + " | ".join(cells) + " |")
    return out


def render_advanced(a: dict) -> list[str]:
    cfg, d = a["config"], a["first_seed_data"]
    scen = list(d["eval_fraud_by_scenario"])
    n = len(cfg["seeds"])
    out = [
        "## 3. Advanced fraud stack",
        "",
        f"`generate_advanced_stream`, {cfg['n_users']} users over {cfg['days']} days. Models train at **day "
        f"{cfg['cut_day']:g}** using only the labels *known by then* and are evaluated on days "
        f"{cfg['cut_day']:g}-{cfg['days']}. The money-mule ring starts on day {cfg['mule_start_day']:g}, after "
        f"training, to test drift. Recall is measured at a fixed **{cfg['alert_rate']:.1%} alert budget** (review "
        f"capacity). Tables show **mean ± standard deviation over {n} generator seeds** ({', '.join(map(str, cfg['seeds']))});"
        " per-scenario columns are means. Seed {0} (first-seed detail below): {1:,} transactions, {2:,} account events, "
        "{3} fraud; the cut sees only {4} of {5} pre-cut frauds labelled; {6:,} evaluation transactions, {7} fraud "
        "({8}).".format(
            cfg["seeds"][0],
            d["transactions"],
            d["account_events"],
            d["fraud"],
            d["labels_known_at_cut"],
            d["fraud_before_cut"],
            d["eval_transactions"],
            d["eval_fraud"],
            ", ".join(f"{k} {v}" for k, v in d["eval_fraud_by_scenario"].items()),
        ),
        "",
        "### Static models (trained once at the cut)",
        "",
    ]
    out += _table(a["static"], scen)
    out += ["", "### What each training safeguard is worth (advanced hybrid)", ""]
    out += _table(a["ablations"], scen)
    out += [
        "",
        "### Leave-one-feature-group-out (advanced hybrid)",
        "",
        "Diagnostic only. Choosing features by this table would be selecting on the evaluation period; do that on a "
        "time-later validation slice of your own data.",
        "",
    ]
    out += _table(a["group_ablation"], scen)
    out += [
        "",
        "### Continual retraining on delayed labels",
        "",
        "Weekly retrains (warm-started from the live champion unless stated) are shadow-tested on the last 5 days "
        "with the labels known at that moment; each period is scored by whichever model was live then.",
        "",
    ]
    out += _table(a["continual"], scen, ("promotions", "retrains"))
    hist = a["history"].get("weekly, bootstrap gate", [])
    if hist:
        out += [
            "",
            f"Shadow decisions, bootstrap gate, seed {cfg['seeds'][0]}: "
            + "; ".join(
                f"day {h['day']}: {'promoted' if h['promoted'] else 'kept champion'} "
                f"({h['shadow_frauds']} known frauds, P(worse)={h['p_challenger_worse']})"
                for h in hist
            )
            + ".",
        ]
    c, g = a["calibration"], a["graph_reputation"]
    out += [
        "",
        f"### Calibration, thresholds, graph reputation (seed {cfg['seeds'][0]})",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Fraud rate: training slice / evaluation period | {c['train_fraud_rate']} / {c['eval_fraud_rate']} |",
        f"| Alert queue: observed fraud rate | {c['queue_observed_fraud_rate']} |",
        f"| Alert queue: mean raw score / mean calibrated score | {c['queue_mean_raw_score']} / "
        f"{c['queue_mean_calibrated_score']} |",
        f"| Brier: raw / calibrated / constant base rate | {c['brier_raw']} / {c['brier_calibrated']} / "
        f"{c['brier_constant_rate']} |",
        f"| Expected cost: one global threshold / per-segment thresholds | "
        f"${c['cost_global_threshold']:,.0f} / ${c['cost_segment_thresholds']:,.0f} |",
        f"| Deployed policy (calibrated score, segment thresholds): flag rate / recall / precision | "
        f"{c['policy_flag_rate']} / {c['policy_recall']} / {c['policy_precision']} |",
        f"| Ranking by raw score at the alert budget: recall / precision | "
        f"{c['raw_topk_recall']} / {c['raw_topk_precision']} |",
        f"| Distinct score values: raw / calibrated (share of calibrated scores exactly 0) | "
        f"{c['raw_distinct_values']} / {c['calibrated_distinct_values']} ({c['calibrated_zero_share']}) |",
        f"| Segments with enough labels to tune (default threshold {c['default_threshold']}) | "
        f"{', '.join(f'{k}: {v}' for k, v in c['segments_tuned'].items()) or 'none'} |",
        f"| Mule recall, static model: first / second half of the ring's life | "
        f"{g['mule_recall_first_half']} / {g['mule_recall_second_half']} |",
        "",
        "### Online scoring",
        "",
        "| Setting | Escalation rate | Jev skipped | Path | p50 ms | p99 ms |",
        "|---|---:|---|---|---:|---:|",
    ]
    for name, o in a["online"].items():
        for path, v in o["latency"].items():
            out.append(
                f"| {name} | {o['escalation_rate']} | {o['skipped'] or '-'} | {path} | {v['p50_ms']} | "
                f"{v['p99_ms']} |"
            )
    out += [
        "",
        f"Advanced featurizer replay: **{a['replay_us_per_txn']} µs/transaction** (pure Python, all feature "
        "groups, includes state updates).",
        "",
        "### Feature-group share of total gain (advanced hybrid)",
        "",
        "Gain share measures how much a group is *used*, not how much it *helps*; see the ablation tables for that.",
        "",
        "| Group | Gain share |",
        "|---|---:|",
    ]
    for k, v in sorted(a["feature_gain"].items(), key=lambda kv: -(kv[1] or 0)):
        out.append(f"| {k} | {v} |")
    out.append("")
    return out
