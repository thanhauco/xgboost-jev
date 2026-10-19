"""Scenes for the advanced fraud stack. Numbers come from docs/benchmarks.json; anything
drawn only to illustrate a mechanism is labelled "illustrative" on screen."""

from __future__ import annotations

import math
import random

from motion import (
    BAD,
    BG,
    EMB,
    GRID,
    JEV,
    META,
    MUTED,
    OK,
    PANEL,
    TEXT,
    WARN,
    XGB,
    Canvas,
    W,
    caption,
    clamp,
    header,
    lerp,
    mix,
    seg,
)
from scenes import action_chip, bget

ADV = "fraud_advanced"


def _row(path_prefix: str, row: str, key: str, default):
    """Benchmark lookup for a table row whose name contains characters bget would split on."""
    from scenes import B

    cur = B
    for part in path_prefix.split("."):
        cur = cur.get(part, {}) if isinstance(cur, dict) else {}
    cur = cur.get(row, {}) if isinstance(cur, dict) else {}
    v = cur.get(key, default) if isinstance(cur, dict) else default
    return default if v is None else v


def node(c, pos, label, color, k=1.0, r=15, fill=None, shape="circle", size=11):
    x, y = pos
    if shape == "circle":
        c.circle(x, y, r, fill=fill or PANEL, outline=color, width=2, k=k)
    else:
        c.rect(
            x - r * 1.9,
            y - r * 0.85,
            r * 3.8,
            r * 1.7,
            fill=fill or PANEL,
            outline=color,
            width=2,
            r=7,
            k=k,
        )
    c.text(x, y, label, size, TEXT, "bold", "mm", k=k)


# ============================================================================ 8. entity graph
def graph_ring(c: Canvas, t: float):
    header(
        c,
        t,
        "Entity graph: rings, mules and reputation",
        "Union-find components with fraud flags; fan-in exposes money-mule payees",
        "xgboost-jev / fraud",
    )
    if t >= 17.6:
        _graph_takeaway(c, t)
        return
    payers = [(84, 170 + i * 44) for i in range(6)]
    M, D, M2 = (300, 300), (450, 205), (450, 395)
    hub = (578, 392)

    for i, p in enumerate(payers):
        node(c, p, f"u{i + 1}", MUTED, seg(t, 0.4 + 0.12 * i, 0.9 + 0.12 * i), r=14)
    flagged = t > 12.0
    mule_col = BAD if flagged else WARN
    node(
        c,
        M,
        "payee M",
        mule_col,
        seg(t, 1.0, 1.5),
        r=15,
        shape="rect",
        fill=mix(BAD, 0.28) if flagged else None,
    )
    c.text(M[0], M[1] + 32, "opened 2 days ago", 11, MUTED, anchor="mm", k=seg(t, 1.3, 1.8))

    # payments into M -> fan-in counter
    arrived = 0
    for i, p in enumerate(payers):
        a, b = 1.9 + 0.55 * i, 2.4 + 0.55 * i
        prog = seg(t, a, b)
        c.arrow(
            (p[0] + 16, p[1]),
            (M[0] - 34, M[1] - 4 + (i - 2.5) * 4),
            mix(WARN, 0.8),
            prog,
            2,
            k=0.9,
            head=7,
        )
        if prog >= 1:
            arrived += 1
    # panel
    c.rect(640, 126, 290, 364, fill=PANEL, r=14, k=seg(t, 0.8, 1.3))
    kp = seg(t, 1.6, 2.1)
    c.text(658, 140, "PAYEE FEATURES FOR THE NEXT PAYMENT", 11, MUTED, "bold", k=kp)
    c.text(658, 166, "payee_fan_in_7d", 13, TEXT, "mono", k=kp)
    c.text(912, 166, f"{arrived}", 18, WARN if arrived >= 4 else TEXT, "bold", "ra", k=kp)
    c.text(658, 192, "payee_age_days", 13, TEXT, "mono", k=kp)
    c.text(912, 192, "2.1", 18, WARN, "bold", "ra", k=kp)
    c.text(658, 218, "payee_new_for_user", 13, TEXT, "mono", k=kp)
    c.text(912, 218, "1", 18, WARN, "bold", "ra", k=kp)

    # mule vs legit medians (benchmark)
    kc = seg(t, 5.6, 6.2) * (1 - seg(t, 12.6, 13.0))
    if kc > 0:
        c.text(658, 250, "FAN-IN ALONE IS NOT ENOUGH", 11, MUTED, "bold", k=kc)
        c.text(658, 268, "median on the evaluation period", 11, MUTED, k=kc)
        c.text(836, 290, "mule", 12, BAD, "bold", "ma", k=kc)
        c.text(900, 290, "legit P2P", 12, OK, "bold", "ma", k=kc)
        for i, (lab, key) in enumerate(
            [
                ("payee age (days)", "payee_age_days"),
                ("new for payer", "payee_new_for_user"),
                ("fan-in 7d", "payee_fan_in_7d"),
            ]
        ):
            mv = bget(f"{ADV}.mule_vs_legit_p2p.{key}.mule_median", None)
            lv = bget(f"{ADV}.mule_vs_legit_p2p.{key}.legit_p2p_median", None)
            y = 312 + i * 24
            c.text(658, y, lab, 12, TEXT, k=kc)
            c.text(
                836,
                y,
                "-" if mv is None else f"{mv:g}",
                14,
                BAD,
                "bold",
                "ma",
                k=kc * seg(t, 6.0 + 0.3 * i, 6.5 + 0.3 * i),
            )
            c.text(
                900,
                y,
                "-" if lv is None else f"{lv:g}",
                14,
                OK,
                "bold",
                "ma",
                k=kc * seg(t, 6.0 + 0.3 * i, 6.5 + 0.3 * i),
            )
        c.text(
            658, 392, "Legit payees (landlords, utilities) also", 11, MUTED, k=kc * seg(t, 7.0, 7.6)
        )
        c.text(
            658,
            408,
            "have high fan-in: age and novelty separate",
            11,
            MUTED,
            k=kc * seg(t, 7.0, 7.6),
        )
        c.text(
            658, 424, "them. The model sees all of it together.", 11, MUTED, k=kc * seg(t, 7.0, 7.6)
        )

    # operator infrastructure -> component
    kd = seg(t, 7.6, 8.4)
    node(c, D, "dev X", META, kd, r=15)
    node(
        c,
        M2,
        "payee M2",
        WARN if not flagged else BAD,
        kd,
        r=15,
        shape="rect",
        fill=mix(BAD, 0.28) if flagged else None,
    )
    c.line([(M[0] + 30, M[1] - 10), (D[0] - 14, D[1] + 8)], META, 2.5, k=seg(t, 8.0, 8.8))
    c.line([(M2[0] - 34, M2[1] - 8), (D[0] - 4, D[1] + 16)], META, 2.5, k=seg(t, 8.4, 9.2))
    kh = seg(t, 9.0, 9.6)
    hull_col = BAD if flagged else META
    c.rect(238, 168, 296, 266, fill=None, outline=hull_col, width=2, r=22, k=kh * 0.9)
    c.text(
        250,
        178,
        "component: size 3" if not flagged else "component: 1 confirmed fraud",
        12,
        hull_col,
        "bold",
        k=kh,
    )
    c.text(250, 196, "devices and logins link accounts", 11, MUTED, k=kh)

    # hub protection
    kn = seg(t, 9.8, 10.4) * (1 - seg(t, 12.6, 13.0))
    if kn > 0:
        c.circle(*hub, 12, fill=PANEL, outline=MUTED, width=2, k=kn)
        c.text(hub[0], hub[1], "NAT", 10, MUTED, "bold", "mm", k=kn)
        for j in range(14):
            ang = j / 14 * math.tau
            p = (hub[0] + 44 * math.cos(ang), hub[1] + 44 * math.sin(ang))
            c.line(
                [(hub[0] + 12 * math.cos(ang), hub[1] + 12 * math.sin(ang)), p],
                GRID,
                1.2,
                k=kn * seg(t, 10.0 + j * 0.03, 10.6),
            )
            c.circle(*p, 3.5, fill=MUTED, k=kn * seg(t, 10.0 + j * 0.03, 10.6))
        c.text(hub[0], hub[1] + 58, "NAT IP, degree > 25", 11, MUTED, anchor="mm", k=kn)
        c.text(hub[0], hub[1] + 74, "hub: not merged", 11, WARN, "bold", "mm", k=kn)

    # chargeback label arrives
    if 11.0 < t < 12.4:
        pos = lerp((640, 150), (M[0] + 8, M[1] - 24), seg(t, 11.0, 12.0))
        c.chip(pos[0], pos[1], "report (day +4)", BAD, k=1.0, size=11, anchor="mm")
    if flagged:
        c.text(M[0], M[1] - 34, "confirmed fraud", 11, BAD, "bold", "mm", k=seg(t, 12.0, 12.6))

    # next payment
    k7 = seg(t, 13.2, 13.8)
    p7 = (84, 170 + 6 * 44 - 6)
    node(c, p7, "u7", TEXT, k7, r=14)
    c.arrow(
        (p7[0] + 16, p7[1] - 6), (M[0] - 34, M[1] + 8), TEXT, seg(t, 13.6, 14.6), 2.5, k=k7, head=8
    )
    kg = seg(t, 14.4, 15.0)
    c.text(658, 244, "GRAPH FEATURES", 11, BAD, "bold", k=kg * (1 - kc))
    for i, (lab, val) in enumerate(
        [("g_payee_fraud", "1"), ("g_payee_comp_fraud", "1"), ("g_payee_comp_size", "3")]
    ):
        c.text(658, 270 + i * 26, lab, 13, TEXT, "mono", k=kg * (1 - kc))
        c.text(912, 270 + i * 26, val, 17, BAD, "bold", "ra", k=kg * (1 - kc))
    sc = lerp(0.04, 0.9, seg(t, 14.6, 15.8))
    kb = seg(t, 14.4, 15.0) * (1 - seg(t, 17.2, 17.8))
    c.bar(60, 466, 330, 12, sc, BAD if sc > 0.5 else WARN, k=kb, label=None)
    c.text(60, 450, "score of the next payment to M", 12, MUTED, "bold", k=kb)
    c.text(400, 472, f"{sc:.2f}", 16, TEXT, "bold", "lm", k=kb)
    if t > 15.8:
        action_chip(c, 520, 470, "DECLINE", seg(t, 15.8, 16.3) * (1 - seg(t, 17.2, 17.8)))

    caption(
        c,
        t,
        [
            (0.5, 1.8, "A new payee account starts receiving transfers"),
            (1.8, 5.6, "Fan-in: how many distinct senders paid this payee in the last 7 days"),
            (5.6, 7.6, "Legit payees also have high fan-in; age and novelty tell them apart"),
            (7.6, 9.8, "Shared devices link the operator's accounts into one component"),
            (
                9.8,
                11.0,
                "Hub entities (shared NAT IPs) are skipped so they cannot glue the whole customer base together",
            ),
            (
                11.0,
                13.0,
                "A report arrives: fraud flags propagate to the whole component - no retrain needed",
            ),
            (13.0, 17.2, "The next payer to this payee carries the reputation as features"),
            (
                17.8,
                21.0,
                "Honest read: helps known patterns; limited help against a brand-new ring",
            ),
        ],
    )


def _graph_takeaway(c: Canvas, t: float):
    kt = seg(t, 17.6, 18.2)
    if kt > 0:
        c.rect(120, 150, 720, 300, fill=PANEL, r=16, k=kt)
        full = _row(f"{ADV}.static", "advanced hybrid (+Jev)", "pr_auc", 0.0)
        nog = _row(f"{ADV}.group_ablation", "minus entity graph / payee", "pr_auc", 0.0)
        sd_full = _row(f"{ADV}.static", "advanced hybrid (+Jev)", "pr_auc_sd", 0.0)
        sd_nog = _row(f"{ADV}.group_ablation", "minus entity graph / payee", "pr_auc_sd", 0.0)
        mule = _row(f"{ADV}.static", "advanced hybrid (+Jev)", "per_scenario", {}).get("mule_ring")
        mule_wk = _row(f"{ADV}.continual", "weekly, bootstrap gate", "per_scenario", {}).get(
            "mule_ring"
        )
        c.text(
            W / 2, 176, "What the benchmark says (mean over 3 seeds)", 15, TEXT, "bold", "mm", k=kt
        )
        for i, (lab, v, col) in enumerate(
            [("PR-AUC, all features", full, XGB), ("PR-AUC, graph features removed", nog, MUTED)]
        ):
            vv = v * seg(t, 18.0 + 0.3 * i, 19.0 + 0.3 * i)
            c.text(170, 222 + i * 46, lab, 14, TEXT, k=kt)
            c.bar(430, 226 + i * 46, 300, 14, vv / max(full, nog, 0.01), col, k=kt)
            c.text(745, 233 + i * 46, f"{vv:.3f}", 15, col, "bold", "lm", k=kt)
        c.text(
            W / 2,
            330,
            f"Spread across seeds: +/-{sd_full:.2f} (all features) vs +/-{sd_nog:.2f} (no graph). Indicative, not conclusive.",
            14,
            WARN,
            "bold",
            "mm",
            k=kt * seg(t, 19.2, 19.8),
        )
        c.text(
            W / 2,
            352,
            "Reputation helps little against a ring the model has never seen:",
            14,
            WARN,
            "bold",
            "mm",
            k=kt * seg(t, 19.2, 19.8),
        )
        if mule is not None:
            c.text(
                W / 2,
                392,
                f"mule-ring recall at the alert budget: {mule:.2f} static, {mule_wk:.2f} with weekly retraining",
                16,
                TEXT,
                "bold",
                "mm",
                k=kt * seg(t, 20.0, 20.6),
            )
    caption(
        c,
        t,
        [
            (
                17.8,
                21.0,
                "Honest read: helps known patterns; limited help against a brand-new ring",
            )
        ],
    )


# ============================================================================ 9. delayed labels
def _timeline_events():
    rng = random.Random(4)
    ev = []  # (day, kind, group, label_day)
    for g, (d0, d1, n, delay) in {
        "ring": (8, 34, 9, (20, 42)),
        "scam": (22, 34, 3, (2, 7)),
        "ato": (14, 33, 3, (3, 9)),
    }.items():
        for _ in range(n):
            d = rng.uniform(d0, d1)
            ev.append((d, "fraud", g, d + rng.uniform(*delay)))
    ev.sort()
    return ev


TL = _timeline_events()
CUT, X0, X1 = 35, 70, 890


def _x(day: float) -> float:
    return lerp(X0, X1, day / 60)


def delayed_labels(c: Canvas, t: float):
    header(
        c,
        t,
        "Delayed labels: what the model can actually learn from",
        "Training at the cut sees only the labels that have arrived - and must not mistake silence for innocence",
        "xgboost-jev / fraud",
    )
    if t >= 17.4:
        _labels_takeaway(c, t)
        return
    ya, yl = 330, 200
    k0 = seg(t, 0.4, 1.0)
    c.line([(X0, ya), (X1, ya)], MUTED, 2, k=k0)
    for d in range(0, 61, 10):
        c.text(_x(d), ya + 14, f"day {d}", 11, MUTED, anchor="mt", k=k0)
    c.text(X0, yl - 30, "label arrives", 11, MUTED, "bold", k=k0)
    c.line([(X0, yl), (X1, yl)], GRID, 1.5, k=k0)
    rng = random.Random(9)
    for _ in range(150):  # legit background
        d = rng.uniform(0, 60)
        if seg(t, 1.0, 2.4) * 60 > d:
            c.circle(_x(d), ya - rng.uniform(4, 40), 2.3, fill=MUTED, k=0.5)
    cursor = lerp(0, CUT, seg(t, 3.4, 7.0))
    known = 0
    poisoned = 0
    pos = {}
    for i, (d, _kind, g, ld) in enumerate(TL):
        shown = seg(t, 1.6 + i * 0.1, 2.0 + i * 0.1)
        x, y = _x(d), ya - 22
        pos[i] = (x, y)
        arrived = ld <= cursor and ld <= CUT
        if t > 7.0 and ld > CUT:
            is_prop = g == "ring" and t > 11.0
            c.circle(x, y, 8, fill=mix(BAD, 0.55) if is_prop else BG, outline=BAD, width=2, k=shown)
            if not is_prop:
                c.text(x, y, "?", 11, BAD, "bold", "mm", k=shown)
            poisoned += 0 if is_prop else 1
        elif arrived or (t > 7.0 and ld <= CUT):
            c.circle(x, y, 8, fill=BAD, outline=BAD, k=shown)
            known += 1
        else:
            c.circle(x, y, 8, fill=BG, outline=BAD, width=2, k=shown)
        if arrived and ld <= CUT and shown > 0.9:
            c.arrow(
                (x, y - 10),
                (_x(ld), yl + 4),
                mix(BAD, 0.6),
                seg(t, 3.4, 7.0) * 0 + 1.0,
                1.4,
                k=0.6,
                head=5,
            )
            c.circle(_x(ld), yl, 4, fill=BAD, k=0.9)
    # cut
    kcut = seg(t, 3.0, 3.5)
    xc = _x(CUT)
    c.line([(xc, 150), (xc, 372)], WARN, 2.5, k=kcut)
    c.text(xc, 140, "training cut", 12, WARN, "bold", "mm", k=kcut)
    c.rect(xc, 150, X1 - xc, 222, fill=BG, r=0, k=kcut * 0.0)
    cx = _x(cursor)
    if 3.4 < t < 7.0:
        c.line([(cx, 160), (cx, 360)], TEXT, 1.5, k=0.5)
    lh = {
        "known": bget(f"{ADV}.label_handling.labels_known_at_cut", None),
        "fraud": bget(f"{ADV}.label_handling.fraud_before_cut", None),
        "prop": bget(f"{ADV}.label_handling.propagated_rows", None),
        "trf": bget(f"{ADV}.label_handling.train_fraud_rows", None),
    }
    ks = seg(t, 7.0, 7.6)
    if ks > 0:
        c.chip(250, 112 + 6, "unlabelled fraud is trained as LEGIT", BAD, k=ks, size=13)
    if lh["known"] is not None:
        c.text(
            36,
            410,
            f"benchmark world: {lh['known']} of {lh['fraud']} pre-cut frauds were labelled at the cut",
            14,
            TEXT,
            "bold",
            k=seg(t, 7.4, 8.0),
        )
    # maturity
    km = seg(t, 8.2, 8.8) * (1 - seg(t, 17.0, 17.6))
    if km > 0:
        xm = _x(CUT - 7)
        c.rect(xm, 262, xc - xm, 90, fill=mix(META, 0.18), outline=META, width=1.5, r=6, k=km)
        for j in range(5):
            c.line(
                [(xm + j * (xc - xm) / 5, 352), (xm + (j + 1) * (xc - xm) / 5, 262)],
                META,
                1,
                k=km * 0.6,
            )
        c.text((xm + xc) / 2, 252, "last 7 days dropped", 11, META, "bold", "mm", k=km)
        c.text((xm + xc) / 2, 238, "(fraud AND legit)", 11, META, "bold", "mm", k=km)
    # propagation
    if t > 9.6:
        ring = [i for i, e in enumerate(TL) if e[2] == "ring"]
        anchor = next((i for i in ring if TL[i][3] <= CUT), None)
        if anchor is not None:
            ax, ay = pos[anchor]
            for n, i in enumerate(ring):
                if i == anchor:
                    continue
                px, py = pos[i]
                kk = seg(t, 9.8 + 0.12 * n, 10.4 + 0.12 * n) * (1 - seg(t, 17.0, 17.6))
                c.line(
                    [(ax, ay - 12), ((ax + px) / 2, ay - 52), (px, py - 12)], META, 1.6, k=kk * 0.8
                )
            c.text(
                _x(20),
                156,
                "same device: confirmed -> weak positives",
                12,
                META,
                "bold",
                "mm",
                k=seg(t, 9.6, 10.2) * (1 - seg(t, 17.0, 17.6)),
            )
            if lh["prop"] is not None:
                c.text(
                    36,
                    432,
                    f"propagated to {lh['prop']} training rows (fraud rows in training: {lh['trf']})",
                    13,
                    META,
                    "bold",
                    k=seg(t, 11.0, 11.6) * (1 - seg(t, 17.0, 17.6)),
                )
    caption(
        c,
        t,
        [
            (0.5, 3.0, "Frauds happen on the timeline; their labels arrive days to weeks later"),
            (3.0, 7.0, "At the cut, only the labels that have already arrived are known"),
            (
                7.0,
                8.2,
                "Everything else looks legitimate - training would learn that fraud is fine",
            ),
            (8.2, 9.6, "Recent rows are dropped symmetrically: fraud AND legit, until they mature"),
            (
                9.6,
                17.0,
                "Confirmed fraud flows to rows that share the attacker's device, IP or payee",
            ),
            (17.6, 21.0, "Safeguards are measured, not assumed"),
        ],
    )


def _labels_takeaway(c: Canvas, t: float):
    kb = seg(t, 17.6, 18.2)
    if kb > 0:
        c.rect(110, 128, 740, 336, fill=PANEL, r=16, k=kb)
        c.text(
            W / 2,
            154,
            "PR-AUC when training only on labels known at the cut (mean over seeds)",
            14,
            TEXT,
            "bold",
            "mm",
            k=kb,
        )
        rows = [
            ("without label propagation", "without entity label propagation", MUTED),
            (
                "default: as-of labels + propagation",
                "default (as-of labels, propagation, monotone)",
                XGB,
            ),
            ("oracle labels (not achievable)", "oracle labels (upper bound, not achievable)", OK),
        ]
        vals = [_row(f"{ADV}.ablations", key, "pr_auc", 0.0) for _lab, key, _c in rows]
        top = max(vals + [0.01])
        for i, ((lab, _key, col), v) in enumerate(zip(rows, vals, strict=True)):
            vv = v * seg(t, 18.0 + 0.4 * i, 19.0 + 0.4 * i)
            c.text(140, 212 + i * 62, lab, 14, TEXT, k=kb)
            c.bar(140, 236 + i * 62, 520, 16, vv / top, col, k=kb)
            c.text(675, 245 + i * 62, f"{vv:.3f}", 17, col, "bold", "lm", k=kb)
        c.text(
            W / 2,
            418,
            "Entity-linked label propagation is the most valuable safeguard in this benchmark.",
            13,
            MUTED,
            "bold",
            "mm",
            k=kb * seg(t, 20.0, 20.6),
        )
    caption(c, t, [(17.8, 21.0, "Safeguards are measured, not assumed")])


# ============================================================================ 10. lifecycle
def lifecycle(c: Canvas, t: float):
    header(
        c,
        t,
        "Model lifecycle: weekly retrain, shadow test, gated promotion",
        "Fraudsters adapt in days; chargebacks arrive in weeks",
        "xgboost-jev / fraud",
    )
    from scenes import B

    hist = (B.get(ADV, {}).get("history", {}) or {}).get("weekly, bootstrap gate", [])
    weeks = [
        (f"day {h['day']:g}", h["p_challenger_worse"], h["shadow_frauds"], h["promoted"])
        for h in hist
    ] or [("day 42", 0.81, 6, True), ("day 49", 1.0, 11, False)]
    if t >= 12.2:
        _lifecycle_late(c, t)
        return
    # illustrative decision flow for one retrain
    kA = seg(t, 0.4, 1.0)
    c.rect(36, 130, 270, 190, fill=PANEL, outline=BAD, width=0, r=14, k=kA)
    c.text(52, 146, "1  LABELS KNOWN NOW", 12, BAD, "bold", k=kA)
    c.text(52, 170, "chargebacks, reports,", 13, TEXT, k=kA)
    c.text(52, 188, "analyst verdicts", 13, TEXT, k=kA)
    c.text(52, 218, "propagate over device / IP / payee", 12, MUTED, k=kA)
    c.text(52, 238, "7-day maturity, as-of cut", 12, MUTED, k=kA)
    c.arrow((308, 225), (338, 225), MUTED, seg(t, 1.2, 1.7), k=kA)
    kB = seg(t, 1.5, 2.1)
    c.rect(340, 130, 270, 190, fill=PANEL, outline=XGB, width=0, r=14, k=kB)
    c.text(356, 146, "2  CHALLENGER", 12, XGB, "bold", k=kB)
    c.text(356, 170, "cold retrain, or warm-start from", 13, TEXT, k=kB)
    c.text(356, 190, "the champion (+120 rounds)", 13, TEXT, k=kB)
    c.text(356, 220, "trained on data older than the", 12, MUTED, k=kB)
    c.text(356, 238, "last 5 days (held out for shadow)", 12, MUTED, k=kB)
    c.arrow((612, 225), (642, 225), MUTED, seg(t, 2.3, 2.8), k=kB)
    kC = seg(t, 2.6, 3.2)
    c.rect(644, 130, 286, 190, fill=PANEL, outline=OK, width=0, r=14, k=kC)
    c.text(660, 146, "3  SHADOW TEST", 12, OK, "bold", k=kC)
    c.text(660, 170, "score the last 5 days with both", 13, TEXT, k=kC)
    c.text(660, 190, "bootstrap P(challenger worse)", 13, TEXT, k=kC)

    # bootstrap dots (illustrative)
    kd = seg(t, 3.6, 4.2)
    rng = random.Random(3)
    c.text(660, 222, "illustrative bootstrap draws:", 11, MUTED, k=kd)
    c.line([(668, 292), (918, 292)], MUTED, 1.5, k=kd)
    c.line([(793, 238), (793, 296)], WARN, 1.5, k=kd)
    for i in range(60):
        v = rng.gauss(-0.2 if t < 8 else 0.25, 0.45)
        x = 793 + clamp(v / 1.3, -1, 1) * 120
        if seg(t, 4.0, 6.0) * 60 > i:
            col = BAD if x < 793 else OK
            c.circle(x, 288 - (i % 7) * 7, 3.2, fill=col, k=kd * 0.9)
    c.text(668, 306, "worse", 11, BAD, "bold", k=kd)
    c.text(918, 306, "better", 11, OK, "bold", "ra", k=kd)

    # decision rule + weekly results
    k2 = seg(t, 6.6, 7.2)
    c.text(36, 346, "Gate:", 14, TEXT, "bold", k=k2)
    c.text(92, 346, "promote unless P(worse) >= 0.9", 14, TEXT, k=k2)
    c.text(
        36,
        368,
        f"A naive 'must beat the champion' test is noise with {min(w[2] for w in weeks)}-{max(w[2] for w in weeks)} known frauds in the window.",
        12,
        MUTED,
        k=k2,
    )
    for i, (day, pw, nf, promoted) in enumerate(weeks):
        x = 52 + i * 218
        kk = seg(t, 7.4 + i * 0.7, 8.0 + i * 0.7)
        c.rect(x - 12, 392, 204, 74, fill=PANEL, r=10, k=kk)
        c.text(x, 402, day, 12, MUTED, "bold", k=kk)
        c.text(x, 420, f"{nf} known frauds in shadow window", 11, MUTED, k=kk)
        c.text(x, 438, f"P(worse) = {pw:.2f}", 13, TEXT, k=kk)
        c.text(x, 454, "benchmark, seed 21", 10, MUTED, k=kk)
        if not promoted:
            c.chip(x + 146, 408, "KEEP", WARN, k=kk, size=11, anchor="mm")
        else:
            c.chip(x + 146, 408, "PROMOTE", OK, k=kk, size=11, anchor="mm")
    caption(
        c,
        t,
        [
            (
                0.4,
                3.0,
                "Every week: retrain on the labels known now (cold, or warm-started from the champion)",
            ),
            (
                3.0,
                6.6,
                "Shadow test: how likely is the challenger worse on the most recent labelled days?",
            ),
            (6.6, 12.4, "Only block a promotion when the evidence says it is worse"),
            (
                12.4,
                15.6,
                "Benchmark: weekly retraining scores about the same as training once; gates are within noise",
            ),
            (
                15.6,
                19.0,
                "A small slice of traffic runs on the challenger; recall is watched per fraud type",
            ),
        ],
    )


def _lifecycle_late(c: Canvas, t: float):
    kt = seg(t, 12.4, 13.0) * (1 - seg(t, 15.0, 15.5))
    if kt > 0:
        c.rect(100, 128, 760, 340, fill=PANEL, r=16, k=kt)
        c.text(
            W / 2,
            152,
            "Weekly retraining vs one-off training (mean over seeds, benchmark world)",
            14,
            TEXT,
            "bold",
            "mm",
            k=kt,
        )
        rows = [
            ("static, trained once at the cut", "static (trained once at cut)", MUTED),
            ("weekly, naive gate", "weekly, naive gate (must beat champion)", WARN),
            ("weekly, bootstrap gate", "weekly, bootstrap gate", XGB),
            ("weekly, cold retrain", "weekly, cold retrain (no warm start)", EMB),
        ]
        vals = [_row(f"{ADV}.continual", key, "pr_auc", 0.0) for _l, key, _c in rows]
        proms = [_row(f"{ADV}.continual", key, "promotions", None) for _l, key, _c in rows]
        top = max(vals + [0.01])
        c.text(160, 186, "PR-AUC", 11, MUTED, "bold", k=kt)
        c.text(840, 186, "promotions", 11, MUTED, "bold", "ra", k=kt)
        for i, ((lab, _key, col), v, pr) in enumerate(zip(rows, vals, proms, strict=True)):
            vv = v * seg(t, 12.8 + 0.3 * i, 13.8 + 0.3 * i)
            y = 214 + i * 46
            c.text(130, y, lab, 14, TEXT, k=kt)
            c.bar(380, y + 4, 300, 14, vv / top, col, k=kt)
            c.text(692, y + 11, f"{vv:.3f}", 15, col, "bold", "lm", k=kt)
            if pr is not None:
                c.text(840, y + 11, f"{pr:g}", 15, TEXT, "bold", "rm", k=kt)
        c.text(
            W / 2,
            410,
            "Results are noisy: a few dozen known frauds per window. Treat gates as safety rails,",
            12,
            MUTED,
            "bold",
            "mm",
            k=kt * seg(t, 14.4, 15.0),
        )
        c.text(
            W / 2,
            428,
            "not as proof that a retrain helped.",
            12,
            MUTED,
            "bold",
            "mm",
            k=kt * seg(t, 14.4, 15.0),
        )
    # champion/challenger + recall monitor
    ke = seg(t, 15.6, 16.2)
    if ke > 0:
        c.rect(36, 128, 888, 340, fill=BG, r=0, k=0)
        c.rect(100, 128, 760, 340, fill=PANEL, r=16, k=ke)
        c.text(
            W / 2, 152, "Champion / challenger and recall monitoring", 14, TEXT, "bold", "mm", k=ke
        )
        c.text(130, 192, "traffic split by hash of transaction id", 12, MUTED, "bold", k=ke)
        c.rect(130, 208, 640, 26, fill=mix(XGB, 0.35), r=8, k=ke)
        c.rect(130 + 640 * 0.95, 208, 640 * 0.05, 26, fill=EMB, r=6, k=ke)
        c.text(450, 221, "champion 95%", 12, TEXT, "bold", "mm", k=ke)
        c.text(130 + 640 * 0.975, 246, "challenger 5%", 12, EMB, "bold", "ma", k=ke)
        c.text(
            130,
            282,
            "RecallMonitor: rolling recall per fraud type, computed as labels arrive (illustrative)",
            12,
            MUTED,
            "bold",
            k=ke,
        )
        for i, (nm, base, now) in enumerate(
            [("card_testing", 0.90, 0.88), ("app_scam", 0.80, 0.78), ("mule_ring", 0.60, 0.25)]
        ):
            y = 310 + i * 40
            v = lerp(base, now, seg(t, 16.6, 18.0))
            c.text(130, y, nm, 13, TEXT, "mono", k=ke)
            c.bar(280, y + 2, 360, 12, v, BAD if (base - v) > 0.25 else OK, k=ke)
            c.text(652, y + 8, f"{v:.2f}", 14, TEXT, "bold", "lm", k=ke)
            if (base - v) > 0.25 and t > 17.8:
                c.chip(770, y + 8, "ALERT", BAD, k=seg(t, 17.8, 18.2), size=12, anchor="mm")
        c.text(
            W / 2,
            440,
            "A drop versus a type's own baseline is usually the first sign attackers adapted.",
            12,
            MUTED,
            "bold",
            "mm",
            k=ke * seg(t, 18.2, 18.8),
        )
    caption(
        c,
        t,
        [
            (
                12.4,
                15.5,
                "Benchmark: weekly retraining scores about the same as training once; gates are within noise",
            ),
            (
                15.6,
                19.0,
                "A small slice of traffic runs on the challenger; recall is watched per fraud type",
            ),
        ],
    )


# ============================================================================ 11. counters + budget
def counters_budget(c: Canvas, t: float):
    header(
        c,
        t,
        "Decayed counters and the latency budget",
        "Constant-size state per entity; skip Jev when the request cannot afford it",
        "xgboost-jev / fraud",
    )
    # left: deque vs decayed
    ka = seg(t, 0.4, 1.0)
    c.rect(36, 126, 440, 340, fill=PANEL, r=14, k=ka)
    c.text(54, 140, "STATE PER CARD", 11, MUTED, "bold", k=ka)
    events = [0.5, 1.1, 1.3, 2.4, 3.0, 3.1, 3.2, 4.4, 5.2, 5.3, 6.1, 6.6]
    cur = seg(t, 1.4, 8.0) * 7.2
    c.line([(60, 215), (456, 215)], MUTED, 1.5, k=ka)
    c.text(60, 224, "events over time", 11, MUTED, k=ka)
    n = 0
    for e in events:
        if e <= cur:
            n += 1
            c.circle(lerp(60, 456, e / 7.2), 215, 5, fill=META, k=ka)
    c.text(54, 250, "deque of events", 13, TEXT, "bold", k=ka)
    c.bar(54, 262, 380, 12, clamp(n / 12), BAD, k=ka)
    c.text(54, 282, f"{n} items kept (grows with activity; capped at 2,000)", 11, MUTED, k=ka)
    c.text(54, 312, "decayed counter", 13, TEXT, "bold", k=ka)
    c.bar(54, 324, 380, 12, 0.12, OK, k=ka)
    c.text(
        54,
        344,
        "3 floats per half-life, however busy: value = value x 2^(-dt/h) + x",
        11,
        MUTED,
        k=ka,
    )
    # decay curve
    kc = seg(t, 8.0, 8.6)
    if kc > 0:
        pts = []
        for i in range(60):
            xx = 60 + i * (396 / 59)
            yy = 440 - 70 * (0.5 ** (i / 20))
            pts.append((xx, yy))
        c.line(pts[: int(seg(t, 8.2, 10.0) * 59) + 2], XGB, 2.5, k=kc)
        for j, lab in enumerate(["h", "2h"]):
            c.line(
                [(60 + (20 + 20 * j) * (396 / 59), 366), (60 + (20 + 20 * j) * (396 / 59), 440)],
                GRID,
                1,
                k=kc,
            )
            c.text(60 + (20 + 20 * j) * (396 / 59), 358, lab, 11, MUTED, anchor="mm", k=kc)
        c.text(60, 372, "weight halves every half-life (1h, 1d, 7d, 30d)", 11, MUTED, k=kc)
    # right: latency budget
    kr = seg(t, 5.0, 5.6)
    c.rect(494, 126, 430, 340, fill=PANEL, r=14, k=kr)
    c.text(512, 140, "REQUEST BUDGET", 11, MUTED, "bold", k=kr)
    x0, x1, ms_max = 520, 900, 60.0

    def mx(ms):
        return lerp(x0, x1, ms / ms_max)

    for ms in (0, 15, 30, 45, 60):
        c.line([(mx(ms), 190), (mx(ms), 200)], MUTED, 1, k=kr)
        c.text(mx(ms), 206, f"{ms}", 11, MUTED, anchor="mt", k=kr)
    c.text(x1, 226, "ms", 11, MUTED, anchor="ra", k=kr)
    online = bget(f"{ADV}.online", {}) or {}
    names = list(online)
    tab = bget(f"{ADV}.online.{names[0]}.latency.tabular.p50_ms", 0.5) if names else 0.5
    hyb = bget(f"{ADV}.online.{names[0]}.latency.hybrid.p50_ms", 27.0) if names else 27.0
    # scenario A: tight budget
    ya = 258
    ba = seg(t, 5.8, 6.4)
    c.text(512, ya - 14, "budget 15 ms", 12, WARN, "bold", k=ba)
    c.line([(mx(15), ya - 4), (mx(15), ya + 28)], WARN, 2.5, k=ba)
    c.rect(mx(0), ya, max(2, mx(tab) - mx(0)), 14, fill=XGB, r=3, k=ba)
    est = hyb
    c.rect(
        mx(tab),
        ya,
        mx(est) - mx(tab),
        14,
        fill=mix(JEV, 0.5),
        outline=JEV,
        width=1,
        r=3,
        k=ba * seg(t, 6.4, 7.4),
    )
    c.text(
        mx(tab) + 6,
        ya + 32,
        f"tabular {tab:.1f} ms + Jev estimate ~{est:.0f} ms > 15 ms",
        11,
        TEXT,
        k=ba * seg(t, 6.6, 7.2),
    )
    if t > 7.4:
        c.chip(760, ya - 14, "SKIP JEV: budget", WARN, k=seg(t, 7.4, 7.8), size=11, anchor="mm")
        c.text(
            512,
            ya + 52,
            "decision comes from the tabular model, flagged as skipped",
            11,
            MUTED,
            k=seg(t, 7.6, 8.0),
        )
    # scenario B: roomy budget
    yb = 352
    bb = seg(t, 9.6, 10.2)
    c.text(512, yb - 14, "budget 50 ms", 12, OK, "bold", k=bb)
    c.line([(mx(50), yb - 4), (mx(50), yb + 28)], OK, 2.5, k=bb)
    c.rect(mx(0), yb, max(2, mx(tab) - mx(0)), 14, fill=XGB, r=3, k=bb)
    c.rect(mx(tab), yb, (mx(est) - mx(tab)) * seg(t, 10.2, 11.4), 14, fill=JEV, r=3, k=bb)
    if t > 11.4:
        c.chip(760, yb - 14, "ESCALATE: hybrid", OK, k=seg(t, 11.4, 11.8), size=11, anchor="mm")
        c.text(
            512,
            yb + 52,
            f"measured p50: tabular {tab:.2f} ms, escalated {hyb:.0f} ms",
            11,
            MUTED,
            k=seg(t, 11.6, 12.0),
        )
    # timeout learning
    kt = seg(t, 12.6, 13.2)
    c.text(
        512,
        438,
        "A Jev timeout raises the latency estimate, so later requests skip on budget",
        11,
        MUTED,
        "bold",
        k=kt,
    )
    esc = bget(f"{ADV}.online.{names[0]}.escalation_rate", None) if names else None
    caption(
        c,
        t,
        [
            (
                0.4,
                5.0,
                "Counters decay continuously, so memory is constant however many events arrive",
            ),
            (5.0, 9.4, "Each request carries a deadline: Jev is only called when it can fit"),
            (9.4, 12.6, "With room to spare, the payment escalates to the hybrid model"),
            (
                12.6,
                15.0,
                "Escalation stays rare"
                + (f" ({esc:.0%} of benchmark traffic)" if esc is not None else ""),
            ),
        ],
    )


ADVANCED_SCENES = {
    "08_entity_graph": (graph_ring, 21.0),
    "09_delayed_labels": (delayed_labels, 21.0),
    "10_model_lifecycle": (lifecycle, 19.5),
    "11_counters_and_budget": (counters_budget, 15.5),
}
