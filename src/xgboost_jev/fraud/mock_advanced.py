"""Harder synthetic world for the advanced fraud stack.

On top of the v1 scenarios, this generator adds what per-transaction velocity
rules miss, plus realistic label timing:

  mule_ring        appears only late in the stream (concept drift). Victims pay
                   a ring of new payee accounts, operated from shared devices,
                   with BENIGN memos -> needs payee fan-in and graph reputation.
  low_and_slow     synthetic accounts on a few shared emulators make small
                   purchases hours apart, never tripping velocity windows.
  ato_warmup       new-device login -> password reset -> new payee -> transfer
                   hours later from a device that is no longer "new".
  app_scam         scam memo and/or coached session text, payee-name mismatch.
  descriptor_spoof look-alike merchant names, with and without leetspeak.
  + card_testing, account_takeover, impossible_travel from v1.

Hard negatives: legit phone changes (new device + password reset + new payee),
high fan-in legit payees (utilities, landlords), legit emulator users.

Labels arrive late: chargebacks 20-60 days, victim reports 1-7 days, ATO 2-10
days; 5% of fraud is never reported. Mock data only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .mock_data import CITIES, MERCHANTS, SCAM_MEMOS, SPOOF_DESCRIPTORS
from .schema import AccountEvent, LabelEvent, Transaction

FIRST = [
    "James",
    "Mary",
    "Robert",
    "Linda",
    "Michael",
    "Sarah",
    "David",
    "Emma",
    "Daniel",
    "Olivia",
    "Kevin",
    "Grace",
    "Thomas",
    "Chloe",
    "Ahmed",
    "Mei",
    "Carlos",
    "Priya",
    "Lucas",
    "Nadia",
]
LAST = ["Smith", "Nguyen", "Garcia", "Brown", "Okafor", "Kim", "Patel", "Rossi", "Muller", "Silva"]
LEGIT_MEMOS = [
    "rent",
    "dinner",
    "thanks!",
    "concert",
    "utilities share",
    "groceries",
    "trip deposit",
    "urgent - rent due tomorrow!",
    "bday gift",
    "bitcoin meetup tickets",
    "",
    "",
    "",
]
UTILITY_PAYEES = [
    ("util_power", "City Power Co"),
    ("util_water", "Metro Water"),
    ("landlord_1", "Parkview Apartments LLC"),
    ("landlord_2", "Oak Street Rentals"),
    ("gym_1", "FitLife Club"),
]
BENIGN_MULE_MEMOS = [
    "invoice 4471",
    "deposit for apartment",
    "rent",
    "payment",
    "car deposit",
    "order #8812",
    "",
]
COACHED = [
    "the bank's fraud team told me to move my savings here to keep them safe",
    "agent on the phone said my account is compromised and to transfer now",
    "they told me not to tell the bank why i am sending this",
]
LOOKALIKE_DESCRIPTORS = SPOOF_DESCRIPTORS + [
    "AMAZON-SECURE-BILLING-HELP",
    "PAYPAL RECOVERY CENTER",
    "APPLE SUPPORT REFUND",
]
LEGIT_DESCRIPTORS = [
    "AMAZON MKTPLACE PMTS",
    "AMAZON.COM",
    "NETFLIX.COM",
    "APPLE.COM/BILL",
    "ETSY.COM",
    "TARGET.COM",
]

DAY = 86400.0
# Legit activity by hour of day (quiet overnight, peaks at lunch and evening).
_HOUR_WEIGHTS = [1, 1, 1, 1, 1, 2, 4, 6, 8, 9, 9, 10, 12, 11, 9, 9, 10, 12, 13, 13, 11, 8, 5, 2]


@dataclass
class _User:
    uid: str
    card: str
    name: str
    home: tuple[float, float]
    devices: list[str]
    ip: str
    mu: float
    rate: float
    favourite: list[tuple[str, str]]
    friends: list[str] = field(default_factory=list)
    payees: set = field(default_factory=set)


@dataclass
class AdvancedStream:
    transactions: list[Transaction]
    account_events: list[AccountEvent]
    label_events: list[LabelEvent]
    labels: dict[str, int]
    scenarios: dict[str, str]
    start_ts: float
    days: int

    def day_of(self, t: Transaction) -> float:
        return (t.timestamp - self.start_ts) / DAY


def generate_advanced_stream(
    n_users: int = 500,
    days: int = 60,
    seed: int = 21,
    start_ts: float = 1_788_000_000.0,
    mule_start_day: float = 38.0,
    fraud_scale: float = 1.0,
) -> AdvancedStream:
    rng = random.Random(seed)
    txns: list[Transaction] = []
    events: list[AccountEvent] = []
    labels: dict[str, int] = {}
    scen: dict[str, str] = {}
    seq = [0]
    horizon = days * DAY

    def jitter(loc, d=0.05):
        return (loc[0] + rng.uniform(-d, d), loc[1] + rng.uniform(-d, d))

    users: list[_User] = []
    for i in range(n_users):
        favs = [(mcc, rng.choice(n)) for mcc, n in rng.sample(list(MERCHANTS.items()), 4)]
        users.append(
            _User(
                f"u{i:05d}",
                f"c{i:05d}",
                f"{rng.choice(FIRST)} {rng.choice(LAST)}",
                rng.choice(CITIES),
                [f"d{i:05d}a"],
                f"10.{i // 250}.{i % 250}.{rng.randint(1, 254)}",
                rng.uniform(2.3, 4.2),
                rng.uniform(1.0, 4.0),
                favs,
            )
        )
    by_id = {u.uid: u for u in users}
    for u in users:
        u.friends = [f.uid for f in rng.sample(users, 4) if f.uid != u.uid]

    def add(u: _User, ts: float, amount: float, label: int, kind: str, **kw) -> Transaction:
        seq[0] += 1
        tid = f"t{seq[0]:08d}"
        mcc, desc = kw.pop("merchant", rng.choice(u.favourite))
        lat, lon = kw.pop("loc", jitter(u.home))
        t = Transaction(
            txn_id=tid,
            timestamp=ts,
            user_id=u.uid,
            card_id=kw.pop("card", u.card),
            amount=round(amount, 2),
            merchant_id=kw.pop("merchant_id", desc.split()[0]),
            mcc=mcc,
            channel=kw.pop("channel", "card_present"),
            device_id=kw.pop("device", rng.choice(u.devices)),
            ip=kw.pop("ip", u.ip),
            lat=lat,
            lon=lon,
            merchant_descriptor=desc,
            memo=kw.pop("memo", ""),
            payee_id=kw.pop("payee_id", None),
            payee_name=kw.pop("payee_name", ""),
            payee_account_name=kw.pop("payee_account_name", ""),
            session_text=kw.pop("session_text", ""),
            is_emulator=kw.pop("is_emulator", False),
            is_rooted=kw.pop("is_rooted", rng.random() < 0.02),
        )
        txns.append(t)
        labels[tid], scen[tid] = label, kind
        return t

    def ev(ts, uid, kind, device=None, ip=None):
        events.append(AccountEvent(ts, uid, kind, device, ip))

    def typed_name(full: str) -> str:
        first, last = full.split()
        return rng.choice([full, first, f"{first[0]} {last}", full.lower(), f"{first} {last[0]}"])

    # ---------------------------------------------------------------- legit life
    for u in users:
        t = start_ts + rng.uniform(0, 6 * 3600)
        next_login = t
        while t < start_ts + horizon:
            t += rng.expovariate(u.rate / DAY)
            hour = rng.choices(range(24), weights=_HOUR_WEIGHTS)[0]
            ts = t - (t % DAY) + 3600 * hour + rng.uniform(0, 3600)
            while next_login < ts:
                ev(next_login, u.uid, "login", rng.choice(u.devices), u.ip)
                next_login += rng.uniform(0.5, 2.0) * DAY
            r = rng.random()
            if r < 0.004:  # legit phone change: the hard negative for ATO scripts
                nd = f"{u.devices[0]}_n{len(u.devices)}"
                ev(ts - 3600, u.uid, "login_new_device", nd, u.ip)
                ev(ts - 3500, u.uid, "password_reset", nd, u.ip)
                u.devices.append(nd)
            if r < 0.08:
                if rng.random() < 0.3:
                    pid, pname = rng.choice(UTILITY_PAYEES)
                    typed, acct = pname.lower(), pname
                else:
                    pid = rng.choice(u.friends)
                    acct = by_id[pid].name
                    typed = typed_name(acct)
                if pid not in u.payees:
                    ev(ts - rng.uniform(60, 3600), u.uid, "new_payee", u.devices[-1], u.ip)
                    u.payees.add(pid)
                add(
                    u,
                    ts,
                    rng.lognormvariate(u.mu + 0.8, 0.7),
                    0,
                    "legit",
                    channel="p2p",
                    merchant=("4829", "P2P TRANSFER"),
                    merchant_id=f"p2p:{pid}",
                    payee_id=pid,
                    payee_name=typed,
                    payee_account_name=acct,
                    memo=rng.choice(LEGIT_MEMOS),
                    device=u.devices[-1],
                )
            elif r < 0.40:
                emu = rng.random() < 0.004
                add(
                    u,
                    ts,
                    rng.lognormvariate(u.mu, 0.6),
                    0,
                    "legit",
                    channel="ecommerce",
                    merchant=("5999", rng.choice(LEGIT_DESCRIPTORS)),
                    is_emulator=emu,
                )
            else:
                add(u, ts, rng.lognormvariate(u.mu, 0.6), 0, "legit")

    # ---------------------------------------------------------------- fraud
    def n_ep(per_500: float) -> int:
        return max(1, int(round(per_500 * n_users / 500 * fraud_scale)))

    def when(lo=0.05, hi=0.97):
        # fraud skews to night-time but overlaps legit hours heavily
        t = start_ts + rng.uniform(horizon * lo, horizon * hi)
        if rng.random() < 0.35:
            t = t - (t % DAY) + 3600 * rng.uniform(0, 6)
        return t

    for _ in range(n_ep(18)):  # card testing
        u, ts = rng.choice(users), when()
        dev, ip = f"dx{rng.randint(0, 60)}", f"185.220.{rng.randint(0, 9)}.{rng.randint(1, 254)}"
        far = jitter(rng.choice(CITIES))
        for k in range(rng.randint(4, 10)):
            add(
                u,
                ts + k * rng.uniform(15, 90),
                rng.uniform(0.5, 3.0),
                1,
                "card_testing",
                channel="ecommerce",
                device=dev,
                ip=ip,
                merchant=("5816", f"DIGITAL GOODS {k}"),
                merchant_id=f"dg_{rng.randint(0, 99)}",
                loc=far,
                is_emulator=rng.random() < 0.5,
            )
        add(
            u,
            ts + 1200,
            rng.uniform(300, 1500),
            1,
            "card_testing",
            channel="ecommerce",
            device=dev,
            ip=ip,
            merchant=("5999", "ELECTRONICS HUB"),
            loc=far,
        )
    for _ in range(n_ep(14)):  # classic ATO
        u, ts = rng.choice(users), when()
        dev, ip = f"dx{rng.randint(0, 60)}", f"185.220.{rng.randint(0, 9)}.{rng.randint(1, 254)}"
        far = jitter(rng.choice(CITIES))
        for k in range(rng.randint(2, 3)):
            add(
                u,
                ts + k * rng.uniform(120, 900),
                rng.uniform(300, 2500),
                1,
                "account_takeover",
                channel="ecommerce",
                device=dev,
                ip=ip,
                merchant=(rng.choice(["6051", "7995", "6540"]), "COINS & CARDS"),
                loc=far,
            )
    for _ in range(n_ep(12)):  # impossible travel
        u, ts = rng.choice(users), when()
        add(u, ts, rng.lognormvariate(u.mu, 0.5), 0, "legit")
        add(
            u,
            ts + rng.uniform(300, 2400),
            rng.uniform(150, 900),
            1,
            "impossible_travel",
            merchant=("5999", "DUTY FREE SHOP"),
            loc=jitter(rng.choice([c for c in CITIES if c != u.home])),
        )
    for _ in range(n_ep(22)):  # descriptor spoof
        u, ts = rng.choice(users), when()
        add(
            u,
            ts,
            rng.lognormvariate(u.mu + 0.3, 0.5),
            1,
            "descriptor_spoof",
            channel="ecommerce",
            merchant=("5999", rng.choice(LOOKALIKE_DESCRIPTORS)),
            merchant_id=f"spoof_{rng.randint(0, 9)}",
        )
    for _ in range(n_ep(28)):  # APP scam with scam text and payee mismatch
        u, ts = rng.choice(users), when()
        pid = f"pay_scam_{rng.randint(0, 999)}"
        ev(ts - rng.uniform(300, 1800), u.uid, "new_payee", u.devices[-1], u.ip)
        coached = rng.random() < 0.5
        add(
            u,
            ts,
            rng.lognormvariate(u.mu + 1.3, 0.5),
            1,
            "app_scam",
            channel="p2p",
            merchant=("4829", "P2P TRANSFER"),
            merchant_id=f"p2p:{pid}",
            payee_id=pid,
            payee_name=rng.choice(
                ["Bank Safe Account", "Security Dept", "HMRC Refunds", "Agent Lee"]
            ),
            payee_account_name=f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            memo=rng.choice(SCAM_MEMOS)
            if not coached or rng.random() < 0.5
            else rng.choice(BENIGN_MULE_MEMOS),
            session_text=rng.choice(COACHED) if coached else "",
            device=u.devices[-1],
        )
    for _ in range(n_ep(34)):  # ATO with warm-up script
        u, t0 = rng.choice(users), when()
        dev, ip = f"dw{rng.randint(0, 999)}", f"45.83.{rng.randint(0, 99)}.{rng.randint(1, 254)}"
        ev(t0, u.uid, "login_new_device", dev, ip)
        ev(t0 + rng.uniform(120, 900), u.uid, "password_reset", dev, ip)
        ev(t0 + rng.uniform(1000, 2400), u.uid, "new_payee", dev, ip)
        for k in range(rng.randint(1, 3)):
            ev(t0 + 3000 + k * 1800, u.uid, "login", dev, ip)
        pid = f"pay_ato_{rng.randint(0, 999)}"
        acct = f"{rng.choice(FIRST)} {rng.choice(LAST)}"
        add(
            u,
            t0 + rng.uniform(2, 20) * 3600,
            rng.lognormvariate(u.mu + 1.0, 0.4),
            1,
            "ato_warmup",
            channel="p2p",
            merchant=("4829", "P2P TRANSFER"),
            merchant_id=f"p2p:{pid}",
            payee_id=pid,
            payee_name=acct,
            payee_account_name=acct,
            memo=rng.choice(["", "", "transfer", "savings"]),
            device=dev,
            ip=ip,
        )

    # low-and-slow synthetic ring: new accounts on shared emulators
    ring_devs = [f"emu_{k}" for k in range(3)]
    ring_ips = [f"91.200.{k}.7" for k in range(2)]
    ring_start = start_ts + 10 * DAY
    for k in range(max(1, int(6 * fraud_scale))):
        su = _User(
            f"syn{k:03d}",
            f"csyn{k:03d}",
            f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            rng.choice(CITIES),
            [rng.choice(ring_devs)],
            rng.choice(ring_ips),
            2.5,
            0.0,
            [("5816", "GAMECOINS STORE")],
        )
        t = ring_start + rng.uniform(0, 5 * DAY)
        ev(t - 3600, su.uid, "login_new_device", su.devices[0], su.ip)
        while t < start_ts + horizon:
            add(
                su,
                t,
                rng.uniform(5, 40),
                1,
                "low_and_slow",
                channel="ecommerce",
                merchant=(
                    "5816",
                    rng.choice(["GAMECOINS STORE", "GIFTCARD DEPOT", "DIGITAL KEYS"]),
                ),
                merchant_id=f"dk_{rng.randint(0, 5)}",
                is_emulator=rng.random() < 0.7,
            )
            t += rng.uniform(10, 30) * 3600

    # money-mule ring that only appears late (concept drift)
    mule_devs = [f"mdev_{k}" for k in range(2)]
    mules = [(f"mule_{k}", f"{rng.choice(FIRST)} {rng.choice(LAST)}") for k in range(6)]
    t_m = start_ts + mule_start_day * DAY
    for mid, _ in mules:
        ev(t_m - DAY, mid, "login_new_device", rng.choice(mule_devs), "77.91.1.3")
    while t_m < start_ts + horizon:
        u = rng.choice(users)
        mid, acct = rng.choice(mules)
        ev(t_m - rng.uniform(300, 3600), u.uid, "new_payee", u.devices[-1], u.ip)
        add(
            u,
            t_m,
            rng.lognormvariate(u.mu + 1.2, 0.5),
            1,
            "mule_ring",
            channel="p2p",
            merchant=("4829", "P2P TRANSFER"),
            merchant_id=f"p2p:{mid}",
            payee_id=mid,
            payee_name=rng.choice(
                [acct, typed_name(acct), f"{rng.choice(FIRST)} {rng.choice(LAST)}"]
            ),
            payee_account_name=acct,
            memo=rng.choice(BENIGN_MULE_MEMOS),
            session_text=rng.choice(COACHED) if rng.random() < 0.3 else "",
            device=u.devices[-1],
        )
        if rng.random() < 0.5:
            ev(t_m + 600, mid, "login", rng.choice(mule_devs), "77.91.1.3")
        t_m += rng.uniform(1.5, 6) * 3600 / fraud_scale

    # ---------------------------------------------------------------- label arrival
    delays = {"app_scam": (1, 7), "mule_ring": (1, 7), "ato_warmup": (2, 10)}
    label_events = []
    for t in txns:
        if labels[t.txn_id] == 1 and rng.random() > 0.05:
            lo, hi = delays.get(scen[t.txn_id], (20, 60))
            src = "report" if scen[t.txn_id] in delays else "chargeback"
            label_events.append(
                LabelEvent(
                    t.timestamp + rng.uniform(lo, hi) * DAY, t.txn_id, 1, src, scen[t.txn_id]
                )
            )
    txns.sort(key=lambda x: x.timestamp)
    events.sort(key=lambda e: e.timestamp)
    label_events.sort(key=lambda e: e.timestamp)
    return AdvancedStream(txns, events, label_events, labels, scen, start_ts, days)
