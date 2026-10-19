"""Synthetic transaction stream with labelled fraud scenarios.

Scenarios and the signal that should catch them:
  card_testing       burst of tiny e-commerce charges, new device  -> velocity
  account_takeover   new device + IP, far away, high-risk MCC      -> behaviour/geo
  impossible_travel  card-present swipe far away minutes later     -> geo-velocity
  app_scam           P2P transfer from the victim's OWN device with
                     a scam memo ("bank security: move to safe
                     account")                                     -> text (Jev)
  descriptor_spoof   checkout at a look-alike merchant descriptor  -> text (Jev)

The last two are designed to look normal on tabular features, which is where
semantic Jev features are expected to add lift. Mock data only.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .schema import Transaction

CITIES = [
    (40.71, -74.01),
    (34.05, -118.24),
    (41.88, -87.63),
    (29.76, -95.37),
    (47.61, -122.33),
    (51.51, -0.13),
    (48.86, 2.35),
    (35.68, 139.69),
    (1.35, 103.82),
    (-33.87, 151.21),
]
MERCHANTS = {
    "5411": ["WHOLEFDS MKT #10234", "TRADER JOE'S #552", "SAFEWAY 1123"],
    "5812": ["CHIPOTLE 0921", "SWEETGREEN NYC", "LOCAL DINER"],
    "5814": ["STARBUCKS #2231", "DUNKIN #4410"],
    "5999": ["AMAZON MKTPLACE PMTS", "ETSY.COM", "TARGET.COM"],
    "4121": ["UBER *TRIP", "LYFT *RIDE"],
    "5541": ["SHELL OIL 5741", "CHEVRON 0091"],
}
SPOOF_DESCRIPTORS = [
    "AMAZ0N-PRIME REFUND-CENTER",
    "PAYPA1 VERIFY-NOW",
    "APP1E ID UNLOCK-DESK",
    "NETF1IX ACCT VERIFICATION",
    "MICR0SOFT REFUND-DESK",
    "AMAZON-SECURE-BILLING-HELP",  # look-alike without misspelling: hard for the lexicon
]
LEGIT_MEMOS = [
    "rent october",
    "dinner split",
    "thanks for the pizza",
    "concert tickets",
    "utilities share",
    "urgent - rent due tomorrow!",
    "gift for mom's birthday",
    "bitcoin meetup tickets",
    "paying back for groceries",
    "team lunch",
    "",
    "",
    "",
]
SCAM_MEMOS = [
    "Bank security team: move funds to safe account immediately",
    "IRS final notice - pay now to avoid arrest",
    "crypto investment opportunity guaranteed returns",
    "tech support refund - send via gift cards asap",
    "you have won the lottery, pay release fee within the hour",
    "fraud department instructed transfer to new account details",
    # no lexicon keywords: semantic-only cases the mock Jev will mostly miss
    "hi it's me, lost my phone, this is my new number, need it today pls",
    "as discussed on the call, keep this between us until it clears",
]


@dataclass
class _User:
    uid: str
    card: str
    home: tuple[float, float]
    devices: list[str]
    ip: str
    spend_mu: float
    rate_per_day: float
    favourite: list[tuple[str, str]]


@dataclass
class LabeledStream:
    transactions: list[Transaction]
    labels: dict[str, int]
    scenarios: dict[str, str]

    def label_of(self, t: Transaction) -> int:
        return self.labels[t.txn_id]


def _jitter(loc, rng, deg=0.05):
    return (loc[0] + rng.uniform(-deg, deg), loc[1] + rng.uniform(-deg, deg))


def generate_transactions(
    n_users: int = 500,
    days: int = 30,
    fraud_users: float = 0.3,
    seed: int = 11,
    start_ts: float = 1_788_000_000.0,
) -> LabeledStream:
    rng = random.Random(seed)
    users = []
    for i in range(n_users):
        favs = [(mcc, rng.choice(names)) for mcc, names in rng.sample(list(MERCHANTS.items()), 4)]
        users.append(
            _User(
                f"u{i:05d}",
                f"c{i:05d}",
                rng.choice(CITIES),
                [f"d{i:05d}a"] + ([f"d{i:05d}b"] if rng.random() < 0.4 else []),
                f"10.{i // 250}.{i % 250}.{rng.randint(1, 254)}",
                rng.uniform(2.3, 4.2),
                rng.uniform(1.0, 5.0),
                favs,
            )
        )
    txns: list[Transaction] = []
    labels: dict[str, int] = {}
    scen: dict[str, str] = {}
    seq = 0

    def add(u: _User, ts: float, amount: float, label: int, kind: str, **kw) -> None:
        nonlocal seq
        seq += 1
        tid = f"t{seq:08d}"
        mcc, desc = kw.pop("merchant", rng.choice(u.favourite))
        lat, lon = kw.pop("loc", _jitter(u.home, rng))
        txns.append(
            Transaction(
                txn_id=tid,
                timestamp=ts,
                user_id=u.uid,
                card_id=u.card,
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
            )
        )
        labels[tid], scen[tid] = label, kind

    horizon = days * 86400
    for u in users:
        t = start_ts + rng.uniform(0, 3600 * 6)
        while t < start_ts + horizon:
            t += rng.expovariate(u.rate_per_day / 86400)
            hour_shift = 3600 * rng.choice([9, 12, 13, 18, 19, 20])
            ts = t - (t % 86400) + hour_shift + rng.uniform(0, 3000)
            if ts >= start_ts + horizon:  # never emit past the requested number of days
                break
            r = rng.random()
            if r < 0.12:
                add(
                    u,
                    ts,
                    rng.lognormvariate(u.spend_mu + 0.8, 0.7),
                    0,
                    "legit",
                    channel="p2p",
                    merchant=("4829", "P2P TRANSFER"),
                    merchant_id=f"p2p_{rng.randint(0, 50)}",
                    memo=rng.choice(LEGIT_MEMOS),
                )
            elif r < 0.45:
                add(
                    u,
                    ts,
                    rng.lognormvariate(u.spend_mu, 0.6),
                    0,
                    "legit",
                    channel="ecommerce",
                    merchant=("5999", rng.choice(MERCHANTS["5999"])),
                )
            else:
                add(u, ts, rng.lognormvariate(u.spend_mu, 0.6), 0, "legit")

    for u in rng.sample(users, int(n_users * fraud_users)):
        ts = start_ts + rng.uniform(horizon * 0.2, horizon * 0.95)
        kind = rng.choice(
            [
                "card_testing",
                "account_takeover",
                "impossible_travel",
                "app_scam",
                "descriptor_spoof",
            ]
        )
        bad_dev, bad_ip = (
            f"dx{rng.randint(0, 40)}",
            f"185.220.{rng.randint(0, 9)}.{rng.randint(1, 254)}",
        )
        far = rng.choice([c for c in CITIES if c != u.home])
        if kind == "card_testing":
            for k in range(rng.randint(5, 12)):
                add(
                    u,
                    ts + k * rng.uniform(15, 60),
                    rng.uniform(0.5, 3.0),
                    1,
                    kind,
                    channel="ecommerce",
                    device=bad_dev,
                    ip=bad_ip,
                    merchant=("5816", f"DIGITAL GOODS {k}"),
                    merchant_id=f"dg_{rng.randint(0, 99)}",
                    loc=_jitter(far, rng),
                )
            add(
                u,
                ts + 900,
                rng.uniform(400, 1500),
                1,
                kind,
                channel="ecommerce",
                device=bad_dev,
                ip=bad_ip,
                merchant=("5999", "ELECTRONICS HUB"),
                loc=_jitter(far, rng),
            )
        elif kind == "account_takeover":
            for k in range(rng.randint(2, 4)):
                add(
                    u,
                    ts + k * rng.uniform(120, 900),
                    rng.uniform(300, 2500),
                    1,
                    kind,
                    channel="ecommerce",
                    device=bad_dev,
                    ip=bad_ip,
                    merchant=(rng.choice(["6051", "7995", "6540"]), "COINS & CARDS"),
                    loc=_jitter(far, rng),
                )
        elif kind == "impossible_travel":
            add(u, ts, rng.lognormvariate(u.spend_mu, 0.5), 0, "legit")
            add(
                u,
                ts + rng.uniform(300, 2400),
                rng.uniform(150, 900),
                1,
                kind,
                merchant=("5999", "DUTY FREE SHOP"),
                loc=_jitter(far, rng),
            )
        elif kind == "app_scam":
            add(
                u,
                ts,
                rng.lognormvariate(u.spend_mu + 1.2, 0.5),
                1,
                kind,
                channel="p2p",
                merchant=("4829", "P2P TRANSFER"),
                merchant_id=f"p2p_{rng.randint(0, 50)}",
                memo=rng.choice(SCAM_MEMOS),
            )
        else:
            add(
                u,
                ts,
                rng.lognormvariate(u.spend_mu + 0.3, 0.5),
                1,
                kind,
                channel="ecommerce",
                merchant=("5999", rng.choice(SPOOF_DESCRIPTORS)),
                merchant_id=f"spoof_{rng.randint(0, 9)}",
            )

    txns.sort(key=lambda x: x.timestamp)
    return LabeledStream(txns, labels, scen)
