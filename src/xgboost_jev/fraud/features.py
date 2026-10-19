"""Streaming transaction features: velocity, behaviour deviation, geo-velocity."""

from __future__ import annotations

import math

import numpy as np

from .schema import Transaction
from .store import WINDOWS_S, OnlineFeatureStore

HIGH_RISK_MCC = {"4829", "6051", "6540", "7995", "5816", "5967", "6211"}
CHANNELS = ("card_present", "ecommerce", "p2p", "atm")


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _feature_names() -> list[str]:
    names = ["amount", "log_amount", "is_round_amount", "hour_sin", "hour_cos", "high_risk_mcc"]
    names += [f"ch_{c}" for c in CHANNELS]
    for w in WINDOWS_S:
        names += [f"card_cnt_{w}", f"card_amt_{w}"]
    names += [
        "card_small_cnt_10m",
        "card_distinct_merch_1h",
        "user_cnt_24h",
        "device_cnt_1h",
        "device_distinct_cards_24h",
        "ip_cnt_10m",
        "ip_distinct_cards_24h",
        "user_history_len",
        "account_age_days",
        "amount_zscore",
        "amount_to_mean_ratio",
        "secs_since_last",
        "is_new_device",
        "is_new_ip",
        "is_new_merchant",
        "km_from_last",
        "speed_kmh",
    ]
    return names


FEATURE_NAMES: list[str] = _feature_names()


class TransactionFeaturizer:
    def __init__(self, store: OnlineFeatureStore | None = None) -> None:
        self.store = store or OnlineFeatureStore()

    def compute(self, t: Transaction) -> dict[str, float]:
        """Read-only: features from history strictly before ``t``."""
        s, now, nan = self.store, t.timestamp, math.nan
        hour = (now % 86400) / 3600
        f: dict[str, float] = {
            "amount": t.amount,
            "log_amount": math.log1p(t.amount),
            "is_round_amount": float(t.amount >= 50 and t.amount % 50 == 0),
            "hour_sin": math.sin(2 * math.pi * hour / 24),
            "hour_cos": math.cos(2 * math.pi * hour / 24),
            "high_risk_mcc": float(t.mcc in HIGH_RISK_MCC),
        }
        for c in CHANNELS:
            f[f"ch_{c}"] = float(t.channel == c)
        card = f"card:{t.card_id}"
        for w, secs in WINDOWS_S.items():
            ev = s.window(card, now, secs)
            f[f"card_cnt_{w}"] = float(len(ev))
            f[f"card_amt_{w}"] = float(sum(e.amount for e in ev))
        f["card_small_cnt_10m"] = float(sum(e.amount < 5 for e in s.window(card, now, 600)))
        f["card_distinct_merch_1h"] = float(len({e.merchant_id for e in s.window(card, now, 3600)}))
        f["user_cnt_24h"] = float(len(s.window(f"user:{t.user_id}", now, 86400)))
        if t.device_id:
            f["device_cnt_1h"] = float(len(s.window(f"device:{t.device_id}", now, 3600)))
            f["device_distinct_cards_24h"] = float(
                len({e.card_id for e in s.window(f"device:{t.device_id}", now, 86400)})
            )
        else:
            f["device_cnt_1h"] = f["device_distinct_cards_24h"] = nan
        if t.ip:
            f["ip_cnt_10m"] = float(len(s.window(f"ip:{t.ip}", now, 600)))
            f["ip_distinct_cards_24h"] = float(
                len({e.card_id for e in s.window(f"ip:{t.ip}", now, 86400)})
            )
        else:
            f["ip_cnt_10m"] = f["ip_distinct_cards_24h"] = nan

        p = s.profile(f"user:{t.user_id}")
        if p is None:
            f.update(
                {
                    k: nan
                    for k in (
                        "account_age_days",
                        "amount_zscore",
                        "amount_to_mean_ratio",
                        "secs_since_last",
                        "km_from_last",
                        "speed_kmh",
                    )
                }
            )
            f.update(user_history_len=0.0, is_new_device=1.0, is_new_ip=1.0, is_new_merchant=1.0)
            return f
        f["user_history_len"] = float(p.n)
        f["account_age_days"] = (now - p.first_seen) / 86400
        std = p.std_log_amt
        f["amount_zscore"] = (f["log_amount"] - p.mean_log_amt) / std if std and std > 1e-6 else nan
        f["amount_to_mean_ratio"] = t.amount / max(math.expm1(p.mean_log_amt), 1.0)
        f["secs_since_last"] = now - p.last_ts if p.last_ts is not None else nan
        f["is_new_device"] = float(bool(t.device_id) and t.device_id not in p.devices)
        f["is_new_ip"] = float(bool(t.ip) and t.ip not in p.ips)
        f["is_new_merchant"] = float(t.merchant_id not in p.merchants)
        if None not in (t.lat, t.lon, p.last_lat, p.last_lon):
            km = haversine_km(p.last_lat, p.last_lon, t.lat, t.lon)
            f["km_from_last"] = km
            dt_h = max((now - (p.last_ts or now)) / 3600, 1 / 60)
            f["speed_kmh"] = km / dt_h
        else:
            f["km_from_last"] = f["speed_kmh"] = nan
        return f

    def vector(self, feats: dict[str, float]) -> np.ndarray:
        return np.array([feats[n] for n in FEATURE_NAMES], dtype=np.float32)

    def observe(self, t: Transaction) -> None:
        self.store.observe(t)
