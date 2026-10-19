"""Human-readable reason codes from XGBoost SHAP contributions."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

REASON_TEXT = {
    "amount": "Large transaction amount",
    "log_amount": "Large transaction amount",
    "is_round_amount": "Round-number amount",
    "ch_p2p": "Peer-to-peer transfer channel",
    "ch_ecommerce": "Card-not-present (e-commerce) channel",
    "card_amt_10m": "High spend on this card in the last 10 min",
    "card_amt_1h": "High spend on this card in the last hour",
    "card_amt_24h": "High spend on this card in the last 24 h",
    "secs_since_last": "Unusual time since the previous transaction",
    "account_age_days": "Young account history",
    "card_cnt_1m": "Burst of transactions on this card in the last minute",
    "card_cnt_10m": "High transaction velocity on this card (10 min)",
    "card_small_cnt_10m": "Many tiny charges in 10 min (card-testing pattern)",
    "card_distinct_merch_1h": "Many distinct merchants within an hour",
    "device_distinct_cards_24h": "Device used with many different cards",
    "ip_distinct_cards_24h": "IP address used with many different cards",
    "is_new_device": "Transaction from a device not seen for this user",
    "is_new_ip": "Transaction from an IP not seen for this user",
    "is_new_merchant": "First purchase at this merchant",
    "amount_zscore": "Amount is unusual for this user",
    "amount_to_mean_ratio": "Amount far above this user's typical spend",
    "km_from_last": "Far from the previous transaction location",
    "speed_kmh": "Implied travel speed is impossible",
    "high_risk_mcc": "High-risk merchant category",
    "jev_f_scam_urgency": "Payment text shows scam-style urgency or coercion",
    "jev_f_impersonation": "Payment text impersonates a bank, agency or support desk",
    "jev_f_investment_scam": "Payment text promises guaranteed returns or prizes",
    "jev_f_payment_redirect": "Payment text redirects funds to gift cards, crypto or a 'safe account'",
    "jev_f_descriptor_spoof": "Merchant descriptor imitates a known brand",
}


def reason_codes(
    booster: xgb.Booster, x: np.ndarray, feature_names: list[str], top_k: int = 3
) -> list[dict]:
    dm = xgb.DMatrix(x.reshape(1, -1), feature_names=feature_names)
    contribs = booster.predict(dm, pred_contribs=True)[0][:-1]
    order = np.argsort(-contribs)
    out = []
    for i in order[:top_k]:
        if contribs[i] <= 0:
            break
        name = feature_names[i]
        out.append(
            {
                "feature": name,
                "contribution": round(float(contribs[i]), 4),
                "value": None if np.isnan(x[i]) else round(float(x[i]), 4),
                "reason": REASON_TEXT.get(name, name.replace("_", " ")),
            }
        )
    return out
