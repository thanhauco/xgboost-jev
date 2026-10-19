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
    "jev_f_coached": "Customer says someone instructed them to move the money",
    "payee_fan_in_7d": "Payee received money from many different senders this week (mule pattern)",
    "payee_age_days": "Payee account is very new",
    "payee_new_for_user": "First payment to this payee",
    "g_payee_fraud": "Payee previously received confirmed fraud",
    "g_payee_comp_fraud": "Payee shares devices with accounts linked to confirmed fraud",
    "g_payee_comp_size": "Payee belongs to a large linked account cluster",
    "g_device_users": "Device shared by many accounts",
    "g_device_fraud": "Device previously used in confirmed fraud",
    "g_ip_fraud": "IP previously used in confirmed fraud",
    "g_card_fraud": "Card previously used in confirmed fraud",
    "g_comp_fraud": "Linked to confirmed-fraud devices/IPs",
    "g_comp_size": "Part of a large cluster of linked accounts",
    "g_2hop_fraud": "Two hops from confirmed-fraud infrastructure",
    "r_merchant": "Merchant with elevated recent fraud rate",
    "r_mcc": "Merchant category with elevated recent fraud rate",
    "peer_amount_z": "Amount unusual for similar customers",
    "dc_card_burst": "Spend today far above this card's 30-day baseline",
    "dc_card_amt_1d": "High spend on this card today",
    "seq_h_since_password_reset": "Recent password reset",
    "seq_h_since_login_new_device": "Recent login from a new device",
    "seq_h_since_new_payee": "Payee added recently",
    "seq_reset_then_payee_48h": "Password reset followed by a new payee within 48 h (takeover script)",
    "seq_markov_surprise": "Unusual sequence of account actions",
    "seq_risky_24h": "Several risky account changes in 24 h",
    "cop_similarity": "Payee name does not match the receiving account",
    "desc_brand_lookalike": "Merchant name imitates a known brand",
    "desc_suspicious_tokens": "Merchant name contains refund/verify/support-style words",
    "r_merchant_vol": "New or low-volume merchant / payee",
    "dc_card_cnt_1h": "Many payments on this card in the last hour",
    "dc_card_amt_30d": "Card spend baseline over 30 days",
    "dc_user_amt_7d": "High spend by this customer this week",
    "dc_device_cnt_1d": "Unusual activity level on this device today",
    "dc_payee_in_cnt_1d": "Payee received many payments today",
    "dc_payee_in_amt_1d": "Payee received large inflows today",
    "seq_h_since_mfa_change": "Recent MFA change",
    "seq_h_since_profile_change": "Recent profile (email/phone) change",
    "desc_leet": "Merchant name uses look-alike characters (0 for o, 1 for l)",
    "dev_emulator": "Device is an emulator",
    "dev_rooted": "Device is rooted/jailbroken",
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
