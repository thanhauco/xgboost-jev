"""Local (no-network) text signals: confirmation of payee and look-alike descriptors."""

from __future__ import annotations

import math
import re
from difflib import SequenceMatcher
from functools import lru_cache

BRANDS = (
    "amazon",
    "paypal",
    "apple",
    "google",
    "microsoft",
    "netflix",
    "walmart",
    "chase",
    "venmo",
    "coinbase",
)
OFFICIAL_DESCRIPTORS = re.compile(
    r"^(amazon mktplace pmts|amazon\.com|amzn mktp|paypal \*|apple\.com/bill|google \*|microsoft \*|netflix\.com)",
    re.I,
)
SUSPICIOUS = re.compile(
    r"\b(refund|verify|verification|secure|security|billing|help|unlock|support|desk|center|claim|recovery)\b",
    re.I,
)
_LEET = str.maketrans(
    {"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "$": "s", "@": "a", "7": "t"}
)
_NON = re.compile(r"[^a-z ]+")


def _norm_name(s: str) -> list[str]:
    return sorted(
        t
        for t in _NON.sub(" ", s.lower()).split()
        if t not in {"mr", "mrs", "ms", "dr", "ltd", "llc", "inc"}
    )


@lru_cache(maxsize=65536)
def name_similarity(typed: str, account: str) -> float:
    """0..1 similarity between the payee name typed by the payer and the account holder name.

    Token-sorted fuzzy ratio, with credit for initials ("J Smith" vs "John Smith")
    and first-name-only entries. NaN when either side is missing.
    """
    a, b = _norm_name(typed), _norm_name(account)
    if not a or not b:
        return math.nan
    base = SequenceMatcher(None, " ".join(a), " ".join(b)).ratio()
    hits = sum(1 for t in a if any(t == u or (len(t) == 1 and u.startswith(t)) for u in b))
    return max(base, hits / max(len(a), 1) * (0.95 if len(a) < len(b) else 1.0))


def descriptor_signals(descriptor: str) -> dict[str, float]:
    return dict(_descriptor_signals(descriptor))  # copy: callers may mutate


@lru_cache(maxsize=65536)
def _descriptor_signals(descriptor: str) -> dict[str, float]:
    """Descriptors repeat heavily (one per merchant), so results are cached."""
    if not descriptor:
        return {
            "desc_brand_lookalike": math.nan,
            "desc_suspicious_tokens": math.nan,
            "desc_leet": math.nan,
        }
    raw = descriptor.lower()
    norm = _NON.sub(" ", raw.translate(_LEET))
    leet = float(norm != _NON.sub(" ", raw))
    official = bool(OFFICIAL_DESCRIPTORS.match(descriptor.strip()))
    best = 0.0
    for tok in norm.split():
        for brand in BRANDS:
            if abs(len(tok) - len(brand)) <= 2:
                best = max(best, SequenceMatcher(None, tok, brand).ratio())
    sus = float(len(SUSPICIOUS.findall(norm)))
    lookalike = 0.0 if official else best * (1.0 if (leet or sus) else 0.5)
    return {"desc_brand_lookalike": lookalike, "desc_suspicious_tokens": sus, "desc_leet": leet}
