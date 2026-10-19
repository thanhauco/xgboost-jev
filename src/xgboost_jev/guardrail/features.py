"""Cheap, deterministic text features computed locally in microseconds."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Iterable

import pandas as pd

_TOKEN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_ZERO_WIDTH = re.compile(r"[​-‏⁠﻿]")
_BASE64 = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_URL = re.compile(r"https?://\S+", re.I)
_IMPERATIVE = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|reveal|pretend|jailbreak|unfiltered|"
    r"developer mode|system prompt|act as|you are now)\b",
    re.I,
)
_DELIMITER = re.compile(r"```|###|<\|[^|]{0,20}\|>|\[/?INST\]|<</?SYS>>|-{5,}|={5,}")
_ROLE_MARKER = re.compile(r"(^|\n)\s*(system|assistant|user|admin|developer)\s*:", re.I)

# Fake-authority markers used by "counterfeit policy" attacks, where a forged
# system/policy block is prepended to make an injection look sanctioned.
_CF_HEADER = re.compile(
    r"(^|\n)\s*[\[<#(]*\s*(system|admin|developer|policy|security|compliance)\s*"
    r"(message|notice|update|override|directive)?\s*[\]>)]*\s*:",
    re.I,
)
_CF_CLAIM = re.compile(
    r"\b((policy|guidelines?|rules?) (has|have) been (updated|changed|lifted)|"
    r"(authorized|approved|sanctioned) by (the )?(developers?|admins?|openai|anthropic|security team)|"
    r"you are (now )?(permitted|allowed|authorized)|safety (filters|checks) (are )?(disabled|off))\b",
    re.I,
)
_CF_WRAPPER = re.compile(r"<\|im_start\|>\s*system|<<SYS>>|\[SYSTEM\]|BEGIN (SYSTEM|POLICY)", re.I)


def shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def counterfeit_policy_features(text: str) -> dict[str, float]:
    head = text[: max(200, len(text) // 3)]
    headers = len(_CF_HEADER.findall(text))
    claims = len(_CF_CLAIM.findall(text))
    wrappers = len(_CF_WRAPPER.findall(text))
    prefix = 1.0 if (_CF_HEADER.search(head) or _CF_WRAPPER.search(head)) else 0.0
    score = min(
        1.0, 0.35 * min(headers, 2) + 0.3 * min(claims, 2) + 0.35 * min(wrappers, 1) + 0.2 * prefix
    )
    return {
        "cf_header_count": float(headers),
        "cf_authority_claims": float(claims),
        "cf_wrapper_count": float(wrappers),
        "cf_prefix": prefix,
        "cf_score": score,
    }


def text_metadata_features(text: str) -> dict[str, float]:
    n = len(text)
    tokens = _TOKEN.findall(text)
    words = [t.lower() for t in tokens if t.isalnum()]
    denom = max(n, 1)
    feats = {
        "char_len": float(n),
        "log_char_len": math.log1p(n),
        "token_count": float(len(tokens)),
        "entropy": shannon_entropy(text),
        "upper_ratio": sum(c.isupper() for c in text) / denom,
        "digit_ratio": sum(c.isdigit() for c in text) / denom,
        "punct_ratio": sum((not c.isalnum()) and (not c.isspace()) for c in text) / denom,
        "non_ascii_ratio": sum(ord(c) > 127 for c in text) / denom,
        "zero_width_count": float(len(_ZERO_WIDTH.findall(text))),
        "base64_span_count": float(len(_BASE64.findall(text))),
        "url_count": float(len(_URL.findall(text))),
        "imperative_count": float(len(_IMPERATIVE.findall(text))),
        "delimiter_count": float(len(_DELIMITER.findall(text))),
        "role_marker_count": float(len(_ROLE_MARKER.findall(text))),
        "repeat_word_ratio": 1 - len(set(words)) / len(words) if words else 0.0,
    }
    feats.update(counterfeit_policy_features(text))
    return feats


def metadata_frame(texts: Iterable[str]) -> pd.DataFrame:
    return pd.DataFrame([text_metadata_features(t) for t in texts])
