"""Offline stand-in for Jev used by tests, benchmarks and local development.

``MockJevClient`` scores questions with weighted regex lexicons pushed through a
logistic function. It is deterministic and needs no network, but it is NOT a
real detector: it exists to exercise the pipelines end to end. Point
``HttpJevClient`` at a real Jev endpoint for production numbers.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Mapping, Sequence

from .client import BaseJevClient, JevError
from .schema import JevQuery, JevResult

Rule = tuple[str, float]

INJECTION_RULES: dict[str, list[Rule]] = {
    "p_override": [
        (
            r"\b(ignore|disregard|forget|override|bypass)\b.{0,40}\b(previous|prior|above|all|earlier|your)\b.{0,20}\b(instructions?|rules?|guidelines?|prompts?)\b",
            4.0,
        ),
        (r"\bnew instructions?\b", 2.0),
        (r"\bdo not follow\b", 1.5),
    ],
    "p_jailbreak": [
        (r"\b(DAN|do anything now|developer mode|jailbreak|unfiltered|no restrictions)\b", 3.5),
        (r"\bwithout (any )?(filters?|limits|restrictions|censorship)\b", 2.5),
    ],
    "p_roleplay": [
        (r"\b(pretend|act as|you are now|roleplay|from now on you)\b", 2.5),
        (r"\b(evil|unrestricted|uncensored) (ai|assistant|persona)\b", 2.5),
    ],
    "p_exfiltration": [
        (
            r"\b(system prompt|hidden prompt|initial instructions|api keys?|secrets?|passwords?|credentials)\b",
            2.5,
        ),
        (
            r"\b(reveal|print|output|leak|show|repeat)\b.{0,30}\b(prompt|secrets?|keys?|instructions)\b",
            2.5,
        ),
    ],
    "p_obfuscation": [
        (r"[A-Za-z0-9+/]{32,}={0,2}", 2.5),
        (r"[​-‏⁠﻿]", 3.0),
        (r"\b(base64|rot13|decode this|reverse the following)\b", 2.0),
    ],
    "p_counterfeit_policy": [
        (
            r"(^|\n)\s*[\[<#(]*\s*(system|admin|developer|policy|security)\s*(message|notice|update|override)?\s*[\]>)]*\s*:",
            3.0,
        ),
        (r"\b(policy|guidelines?) (has|have) been updated\b", 2.5),
        (
            r"\b(authorized|approved) by (the )?(developers?|admins?|openai|anthropic|security team)\b",
            2.5,
        ),
    ],
    "p_tool_abuse": [
        (r"\b(send|post|upload|forward|exfiltrate)\b.{0,40}\b(https?://|webhook|email|url)\b", 3.0),
        (r"\b(execute|run) (this|the following) (code|command|script)\b", 2.5),
        (r"\b(rm -rf|curl |wget )", 2.5),
    ],
}
INJECTION_RULES["is_malicious"] = [
    r
    for k in ("p_override", "p_jailbreak", "p_exfiltration", "p_tool_abuse")
    for r in INJECTION_RULES[k]
]

FRAUD_RULES: dict[str, list[Rule]] = {
    "f_scam_urgency": [
        (r"\b(urgent|immediately|right now|within (the )?hour|final notice|act now|asap)\b", 2.5),
        (r"\b(or (your )?account will be|to avoid (arrest|suspension|penalty))\b", 3.0),
    ],
    "f_impersonation": [
        (
            r"\b(bank security|fraud department|irs|tax office|tech support|microsoft support|police|customs)\b",
            3.0,
        ),
    ],
    "f_investment_scam": [
        (
            r"\b(guaranteed (returns?|profit)|double your|crypto (investment|opportunity)|you (have )?won|prize|lottery)\b",
            3.5,
        ),
    ],
    "f_payment_redirect": [
        (
            r"\b(gift ?cards?|safe account|bitcoin|usdt|wire to|new account details|crypto wallet)\b",
            3.0,
        ),
    ],
    "f_descriptor_spoof": [
        (r"\b(amaz0n|paypa1|app1e|g00gle|micr0soft|netf1ix)\b", 4.0),
        (r"\b(refund|verify|verification|unlock)[-_ ]?(center|desk|now|acct)\b", 2.5),
    ],
}

DEFAULT_RULES: dict[str, list[Rule]] = {**INJECTION_RULES, **FRAUD_RULES}
_COUNTERFEIT = re.compile(INJECTION_RULES["p_counterfeit_policy"][0][0], re.I | re.M)


class MockJevClient(BaseJevClient):
    """Deterministic lexicon scorer that mimics Jev's request/response contract.

    Args:
        rules: question id -> [(regex, weight)]. Unknown ids return NaN.
        bias: logistic intercept; with no matches p = sigmoid(bias).
        gain: multiplier on matched rule weights (one strong hit -> p ~ 0.9).
        latency_ms: artificial per-call delay to mirror a network round trip.
        counterfeit_susceptibility: multiply injection scores by
            ``1 - counterfeit_susceptibility`` when the text opens with a fake
            policy header, to simulate a model steered by counterfeit policies.
        fail_rate: deterministic fraction of calls that raise, to test fallbacks.
    """

    def __init__(
        self,
        rules: Mapping[str, Sequence[Rule]] | None = None,
        bias: float = -3.0,
        gain: float = 1.8,
        latency_ms: float = 0.0,
        counterfeit_susceptibility: float = 0.0,
        fail_rate: float = 0.0,
    ) -> None:
        self.rules = {
            qid: [(re.compile(p, re.I | re.M), w) for p, w in rs]
            for qid, rs in (rules or DEFAULT_RULES).items()
        }
        self.bias = bias
        self.gain = gain
        self.latency_ms = latency_ms
        self.counterfeit_susceptibility = counterfeit_susceptibility
        self.fail_rate = fail_rate
        self.calls = 0

    def _score(self, qid: str, text: str) -> float:
        if qid not in self.rules:
            return math.nan
        z = self.bias + self.gain * sum(
            w * min(len(rx.findall(text)), 2) for rx, w in self.rules[qid]
        )
        p = 1.0 / (1.0 + math.exp(-z))
        if (
            self.counterfeit_susceptibility
            and not qid.startswith("f_")
            and qid != "p_counterfeit_policy"
            and _COUNTERFEIT.search(text[:200])
        ):
            p *= 1.0 - self.counterfeit_susceptibility
        return round(p, 6)

    def ask(self, query: JevQuery) -> JevResult:
        self.calls += 1
        start = time.perf_counter()
        if self.fail_rate and (self.calls * 0.6180339887) % 1.0 < self.fail_rate:
            raise JevError("mock Jev failure")
        if self.latency_ms:
            time.sleep(self.latency_ms / 1000)
        probs = {q.id: self._score(q.id, query.state) for q in query.questions}
        return JevResult(probs, (time.perf_counter() - start) * 1000, "mock")
