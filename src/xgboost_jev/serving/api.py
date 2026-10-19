"""FastAPI service exposing the guardrail router and the fraud scorer.

Environment:
  JEV_API_URL / JEV_API_KEY      real Jev endpoint (otherwise the offline mock is used)
  XGBJEV_GUARDRAIL_MODEL         directory saved by GuardrailModel.save (static model)
  XGBJEV_FRAUD_MODEL             directory saved by FraudModels.save
If a model path is unset, a small model is trained on mock data at startup.
"""

from __future__ import annotations

import logging
import os
import time
from collections import OrderedDict

from pydantic import BaseModel  # noqa: E402 - installed with the [serve] extra

from ..fraud import (
    AccountEvent,
    AdvancedFraudScorer,
    AnalystAssistant,
    FraudModels,
    LabelEvent,
    RealTimeFraudScorer,
    Transaction,
    generate_transactions,
    train_fraud_models,
)
from ..guardrail import (
    EnsembleGuard,
    GuardrailModel,
    LatencyRouter,
    generate_injection_dataset,
    train_guardrail,
)
from ..jev import HttpJevClient, JevClient, MockJevClient, ResilientJevClient
from ..monitoring import LatencyRecorder

log = logging.getLogger(__name__)


class GuardRequest(BaseModel):
    text: str


class TxnRequest(BaseModel):
    txn_id: str
    timestamp: float | None = None
    user_id: str
    card_id: str
    amount: float
    merchant_id: str
    mcc: str = "5999"
    channel: str = "card_present"
    device_id: str | None = None
    ip: str | None = None
    lat: float | None = None
    lon: float | None = None
    merchant_descriptor: str = ""
    memo: str = ""
    payee_id: str | None = None
    payee_name: str = ""
    payee_account_name: str = ""
    session_text: str = ""
    is_emulator: bool | None = None
    is_rooted: bool | None = None


class AccountEventRequest(BaseModel):
    user_id: str
    kind: str
    timestamp: float | None = None
    device_id: str | None = None
    ip: str | None = None


class LabelRequest(BaseModel):
    txn_id: str
    label: int = 1
    source: str = "analyst"
    fraud_type: str = ""
    timestamp: float | None = None


class RuleRequest(BaseModel):
    id: str
    action: str
    conditions: list[dict]
    priority: int = 100
    reason: str = ""
    expires_at: float | None = None


def default_jev() -> JevClient:
    if os.environ.get("JEV_API_URL"):
        return ResilientJevClient(HttpJevClient())
    log.warning("JEV_API_URL not set - using MockJevClient (offline lexicon stand-in)")
    return ResilientJevClient(MockJevClient())


def build_services(jev: JevClient | None = None):
    jev = jev or default_jev()
    gpath, fpath = os.environ.get("XGBJEV_GUARDRAIL_MODEL"), os.environ.get("XGBJEV_FRAUD_MODEL")
    if gpath:
        static = GuardrailModel.load(gpath)
    else:
        df = generate_injection_dataset(2000, split="train")
        static = train_guardrail(df.text, df.label, use_jev=False)
    fraud_models = (
        FraudModels.load(fpath)
        if fpath
        else train_fraud_models(generate_transactions(n_users=200, days=14), jev)[0]
    )
    router = LatencyRouter(jev, EnsembleGuard(static, jev))
    return router, RealTimeFraudScorer(fraud_models, jev)


def create_app(
    router: LatencyRouter | None = None,
    fraud: RealTimeFraudScorer | None = None,
    advanced: AdvancedFraudScorer | None = None,
    analyst: AnalystAssistant | None = None,
):
    from fastapi import FastAPI, HTTPException

    if router is None or fraud is None:
        router, fraud = build_services()
    latency = LatencyRecorder()
    app = FastAPI(title="xgboost-jev", version="0.1.0")

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.post("/v1/guardrail/check")
    def guardrail_check(req: GuardRequest):
        d = router.check(req.text)
        latency.record(f"guardrail_{d.stage}", d.latency_ms)
        return d.to_dict()

    @app.post("/v1/fraud/score")
    def fraud_score(req: TxnRequest):
        data = req.model_dump()
        data["timestamp"] = data["timestamp"] or time.time()
        d = fraud.score(Transaction.from_dict(data))
        latency.record(f"fraud_{d.model}", d.latency_ms)
        return d.to_dict()

    recent: OrderedDict[str, tuple] = OrderedDict()  # txn_id -> (txn, decision) for case files
    assistant = analyst or AnalystAssistant()

    def _adv() -> AdvancedFraudScorer:
        if advanced is None:
            raise HTTPException(503, "advanced fraud scorer not configured")
        return advanced

    @app.post("/v2/fraud/score")
    def fraud_score_v2(req: TxnRequest, budget_ms: float | None = None):
        sc = _adv()
        data = req.model_dump()
        data["timestamp"] = data["timestamp"] or time.time()
        t = Transaction.from_dict(data)
        d = sc.score(t, budget_ms=budget_ms)
        latency.record(f"fraud_v2_{d.model}", d.latency_ms)
        recent[t.txn_id] = (t, d)
        while len(recent) > 10_000:
            recent.popitem(last=False)
        return d.to_dict()

    @app.post("/v2/fraud/events")
    def account_event(req: AccountEventRequest):
        _adv().on_account_event(
            AccountEvent(req.timestamp or time.time(), req.user_id, req.kind, req.device_id, req.ip)
        )
        return {"ok": True}

    @app.post("/v2/fraud/labels")
    def label(req: LabelRequest):
        _adv().on_label(
            LabelEvent(
                req.timestamp or time.time(), req.txn_id, req.label, req.source, req.fraud_type
            )
        )
        return {"ok": True}

    @app.get("/v2/fraud/rules")
    def list_rules():
        return [r.to_dict() for r in _adv().rules.list()]

    @app.post("/v2/fraud/rules")
    def add_rule(req: RuleRequest):
        try:
            return _adv().rules.add(req.model_dump()).to_dict()
        except (ValueError, KeyError) as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.delete("/v2/fraud/rules/{rule_id}")
    def delete_rule(rule_id: str):
        if not _adv().rules.remove(rule_id):
            raise HTTPException(404, "no such rule")
        return {"ok": True}

    @app.get("/v2/fraud/case/{txn_id}")
    def case(txn_id: str):
        sc = _adv()
        if txn_id not in recent:
            raise HTTPException(404, "transaction not in the recent-decisions buffer")
        t, d = recent[txn_id]
        return assistant.build_case(t, d, sc.featurizer).to_dict()

    @app.get("/metrics")
    def metrics():
        s = router.stats
        return {
            "latency": latency.summary(),
            "router": {
                "total": s.total,
                "short_circuit_rate": round(s.short_circuit_rate, 4),
                "stage2": s.stage2,
                "counterfeit_escalations": s.counterfeit_escalations,
                "jev_failures": s.jev_failures,
            },
        }

    return app


def main() -> None:  # pragma: no cover
    import uvicorn

    uvicorn.run(
        create_app(),
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "8000")),
    )
