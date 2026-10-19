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

from pydantic import BaseModel  # noqa: E402 - installed with the [serve] extra

from ..fraud import (
    FraudModels,
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


def create_app(router: LatencyRouter | None = None, fraud: RealTimeFraudScorer | None = None):
    from fastapi import FastAPI

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
