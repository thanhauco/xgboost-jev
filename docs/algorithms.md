# Algorithms, animated

Each algorithm in this repo has a 14–16 second motion-graphics explainer. The GIFs play inline on GitHub; each one links to a sharper MP4 (960×540, 30 fps, H.264).

Numbers shown in the animations come from [`benchmarks.json`](benchmarks.json), produced by `python -m xgboost_jev.bench`, so the videos and the [benchmark report](benchmarks.md) always agree. To regenerate after re-running the benchmark:

```bash
python tools/animations/render.py            # all scenes, needs ffmpeg + Pillow
python tools/animations/render.py --only 03  # one scene
python tools/animations/render.py --only 03 --still 6.0   # single PNG frame for review
```

Design details are in [architecture.md](architecture.md).

---

## 1. Multi-layer feature cascade

[![Feature cascade](media/01_feature_cascade.gif)](media/01_feature_cascade.mp4)

Untrusted text fans out into three feature producers: local metadata, hashed n-gram embeddings reduced by SVD, and Jev, which answers several `noul` questions in one round trip. Their outputs form one numeric row, and XGBoost turns that row into a single attack score. In the second half of the clip Jev times out and its columns become NaN. Because the model was trained with Jev dropout, each split follows its learned default branch and the text is still blocked.

Code: `guardrail/cascade.py`, `guardrail/features.py`, `guardrail/embeddings.py`

## 2. Ensemble hardening against counterfeit policies

[![Ensemble hardening](media/02_ensemble_hardening.gif)](media/02_ensemble_hardening.mp4)

A forged `[SYSTEM]` policy header talks Jev's probability down, while the static XGBoost voter, which does not follow instructions, stays high. A local counterfeit detector shrinks Jev's vote. Asymmetric trust means a low Jev score can never veto XGBoost's attack verdict, which is judged at XGBoost's own tuned threshold.

Code: `guardrail/ensemble.py`, `guardrail/features.py::counterfeit_policy_features`

## 3. Multi-stage latency routing

[![Latency routing](media/03_latency_routing.gif)](media/03_latency_routing.mp4)

Every request gets one cheap Jev question. Confident answers short-circuit; ambiguous answers, Jev failures, and fake-policy requests that trick Stage 1 low all go to the deep check.

Code: `guardrail/router.py`

## 4. Real-time fraud scorer with tiered Jev escalation

[![Fraud scorer](media/04_fraud_realtime_scorer.gif)](media/04_fraud_realtime_scorer.mp4)

An authorized-push-payment scam looks normal on tabular features, since it comes from the victim's own phone at home. Because the payment is P2P with a memo, it escalates to Jev's five semantic questions. The hybrid booster re-scores it, the hybrid's own cost-tuned policy declines it, and SHAP reason codes explain the decision.

Code: `fraud/scorer.py`, `fraud/semantic.py`, `fraud/explain.py`

## 5. Streaming features: velocity, geo-velocity, point-in-time

[![Streaming features](media/05_streaming_features.gif)](media/05_streaming_features.mp4)

The clip shows three things:
- Per-entity sliding windows catch a card-testing burst.
- Haversine distance divided by elapsed time exposes impossible travel.
- Compute-then-observe ordering keeps training replay and online scoring identical.

Code: `fraud/store.py`, `fraud/features.py`

## 6. Resilience: cache, circuit breaker, NaN degradation

[![Resilience](media/06_resilience.gif)](media/06_resilience.mp4)

Cache hits return without a network call. Repeated Jev failures open the breaker, and from then on calls skip Jev and return NaN columns immediately. After the reset timeout a half-open probe closes the breaker again. The benchmark shows the cascade loses almost no ROC-AUC while Jev is down.

Code: `jev/resilience.py`

## 7. Cost-based decline threshold and PSI drift

[![Thresholds and drift](media/07_cost_threshold_drift.gif)](media/07_cost_threshold_drift.mp4)

Sweeping the threshold trades missed-fraud dollars against false-decline cost, and each model gets its own minimum-cost threshold. In production, PSI compares live feature distributions against the training reference and alerts above 0.25. The cost curve in this clip is illustrative; the threshold values and PSI numbers are from the benchmark.

Code: `fraud/scorer.py::choose_decline_threshold`, `monitoring/drift.py`
