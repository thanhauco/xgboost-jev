"""Pattern 1 - multi-layer feature cascade.

    untrusted text --+--> metadata (length, tokens, entropy, counterfeit markers) --+
                     +--> hashed n-gram embeddings (SVD reduced) -------------------+--> XGBoost
                     +--> Jev noul probabilities (one parallel round trip) ---------+

Jev answers become plain float columns. During training a fraction of rows has
its Jev columns set to NaN ("Jev dropout") so the booster learns sensible
default branches for when Jev times out in production.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from ..jev.client import JevClient
from ..jev.schema import Question
from .embeddings import HashingSVDEmbedder
from .features import metadata_frame
from .questions import INJECTION_QUESTIONS, injection_query

DEFAULT_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "max_depth": 5,
    "eta": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
    "min_child_weight": 2,
    "tree_method": "hist",
    "nthread": 1,
}


class GuardrailFeatureBuilder:
    def __init__(
        self,
        jev: JevClient | None = None,
        embedder: HashingSVDEmbedder | None = None,
        questions: Sequence[Question] = INJECTION_QUESTIONS,
        use_jev: bool = True,
    ) -> None:
        self.jev = jev
        self.embedder = embedder or HashingSVDEmbedder()
        self.questions = tuple(questions)
        self.use_jev = use_jev

    @property
    def jev_columns(self) -> list[str]:
        return [f"jev_{q.id}" for q in self.questions] if self.use_jev else []

    def fit(self, texts: Sequence[str]) -> GuardrailFeatureBuilder:
        self.embedder.fit(list(texts))
        return self

    def jev_frame(self, texts: Sequence[str]) -> pd.DataFrame:
        cols = self.jev_columns
        if not cols:
            return pd.DataFrame(index=range(len(texts)))
        if self.jev is None:
            return pd.DataFrame(np.nan, index=range(len(texts)), columns=cols)
        results = self.jev.ask_many([injection_query(t, self.questions) for t in texts])
        return pd.DataFrame([[r.get(q.id) for q in self.questions] for r in results], columns=cols)

    def transform(
        self, texts: Sequence[str], jev_frame: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        texts = list(texts)
        meta = metadata_frame(texts)
        emb = pd.DataFrame(self.embedder.transform(texts), columns=self.embedder.columns)
        parts = [meta, emb]
        if self.use_jev:
            parts.append(jev_frame if jev_frame is not None else self.jev_frame(texts))
        return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)


@dataclass
class GuardrailModel:
    builder: GuardrailFeatureBuilder
    booster: xgb.Booster
    feature_names: list[str]
    threshold: float = 0.5
    metrics: dict = field(default_factory=dict)

    def score_frame(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame.reindex(columns=self.feature_names).to_numpy(dtype=np.float32)
        return self.booster.inplace_predict(x)

    def score_many(self, texts: Sequence[str]) -> np.ndarray:
        return self.score_frame(self.builder.transform(texts))

    def score(self, text: str) -> float:
        return float(self.score_many([text])[0])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(path / "booster.json")
        joblib.dump(self.builder.embedder, path / "embedder.joblib")
        meta = {
            "feature_names": self.feature_names,
            "threshold": self.threshold,
            "use_jev": self.builder.use_jev,
            "questions": [q.to_dict() for q in self.builder.questions],
            "metrics": self.metrics,
        }
        (path / "meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, path: str | Path, jev: JevClient | None = None) -> GuardrailModel:
        path = Path(path)
        meta = json.loads((path / "meta.json").read_text())
        booster = xgb.Booster()
        booster.load_model(path / "booster.json")
        questions = [Question(q["id"], q["body"], q["type"]) for q in meta["questions"]]
        builder = GuardrailFeatureBuilder(
            jev, joblib.load(path / "embedder.joblib"), questions, meta["use_jev"]
        )
        return cls(builder, booster, meta["feature_names"], meta["threshold"], meta["metrics"])


def threshold_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    neg = np.sort(scores[y == 0])
    if len(neg) == 0:
        return 0.5
    idx = min(len(neg) - 1, int(np.ceil((1 - target_fpr) * len(neg))) - 1)
    return float(min(0.99, max(neg[max(idx, 0)] + 1e-6, 0.01)))


def classification_metrics(y: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "pr_auc": float(average_precision_score(y, scores)),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "fpr": fp / max(fp + tn, 1),
        "threshold": threshold,
    }


def train_guardrail(
    texts: Sequence[str],
    labels: Sequence[int],
    jev: JevClient | None = None,
    use_jev: bool = True,
    jev_dropout: float = 0.15,
    target_fpr: float = 0.01,
    num_boost_round: int = 300,
    params: dict | None = None,
    seed: int = 0,
) -> GuardrailModel:
    texts = list(texts)
    y = np.asarray(labels, dtype=np.int32)
    idx_tr, idx_va = train_test_split(
        np.arange(len(texts)), test_size=0.25, stratify=y, random_state=seed
    )
    builder = GuardrailFeatureBuilder(jev=jev, use_jev=use_jev)
    builder.fit([texts[i] for i in idx_tr])
    frame = builder.transform(texts)
    x_tr, x_va = frame.iloc[idx_tr].copy(), frame.iloc[idx_va]
    if use_jev and jev_dropout > 0:
        rng = np.random.default_rng(seed)
        drop = rng.random(len(x_tr)) < jev_dropout
        x_tr.loc[drop, builder.jev_columns] = np.nan

    p = {**DEFAULT_PARAMS, **(params or {}), "seed": seed}
    p["scale_pos_weight"] = float((y[idx_tr] == 0).sum() / max((y[idx_tr] == 1).sum(), 1))
    dtr = xgb.DMatrix(x_tr, label=y[idx_tr])
    dva = xgb.DMatrix(x_va, label=y[idx_va])
    booster = xgb.train(
        p,
        dtr,
        num_boost_round,
        evals=[(dva, "valid")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    booster = booster[: booster.best_iteration + 1]
    model = GuardrailModel(builder, booster, list(frame.columns))
    va_scores = model.score_frame(x_va)
    model.threshold = threshold_at_fpr(y[idx_va], va_scores, target_fpr)
    model.metrics = classification_metrics(y[idx_va], va_scores, model.threshold)
    return model
