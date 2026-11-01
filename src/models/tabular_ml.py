"""
Chương 2: Học Máy (Machine Learning) & Explainable AI (XAI)
Gradient Boosting trên dữ liệu bảng; xử lý trực tiếp giá trị thiếu (NaN) nên dự đoán được khi người dùng
chỉ khai một phần thông tin.

Dùng chung cho cả hai mô hình của nhánh DB → ML/DL:
  - risk_report(): xác suất + giải thích cục bộ bằng che đặc trưng (occlusion): xác suất thay đổi bao nhiêu
    nếu đặc trưng đó chưa được biết.
"""

from datetime import datetime
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src.config import settings
from src.models.datasets import DatasetSpec
from src.schemas.patient_profile import PatientProfile


class GradientBoostingRisk:
    kind = "ML"
    algorithm = "HistGradientBoostingClassifier"

    def __init__(self, spec: DatasetSpec, model: HistGradientBoostingClassifier, base_rate: float,
                 metrics: Dict[str, Any] = None):
        self.spec, self.model, self.base_rate = spec, model, base_rate
        self.metrics = metrics or {}

    @staticmethod
    def path(spec: DatasetSpec):
        return settings.model_dir / f"{spec.name}_gbdt.joblib"

    @classmethod
    def train(cls, spec: DatasetSpec, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingRisk":
        model = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                               l2_regularization=1.0, early_stopping=True, random_state=42)
        model.fit(X[spec.features].astype(float), y)
        return cls(spec, model, float(y.mean()))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.spec.features].astype(float))[:, 1]

    def save(self) -> None:
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": self.model, "base_rate": self.base_rate, "metrics": self.metrics,
                     "saved_at": datetime.now().isoformat(timespec="seconds")}, self.path(self.spec))

    @classmethod
    def load(cls, spec: DatasetSpec) -> "GradientBoostingRisk":
        bundle = joblib.load(cls.path(spec))
        return cls(spec, bundle["model"], bundle["base_rate"], bundle["metrics"])


def risk_report(model, profile: PatientProfile) -> Dict[str, Any]:
    """Dự đoán cho một bệnh nhân + giải thích đặc trưng; dùng được cho cả GBDT (ML) và MLP (DL)."""
    spec = model.spec
    row = spec.profile_to_row(profile)
    known = [f for f in spec.features if row[f] is not None]
    base = pd.DataFrame([{f: (np.nan if v is None else v) for f, v in row.items()}], columns=spec.features)
    prob = float(model.predict_proba(base)[0])

    contributions: List[Dict[str, Any]] = []
    if known:
        occluded = pd.concat([base] * len(known), ignore_index=True)
        for i, f in enumerate(known):
            occluded.loc[i, f] = np.nan
        deltas = prob - model.predict_proba(occluded)
        for f, d in zip(known, deltas):
            if abs(d) >= 0.005:
                contributions.append({"feature": f, "feature_vi": spec.feature_vi[f], "value": row[f],
                                      "impact": round(float(d), 4), "direction": "RISK_UP" if d > 0 else "PROTECTIVE"})
    contributions.sort(key=lambda c: abs(c["impact"]), reverse=True)
    return {
        "model": model.algorithm,
        "kind": model.kind,
        "risk_probability": round(prob, 4),
        "relative_risk": round(prob / model.base_rate, 2),
        "top_risk_factors": contributions[:6],
        "imputed_features": [f for f in spec.features if f not in known],
        "data_completeness": round(len(known) / len(spec.features), 2),
        "model_metrics": model.metrics,
    }
