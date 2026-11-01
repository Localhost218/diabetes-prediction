"""
Tác tử Dự đoán (nhánh DB → ML/DL → Prediction / Classification trên sơ đồ)
  - ML (Chương 2): Gradient Boosting
  - DL: mạng nơ-ron MLP (PyTorch)
  Cả hai huấn luyện trên cùng dữ liệu từ PostgreSQL (mặc định NHANES 2011–2018); kết quả là trung bình của hai mô hình,
  độ lệch giữa chúng là căn cứ để hội đồng chất vấn.
  - RNN Temporal (Chương 4): chỉ chạy khi có chuỗi đo glucose (≥ 5 lần)
  - CNN Vision (Chương 3): chỉ chạy khi có đặc trưng ảnh đáy mắt
Không tự sinh dữ liệu giả cho CNN/RNN khi người dùng không cung cấp.
"""

import asyncio
from typing import Any, Dict

from src.communication.message_bus import AgentMessage, BaseAgent
from src.config import settings
from src.models.cnn_vision import CNNRetinopathyEngine
from src.models.datasets import DATASETS
from src.models.deep_mlp import MLPRisk
from src.models.rnn_temporal import RNNTemporalEngine
from src.models.tabular_ml import GradientBoostingRisk, risk_report
from src.schemas.patient_profile import PatientProfile

DISAGREEMENT_THRESHOLD = 0.15


class PredictionAgent(BaseAgent):
    name = "PredictionAgent"
    handles = {"PREDICT_RISK", "CHALLENGE_ML_VS_LABS", "CHALLENGE_DATA_QUALITY", "CHALLENGE_MODEL_DISAGREEMENT"}

    def __init__(self, bus):
        super().__init__(bus)
        self.spec = DATASETS[settings.prediction_dataset]
        try:
            self.gbdt, self.mlp = GradientBoostingRisk.load(self.spec), MLPRisk.load(self.spec)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Chưa có mô hình cho bộ dữ liệu '{self.spec.name}'. "
                                    f"Chạy: python scripts/train_model.py --dataset {self.spec.name}") from exc
        self.cnn_vision = CNNRetinopathyEngine()
        self.rnn_temporal = RNNTemporalEngine()

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        if msg.msg_type == "PREDICT_RISK":
            return self.reply(msg, "RISK_PREDICTION", await asyncio.to_thread(self._predict, msg.payload))
        handler = {"CHALLENGE_ML_VS_LABS": self._defend_ml_vs_labs, "CHALLENGE_DATA_QUALITY": self._defend_data_quality,
                   "CHALLENGE_MODEL_DISAGREEMENT": self._explain_disagreement}[msg.msg_type]
        return self.reply(msg, "CHALLENGE_RESPONSE", handler(msg.payload), "RESPONSE")

    # ------------------------------------------------------------------
    def _predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile = PatientProfile.model_validate(payload["profile"])
        result: Dict[str, Any] = {"risk_model": self._ensemble(profile)}

        if profile.fundus_features:
            result["cnn_vision"] = self.cnn_vision.analyze_fundus(profile.fundus_features)
        else:
            result["cnn_vision"] = {"skipped": True, "reason": "Không có dữ liệu ảnh đáy mắt"}

        if profile.glucose_history and len(profile.glucose_history) >= 5:
            rnn = self.rnn_temporal.analyze_glucose_trajectory(profile.glucose_history)
            rnn["forecast_note"] = "Dự báo điểm kế tiếp từ RNN chưa huấn luyện, chỉ mang tính minh họa"
            result["rnn_temporal"] = rnn
        else:
            result["rnn_temporal"] = {"skipped": True, "reason": "Không có chuỗi đo glucose (cần ≥ 5 lần đo)"}
        return result

    def _ensemble(self, profile: PatientProfile) -> Dict[str, Any]:
        ml, dl = risk_report(self.gbdt, profile), risk_report(self.mlp, profile)
        prob = (ml["risk_probability"] + dl["risk_probability"]) / 2

        # Gộp giải thích của hai mô hình: trung bình mức đóng góp của từng đặc trưng
        impacts: Dict[str, Dict[str, Any]] = {}
        for report in (ml, dl):
            for f in report["top_risk_factors"]:
                entry = impacts.setdefault(f["feature"], {**f, "impact": 0.0})
                entry["impact"] += f["impact"] / 2
        factors = sorted(impacts.values(), key=lambda f: abs(f["impact"]), reverse=True)
        for f in factors:
            f["impact"] = round(f["impact"], 4)
            f["direction"] = "RISK_UP" if f["impact"] > 0 else "PROTECTIVE"

        return {
            "risk_probability": round(prob, 4),
            "relative_risk": round(prob / self.gbdt.base_rate, 2),
            "population_base_rate": round(self.gbdt.base_rate, 4),
            "top_risk_factors": [f for f in factors if abs(f["impact"]) >= 0.005][:6],
            "imputed_features": ml["imputed_features"],
            "data_completeness": ml["data_completeness"],
            "dataset": self.spec.title,
            "models": {"ml": ml, "dl": dl},
            "disagreement": round(abs(ml["risk_probability"] - dl["risk_probability"]), 4),
            "model_metrics": {"ml": self.gbdt.metrics, "dl": self.mlp.metrics},
        }

    # ------------------------------------------------------------------ trả lời chất vấn của hội đồng
    def _defend_ml_vs_labs(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rm = payload["risk_model"]
        factors = ", ".join(f["feature_vi"] for f in rm["top_risk_factors"] if f["direction"] == "RISK_UP")[:200]
        if payload["direction"] == "ml_high_labs_normal":
            return {
                "position": "maintain_long_term_risk",
                "statement": (f"Hai mô hình (ML + DL) ước lượng nguy cơ {rm['risk_probability'] * 100:.0f}% (gấp "
                              f"{rm['relative_risk']} lần trung bình) dựa trên yếu tố nguy cơ: {factors or 'tổ hợp nhiều yếu tố'}. "
                              "Mô hình không dùng glucose/HbA1c nên phản ánh nguy cơ tiến triển, không phải chẩn đoán hiện tại. "
                              "Đồng ý xét nghiệm là căn cứ chẩn đoán; đề nghị giữ khuyến cáo tầm soát định kỳ."),
                "concede_diagnosis_to_labs": True,
            }
        return {
            "position": "concede",
            "statement": ("Hai mô hình chỉ dựa trên yếu tố nguy cơ và cho nguy cơ thấp, nhưng xét nghiệm đường huyết đã vượt "
                          "ngưỡng chẩn đoán. Xét nghiệm là tiêu chuẩn vàng — nhánh dự đoán rút lại nhận định nguy cơ thấp."),
            "concede_diagnosis_to_labs": True,
        }

    def _defend_data_quality(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rm = payload["risk_model"]
        completeness = rm["data_completeness"]
        adjusted = round(min(0.9, 0.35 + completeness), 2)
        missing = ", ".join(self.spec.feature_vi[f] for f in rm["imputed_features"][:5])
        return {
            "position": "lower_confidence",
            "statement": (f"Mới có {completeness * 100:.0f}% đặc trưng (thiếu: {missing}). Mô hình được huấn luyện để dự đoán "
                          f"khi thiếu dữ liệu (AUC khi thiếu ≈ {rm['model_metrics']['ml'].get('partial_data_auc', 'n/a')}), "
                          f"nhưng hạ độ tin cậy xuống {adjusted}."),
            "adjusted_confidence": adjusted,
            "imputed_features": rm["imputed_features"],
        }

    def _explain_disagreement(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        rm = payload["risk_model"]
        ml, dl = rm["models"]["ml"], rm["models"]["dl"]
        ml_imp = {f["feature"]: f["impact"] for f in ml["top_risk_factors"]}
        dl_imp = {f["feature"]: f["impact"] for f in dl["top_risk_factors"]}
        diffs = sorted(set(ml_imp) | set(dl_imp), key=lambda f: abs(ml_imp.get(f, 0) - dl_imp.get(f, 0)), reverse=True)
        drivers = [f"{self.spec.feature_vi[f]} (ML {ml_imp.get(f, 0) * 100:+.1f}%, DL {dl_imp.get(f, 0) * 100:+.1f}%)"
                   for f in diffs[:2]]
        better = "DL" if self.mlp.metrics.get("roc_auc", 0) >= self.gbdt.metrics.get("roc_auc", 0) else "ML"
        return {
            "position": "use_ensemble_lower_confidence",
            "statement": (f"ML (Gradient Boosting) cho {ml['risk_probability'] * 100:.0f}%, DL (MLP) cho "
                          f"{dl['risk_probability'] * 100:.0f}%. Khác biệt chủ yếu ở cách hai mô hình đánh giá: "
                          f"{'; '.join(drivers) or 'tương tác giữa các yếu tố'}. Ca này nằm ở vùng hai mô hình học khác nhau "
                          f"(thường do dữ liệu thiếu hoặc tổ hợp yếu tố hiếm) → dùng trung bình "
                          f"{rm['risk_probability'] * 100:.0f}% và hạ độ tin cậy; trên tập kiểm tra {better} có AUC cao hơn."),
            "adjusted_confidence": 0.6,
        }
