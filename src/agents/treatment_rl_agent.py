"""
Tác tử Gợi ý Can thiệp Học Tăng Cường (Treatment RL Agent)
Q-Learning (Off-Policy TD Control) chọn bước can thiệp kết hợp hướng dẫn ADA 2024, Bộ Y tế,
phác đồ cá thể hóa Vinmec và dữ liệu thuốc thực tế từ Nhà thuốc Long Châu.
Kết quả được ràng buộc bởi kết luận hội đồng và tri thức chống chỉ định từ Knowledge Graph.
"""

from typing import Any, Dict, List

from src.communication.message_bus import AgentMessage, BaseAgent
from src.models.qlearning_treatment import ACTION_DESCRIPTIONS, DiabetesQLearningEngine
from src.schemas.patient_profile import PatientProfile


class TreatmentRLAgent(BaseAgent):
    name = "TreatmentRLAgent"
    handles = {"RECOMMEND_TREATMENT"}

    def __init__(self, bus):
        super().__init__(bus)
        self.rl_engine = DiabetesQLearningEngine()

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        profile = PatientProfile.model_validate(msg.payload["profile"])
        consensus, knowledge = msg.payload["consensus"], msg.payload["knowledge"]

        glucose, glucose_basis = _state_glucose(profile)
        bmi = profile.get("bmi", 22.0)
        has_bp = bool(profile.get("high_bp"))
        has_cvd = bool(profile.get("heart_disease") or profile.get("stroke"))

        plan = self.rl_engine.recommend_treatment({
            "Glucose": glucose,
            "BMI": bmi,
            "HighBP": int(has_bp),
            "HeartDiseaseorAttack": int(has_cvd),
        })
        notes: List[str] = [f"Trạng thái MDP tính từ {glucose_basis} (BMI={bmi:.1f})."]

        # Ràng buộc an toàn của hội đồng lên chính sách RL
        action_id = plan["optimal_action_id"]
        if consensus["glycemic_status"] in ("normal", "prediabetes_range", "unknown") and action_id > 0:
            action_id = 0
            notes.append("Hội đồng chưa xác định đái tháo đường → giới hạn ở can thiệp lối sống tích cực.")
            if consensus["glycemic_status"] == "prediabetes_range" and bmi >= 25.0:
                notes.append(
                    "Theo ADA 2024 & Vinmec: Với tiền đái tháo đường kèm thừa cân/béo phì (BMI ≥ 25), "
                    "ngoài giảm 5–7% cân nặng có thể trao đổi với bác sĩ về dự phòng bằng Metformin (Glucophage 500–850mg)."
                )
        if any(a["level"] in ("emergency", "urgent") for a in consensus["safety_guardrails"]):
            action_id = 3
            notes.append("Safety Guardrail kích hoạt → ưu tiên khám chuyên khoa Nội tiết khẩn.")
        if consensus["confirmation_required"] and action_id in (1, 2):
            notes.append(
                f"RL đề xuất \"{ACTION_DESCRIPTIONS[action_id]['name']}\" nhưng chẩn đoán chưa được xác nhận → "
                "trước mắt thay đổi lối sống và xét nghiệm lại (FPG hoặc HbA1c); bác sĩ quyết định việc dùng thuốc."
            )
            action_id = 0

        # Cá thể hóa khuyến nghị thuốc theo trục Tim-Thận & Cân nặng (ADA 2024 + Vinmec + Long Châu)
        if action_id in (1, 2):
            if has_cvd or has_bp:
                notes.append(
                    "Phác đồ Tim – Thận (ADA/Vinmec): Có bệnh đồng mắc huyết áp/tim mạch → ưu tiên phối hợp nhóm ức chế SGLT2 "
                    "(Dapagliflozin/Forxiga 10mg hoặc Empagliflozin/Jardiance 10–25mg) hoặc GLP-1 RA để bảo vệ tim – thận."
                )
            if bmi >= 23.0:
                notes.append(
                    "Trục kiểm soát cân nặng (người châu Á BMI ≥ 23): Ưu tiên thuốc hỗ trợ giảm cân/không gây tăng cân "
                    "(GLP-1 RA, SGLT2i, Metformin, DPP-4i); hạn chế dùng đơn độc Sulfonylurea (Diamicron MR) vì nguy cơ tăng cân và hạ đường huyết."
                )
            if (profile.get("age_years") or 0) >= 65:
                notes.append(
                    "Lưu ý dược lâm sàng cho người ≥ 65 tuổi (Long Châu/Vinmec): Tránh hạ đường huyết, kiểm tra eGFR trước khi dùng Metformin "
                    "(chống chỉ định khi eGFR < 30); có thể cân nhắc Linagliptin (Trajenta 5mg) không cần chỉnh liều theo thận."
                )

        for c in knowledge["graph_rag"]["recommendations"]["contraindications"]:
            notes.append(f"Lưu ý từ đồ thị tri thức: {c}.")

        action = ACTION_DESCRIPTIONS[action_id]
        return self.reply(msg, "TREATMENT_SUGGESTION", {
            "recommended_action": action["name"],
            "intervention_details": action["details"],
            "reference_drugs_vn": action["reference_drugs_vn"],
            "monitoring_plan": action["monitoring_plan"],
            "action_id": action_id,
            "rl_action_id": plan["optimal_action_id"],
            "overridden_by_consensus": action_id != plan["optimal_action_id"],
            "q_values": plan["action_q_distribution"],
            "state_id": plan["state_id"],
            "notes": notes,
            "disclaimer": "Định hướng tham khảo (ADA 2024 / Bộ Y tế / Vinmec / Dược điển Long Châu) để trao đổi với bác sĩ; không tự ý mua và dùng thuốc kê đơn.",
        })


def _state_glucose(profile: PatientProfile):
    if profile.get("glucose_fasting_mgdl") is not None:
        return profile.get("glucose_fasting_mgdl"), "glucose lúc đói"
    if profile.get("hba1c_percent") is not None:
        # eAG (ADAG): glucose trung bình ước tính = 28,7 × HbA1c − 46,7
        return round(28.7 * profile.get("hba1c_percent") - 46.7, 1), "glucose trung bình ước tính từ HbA1c (eAG)"
    if profile.glucose_history:
        return round(sum(profile.glucose_history) / len(profile.glucose_history), 1), "trung bình chuỗi đo glucose"
    return profile.get("glucose_random_mgdl", 100.0), "glucose bất kỳ"
