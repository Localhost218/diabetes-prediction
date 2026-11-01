"""
Tác tử Gợi ý Can thiệp Học Tăng Cường (Treatment RL Agent)
Q-Learning (Off-Policy TD Control) chọn bước can thiệp; kết quả được ràng buộc bởi kết luận hội đồng
và tri thức chống chỉ định từ Knowledge Graph. Đây là định hướng để trao đổi với bác sĩ, không phải đơn thuốc.
"""

from typing import Any, Dict

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
        plan = self.rl_engine.recommend_treatment({
            "Glucose": glucose, "BMI": profile.get("bmi", 22.0),
            "HighBP": int(bool(profile.get("high_bp"))), "HeartDiseaseorAttack": int(bool(profile.get("heart_disease"))),
        })
        notes = [f"Trạng thái MDP tính từ {glucose_basis}."]

        # Ràng buộc an toàn của hội đồng lên chính sách RL
        action_id = plan["optimal_action_id"]
        if consensus["glycemic_status"] in ("normal", "prediabetes_range", "unknown") and action_id > 0:
            action_id = 0
            notes.append("Hội đồng chưa xác định đái tháo đường → giới hạn ở can thiệp lối sống.")
        if any(a["level"] in ("emergency", "urgent") for a in consensus["safety_guardrails"]):
            action_id = 3
            notes.append("Safety Guardrail kích hoạt → ưu tiên khám chuyên khoa.")
        if consensus["confirmation_required"] and action_id in (1, 2):
            notes.append(f"RL đề xuất \"{ACTION_DESCRIPTIONS[action_id]['name']}\" nhưng chẩn đoán chưa được xác nhận → "
                         "trước mắt thay đổi lối sống và xét nghiệm lại (FPG hoặc HbA1c); bác sĩ quyết định việc dùng thuốc.")
            action_id = 0
        for c in knowledge["graph_rag"]["recommendations"]["contraindications"]:
            notes.append(f"Lưu ý từ đồ thị tri thức: {c}.")

        action = ACTION_DESCRIPTIONS[action_id]
        return self.reply(msg, "TREATMENT_SUGGESTION", {
            "recommended_action": action["name"],
            "intervention_details": action["details"],
            "action_id": action_id,
            "rl_action_id": plan["optimal_action_id"],
            "overridden_by_consensus": action_id != plan["optimal_action_id"],
            "q_values": plan["action_q_distribution"],
            "state_id": plan["state_id"],
            "notes": notes,
            "disclaimer": "Định hướng tham khảo để trao đổi với bác sĩ; không tự ý dùng thuốc.",
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
