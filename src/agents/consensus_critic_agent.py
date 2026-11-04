"""
Tác tử Hội chẩn & Phản biện (Consensus & Critic Debate Agent)
Hiện thực hóa khối vòng tròn lớn có các ký hiệu (+ / -) trên sơ đồ của thầy:
  - Dấu (+): bằng chứng tương hỗ, đồng thuận giữa nhánh ML và nhánh tri thức.
  - Dấu (-): mâu thuẫn, điểm cần thận trọng.
  - Tranh biện nhiều vòng: khi phát hiện mâu thuẫn, tác tử gửi CHALLENGE qua MessageBus tới tác tử liên quan,
    nhận RESPONSE và cập nhật kết luận; lặp đến khi không còn vấn đề mới hoặc hết MAX_DEBATE_ROUNDS.
  - Safety Guardrail: các ngưỡng nguy hiểm luôn được ưu tiên, bất kể kết quả mô hình.
"""

import asyncio
from typing import Any, Dict, List, Tuple

from src.communication.message_bus import AgentMessage, BaseAgent
from src.config import settings
from src.knowledge.scope import scope_notes
from src.schemas.patient_profile import FIELD_SPECS, PatientProfile, format_value, source_vi

ML_HIGH, ML_LOW = 0.30, 0.15
ML_DL_DISAGREEMENT = 0.15  # lệch xác suất giữa ML và DL đủ lớn để hội đồng chất vấn
LEVELS = {
    "very_high": "Rất cao — xét nghiệm ở ngưỡng đái tháo đường",
    "high": "Cao — tiền đái tháo đường hoặc nhiều yếu tố nguy cơ",
    "moderate": "Trung bình — cần theo dõi và tầm soát định kỳ",
    "low": "Thấp — duy trì lối sống lành mạnh",
}


class ConsensusCriticAgent(BaseAgent):
    name = "ConsensusCriticAgent"
    handles = {"DELIBERATE"}

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        profile = PatientProfile.model_validate(msg.payload["profile"])
        prediction, knowledge = msg.payload["prediction"], msg.payload["knowledge"]
        state = _initial_state(profile, prediction, knowledge)

        # ---------------- tranh biện nhiều vòng ----------------
        debate: List[Dict[str, Any]] = []
        raised: set = set()
        for round_no in range(1, settings.max_debate_rounds + 1):
            issues = [i for i in _detect_issues(state) if i[0] not in raised]
            if not issues:
                break
            raised.update(i[0] for i in issues)
            replies = await asyncio.gather(*(
                self.ask(receiver, msg_type, payload, msg.conversation_id, performative="CHALLENGE")
                for _, receiver, msg_type, payload in issues
            ))
            for (issue_id, receiver, msg_type, _), reply in zip(issues, replies):
                debate.append({"round": round_no, "issue": issue_id, "challenged": receiver,
                               "challenge": msg_type, "performative": reply.performative, "response": reply.payload})
                if reply.performative == "RESPONSE":
                    _apply_response(state, issue_id, reply.payload)

        return self.reply(msg, "CONSENSUS_RESULT", _conclude(state, profile, debate))


# ---------------------------------------------------------------------------
def _initial_state(profile: PatientProfile, prediction: Dict, knowledge: Dict) -> Dict[str, Any]:
    ml = prediction["risk_model"]
    return {
        "profile": profile, "prediction": prediction, "knowledge": knowledge,
        "ml": ml, "ml_confidence": 0.85, "glycemia": knowledge["glycemia"],
        "plus": [], "minus": [], "confirmation_required": knowledge["glycemia"]["needs_confirmation"],
    }


def _detect_issues(state: Dict[str, Any]) -> List[Tuple[str, str, str, Dict[str, Any]]]:
    """Trả về các vấn đề cần chất vấn: (mã, tác tử bị chất vấn, loại CHALLENGE, payload)."""
    ml, gly = state["ml"], state["glycemia"]
    issues = []
    if ml["data_completeness"] < 0.35:
        issues.append(("data_quality", "PredictionAgent", "CHALLENGE_DATA_QUALITY", {"risk_model": ml}))
    if ml.get("disagreement", 0) >= ML_DL_DISAGREEMENT:
        issues.append(("ml_dl_disagree", "PredictionAgent", "CHALLENGE_MODEL_DISAGREEMENT", {"risk_model": ml}))
    if ml["risk_probability"] >= ML_HIGH and gly["status"] == "normal":
        issues.append(("ml_high_labs_normal", "PredictionAgent", "CHALLENGE_ML_VS_LABS",
                       {"risk_model": ml, "glycemia": gly, "direction": "ml_high_labs_normal"}))
    if ml["risk_probability"] < ML_LOW and gly["status"] == "diabetes_range":
        issues.append(("ml_low_labs_diabetic", "PredictionAgent", "CHALLENGE_ML_VS_LABS",
                       {"risk_model": ml, "glycemia": gly, "direction": "ml_low_labs_diabetic"}))
    if gly["status"] == "diabetes_range":
        issues.append(("verify_diagnostic_criteria", "KnowledgeAgent", "CHALLENGE_VERIFY_CRITERIA", {"glycemia": gly}))
    # Vòng sau: ML giữ quan điểm nguy cơ dài hạn → hỏi nhánh tri thức lịch tầm soát cho người xét nghiệm bình thường
    if state.get("ml_maintains_long_term_risk"):
        issues.append(("screening_plan", "KnowledgeAgent", "CHALLENGE_SCREENING_PLAN",
                       {"glycemia": gly, "risk_factors": state["knowledge"]["graph_rag"]["start_entities"]}))
    return issues


def _apply_response(state: Dict[str, Any], issue: str, response: Dict[str, Any]) -> None:
    if issue == "data_quality":
        state["ml_confidence"] = min(state["ml_confidence"], response["adjusted_confidence"])
        state["minus"].append(f"[-] Dữ liệu chưa đầy đủ: {response['statement']}")
    elif issue == "ml_dl_disagree":
        state["ml_confidence"] = min(state["ml_confidence"], response["adjusted_confidence"])
        state["minus"].append(f"[-] Mô hình ML và DL bất đồng. Giải trình của PredictionAgent: {response['statement']}")
    elif issue == "ml_high_labs_normal":
        state["minus"].append(f"[-] ML cảnh báo nguy cơ cao nhưng xét nghiệm hiện bình thường. Phản biện của PredictionAgent: "
                              f"{response['statement']}")
        state["ml_maintains_long_term_risk"] = response["position"] == "maintain_long_term_risk"
    elif issue == "screening_plan":
        state["plus"].append(f"[+] Thống nhất kế hoạch theo dõi: {response['statement']}")
        state["screening_plan"] = response
    elif issue == "ml_low_labs_diabetic":
        state["minus"].append(f"[-] ML cho nguy cơ thấp nhưng xét nghiệm vượt ngưỡng đái tháo đường. {response['statement']}")
    elif issue == "verify_diagnostic_criteria":
        state["confirmation_required"] = response["position"] == "confirmation_required"
        sign = "[-]" if state["confirmation_required"] else "[+]"
        state["minus" if state["confirmation_required"] else "plus"].append(f"{sign} {response['statement']}")


def _guardrails(profile: PatientProfile) -> List[Dict[str, str]]:
    g = profile.get
    alerts = []
    worst_glucose = max(g("glucose_fasting_mgdl") or 0, g("glucose_random_mgdl") or 0, g("ogtt_2h_mgdl") or 0)
    if worst_glucose >= 600:
        alerts.append({"level": "emergency", "message": f"Glucose {worst_glucose:g} mg/dL — nguy cơ tăng áp lực thẩm thấu, "
                                                         "cần đến cơ sở y tế cấp cứu ngay."})
    elif worst_glucose >= 300 or (g("hba1c_percent") or 0) >= 10:
        alerts.append({"level": "urgent", "message": "Đường huyết rất cao (glucose ≥ 300 mg/dL hoặc HbA1c ≥ 10%) — cần khám "
                                                      "chuyên khoa Nội tiết sớm, có thể cần điều trị insulin."})
    history = profile.glucose_history or []
    if history and min(history) < 54:
        alerts.append({"level": "urgent", "message": f"Có lần đo glucose {min(history):g} mg/dL (< 54) — hạ đường huyết mức 2, "
                                                      "cần trao đổi với bác sĩ về điều chỉnh thuốc."})
    if (g("bmi") or 0) >= 25 and g("high_bp") and (g("family_history") or g("heart_disease")):
        alerts.append({"level": "priority", "message": "Nhóm nguy cơ đỏ: béo phì (BMI ≥ 25, chuẩn châu Á) + tăng huyết áp + "
                                                        "tiền sử gia đình/tim mạch — ưu tiên xét nghiệm và kiểm soát yếu tố tim mạch."})
    return alerts


def _conclude(state: Dict[str, Any], profile: PatientProfile, debate: List[Dict[str, Any]]) -> Dict[str, Any]:
    ml, gly, knowledge = state["ml"], state["glycemia"], state["knowledge"]
    prob, status = ml["risk_probability"], gly["status"]
    plus, minus = state["plus"], state["minus"]

    # (+) bằng chứng xét nghiệm theo tiêu chuẩn
    for f in gly["findings"]:
        if f["status"] in ("diabetes_range", "prediabetes_range"):
            plus.insert(0, f"[+] {f['test']} = {f['value']:g} {f['unit']}: {f['status_vi'].lower()} ({f['threshold']}, ADA 2024).")
    # (+) đồng thuận giữa hai mô hình của nhánh dự đoán
    models = ml.get("models")
    if models and ml["disagreement"] < 0.05:
        plus.append(f"[+] Hai mô hình đồng thuận: ML (Gradient Boosting) {models['ml']['risk_probability'] * 100:.0f}% · "
                    f"DL (MLP) {models['dl']['risk_probability'] * 100:.0f}%.")
    # (+) đồng thuận giữa nhánh dự đoán và nhánh tri thức
    if prob >= ML_HIGH and status in ("diabetes_range", "prediabetes_range"):
        plus.append(f"[+] Đồng thuận: mô hình dự đoán ước lượng nguy cơ {prob * 100:.0f}% và xét nghiệm cũng ở ngưỡng bất thường.")
    if prob < ML_LOW and status == "normal":
        plus.append(f"[+] Đồng thuận: mô hình dự đoán ước lượng nguy cơ thấp ({prob * 100:.0f}%) và xét nghiệm bình thường.")
    up = [f["feature_vi"] for f in ml["top_risk_factors"] if f["direction"] == "RISK_UP"][:3]
    if up and prob >= ML_LOW:
        plus.append(f"[+] Yếu tố làm tăng nguy cơ nhiều nhất theo mô hình: {', '.join(up)}.")
    entities = knowledge["graph_rag"]["start_entities"]
    risk_nodes = [e for e in entities if e not in ("Đái tháo đường típ 2", "Tiền đái tháo đường")]
    if len(risk_nodes) >= 2:
        plus.append(f"[+] Đồ thị tri thức khớp {len(risk_nodes)} yếu tố nguy cơ theo ADA: {', '.join(risk_nodes)}.")

    # (-) chất lượng dữ liệu đầu vào
    for c in profile.conflicts:
        minus.append(f"[-] {FIELD_SPECS[c.field].label_vi} không thống nhất: {format_value(c.field, c.existing.value)} "
                     f"({source_vi(c.existing.source)}) và {format_value(c.field, c.incoming.value)} "
                     f"({source_vi(c.incoming.source)}) — đang dùng giá trị sau, cần xác nhận lại.")
    for f in gly["findings"]:
        flags = f.get("flags", [])
        if "fasting_unclear" in flags:
            minus.append(f"[-] Chưa rõ glucose {f['value']:g} mg/dL có đo lúc đói hay không — kết luận có thể thay đổi.")
        if "unit_inferred" in flags:
            minus.append(f"[-] {f['test']} không ghi đơn vị (\"{f.get('evidence') or f['value']}\"), hệ thống hiểu là "
                         f"{f['value']:g} {f['unit']} — hãy xác nhận lại đơn vị (mmol/L hay mg/dL).")
    rnn = state["prediction"].get("rnn_temporal", {})
    if not rnn.get("skipped") and rnn.get("tbr_percentage", 0) > 4:
        minus.append(f"[-] Chuỗi đo có {rnn['tbr_percentage']}% thời gian dưới 70 mg/dL (mục tiêu < 4%) — nguy cơ hạ đường huyết.")

    # Phân tầng: xét nghiệm là căn cứ chính, ML bổ sung nguy cơ dài hạn
    if status == "diabetes_range":
        level, score = "very_high", max(prob, 0.9)
    elif status == "prediabetes_range":
        level, score = "high", max(prob, 0.6)
    elif prob >= ML_HIGH:
        level, score = "high", prob
    elif prob >= ML_LOW:
        level, score = "moderate", prob
    else:
        level, score = "low", prob

    # Hồ sơ nằm ngoài giả định của mô hình (thai phụ, đã chẩn đoán, đang dùng thuốc...) → ghi (-) và cảnh báo phạm vi
    scope_alerts, scope_minus = scope_notes(profile)
    minus += scope_minus
    guardrails = _guardrails(profile)
    if any(a["level"] in ("emergency", "urgent") for a in guardrails):
        level, score = "very_high", max(score, 0.95)
    elif any(a["level"] == "priority" for a in guardrails) and level in ("low", "moderate"):
        level = "high"

    label = LEVELS[level]
    if status == "diabetes_range":
        label = ("Rất cao — nghi ngờ đái tháo đường, cần xét nghiệm lại để xác nhận" if state["confirmation_required"]
                 else "Rất cao — đủ tiêu chí chẩn đoán đái tháo đường theo xét nghiệm")
    return {
        "risk_level": level,
        "risk_label": label,
        "consensus_risk_score": round(float(score), 3),
        "ml_probability": prob,
        "ml_confidence": state["ml_confidence"],
        "glycemic_status": status,
        "confirmation_required": state["confirmation_required"],
        "positive_evidence_plus": plus,
        "negative_contradictions_minus": minus,
        "safety_guardrails": guardrails + scope_alerts,
        "screening_plan": state.get("screening_plan"),
        "debate_rounds": max((d["round"] for d in debate), default=0),
        "debate": debate,
        "summary": f"Hội đồng ghi nhận {len(plus)} bằng chứng (+) và {len(minus)} điểm cần lưu ý (-) sau "
                   f"{max((d['round'] for d in debate), default=0)} vòng tranh biện. Kết luận: {label}.",
    }
