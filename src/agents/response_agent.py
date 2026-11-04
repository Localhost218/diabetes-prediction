"""
Tác tử Phản hồi (Response Agent)
Viết câu trả lời tiếng Việt từ kết quả của các tác tử. Với LLM: chỉ được dùng dữ liệu trong JSON đầu vào
(grounded generation). Không có LLM: dùng mẫu câu có cấu trúc.
"""

import json
from typing import Any, Dict, List

from src.communication.message_bus import AgentMessage, BaseAgent
from src.llm.client import llm
from src.llm.prompts import CONTEXT_SYSTEM, KNOWLEDGE_SYSTEM, RESPONSE_SYSTEM, SMALL_TALK_SYSTEM, format_history
from src.schemas.patient_profile import FIELD_SPECS

# Từ khóa trong câu hỏi → trường hồ sơ (dùng cho mẫu trả lời khi không có LLM)
FIELD_KEYWORDS = [
    (r"chiều cao|cao bao nhiêu|cao mấy", "Chiều cao"), (r"cân nặng|nặng bao nhiêu|nặng mấy", "Cân nặng"),
    (r"tuổi", "Tuổi"),
    (r"bmi", "BMI"), (r"hba1c|a1c", "HbA1c"), (r"đường huyết|glucose|đường máu", "Đường huyết"),
    (r"huyết áp", "Huyết áp"), (r"vòng eo|eo", "Vòng eo"), (r"mỡ máu|cholesterol", "mỡ"),
]

DISCLAIMER = "_Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ._"


class ResponseAgent(BaseAgent):
    name = "ResponseAgent"
    handles = {"COMPOSE_ASSESSMENT", "COMPOSE_FOLLOW_UP", "COMPOSE_KNOWLEDGE_ANSWER", "COMPOSE_SMALL_TALK",
               "COMPOSE_CONTEXTUAL_ANSWER", "COMPOSE_CLARIFICATION"}

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        p = msg.payload
        history = _history_block(p.get("history"))
        if msg.msg_type == "COMPOSE_ASSESSMENT":
            text = await llm.generate(RESPONSE_SYSTEM, [history + _assessment_brief(p)]) or _assessment_template(p)
        elif msg.msg_type == "COMPOSE_FOLLOW_UP":
            text = _follow_up_template(p)
        elif msg.msg_type == "COMPOSE_CLARIFICATION":
            text = _clarification_template(p)
        elif msg.msg_type == "COMPOSE_CONTEXTUAL_ANSWER":
            text = await llm.generate(CONTEXT_SYSTEM, [history + _contextual_brief(p)]) or _contextual_template(p)
        elif msg.msg_type == "COMPOSE_KNOWLEDGE_ANSWER":
            text = await llm.generate(KNOWLEDGE_SYSTEM, [history + _knowledge_brief(p)]) or _knowledge_template(p)
        else:
            text = await llm.generate(SMALL_TALK_SYSTEM, [history + p["text"]]) or _small_talk_template()
        if "tham khảo" not in text[-300:] and msg.msg_type in ("COMPOSE_ASSESSMENT", "COMPOSE_KNOWLEDGE_ANSWER"):
            text = f"{text}\n\n{DISCLAIMER}"
        if msg.msg_type == "COMPOSE_ASSESSMENT" and p.get("questions"):
            # Câu hỏi làm rõ được gắn nguyên văn (không qua LLM viết lại) để chắc chắn người dùng nhìn thấy
            text = f"{text}\n\n---\n**Để kết quả chính xác hơn, mình cần hỏi thêm:**\n{_questions_block(p['questions'])}"
        return self.reply(msg, "RESPONSE_TEXT", {"text": text, "generated_by": "llm" if llm.enabled else "template"})


# ------------------------------------------------------------------ dữ liệu cho LLM
def _history_block(history) -> str:
    return f"Lịch sử hội thoại gần đây:\n{format_history(history)}\n\n" if history else ""


def _contextual_brief(p: Dict[str, Any]) -> str:
    last = p.get("last_assessment") or {}
    brief = {
        "cau_hoi": p["question"],
        "ho_so_da_ghi_nhan": p["profile_summary"],
        "chi_so_bat_buoc_con_thieu": [FIELD_SPECS[f].label_vi for f in p.get("missing_required", [])],
        "chi_so_nen_bo_sung": [FIELD_SPECS[f].label_vi for f in p.get("missing_recommended", [])],
        "ket_qua_hoi_chan_gan_nhat": {k: last.get(k) for k in ("risk_label", "positive_evidence_plus",
                                                               "negative_contradictions_minus", "safety_guardrails",
                                                               "treatment", "citations") if last.get(k)} or None,
    }
    return "Dữ liệu (JSON):\n" + json.dumps(brief, ensure_ascii=False, indent=1)


def _assessment_brief(p: Dict[str, Any]) -> str:
    c, k, t, pred = p["consensus"], p["knowledge"], p["treatment"], p["prediction"]["risk_model"]
    brief = {
        "ket_luan": c["risk_label"],
        "diem_nguy_co_hoi_dong": c["consensus_risk_score"],
        "xac_suat_du_doan_theo_yeu_to_nguy_co": pred["risk_probability"],
        "xac_suat_tung_mo_hinh": {k: v["risk_probability"] for k, v in pred.get("models", {}).items()},
        "gap_bao_nhieu_lan_trung_binh": pred["relative_risk"],
        "xet_nghiem": [{k2: f[k2] for k2 in ("test", "value", "unit", "status_vi", "threshold")}
                       for f in k["glycemia"]["findings"]],
        "can_xet_nghiem_lai_xac_nhan": c["confirmation_required"],
        "bang_chung_cong": c["positive_evidence_plus"],
        "diem_luu_y_tru": c["negative_contradictions_minus"],
        "canh_bao_an_toan": c["safety_guardrails"],
        "khuyen_cao_tam_soat_tu_do_thi": k["graph_rag"]["recommendations"]["screenings"],
        "can_thiep_tu_do_thi": k["graph_rag"]["recommendations"]["interventions"],
        "goi_y_can_thiep_RL": {"hanh_dong": t["recommended_action"], "chi_tiet": t["intervention_details"],
                               "ghi_chu": t["notes"]},
        "citations": [f"{x['title']} — {x['section']}" for x in k["citations"]],
        "doan_tai_lieu": [f"[{x['title']} — {x['heading']}] {x['content'][:600]}" for x in k["passages"][:4]],
        "missing_recommended": [FIELD_SPECS[f].label_vi for f in p.get("missing_recommended", [])],
        "du_lieu_moi_cap_nhat": p.get("updated_fields", []),
    }
    return "Kết quả hội đồng (JSON):\n" + json.dumps(brief, ensure_ascii=False, indent=1)


def _knowledge_brief(p: Dict[str, Any]) -> str:
    passages = "\n\n".join(f"[Nguồn: {x['title']} — {x['heading']}]\n{x['content']}" for x in p["passages"])
    return f"Câu hỏi: {p['question']}\n\nTài liệu:\n{passages}"


# ------------------------------------------------------------------ mẫu câu dự phòng
def _assessment_template(p: Dict[str, Any]) -> str:
    c, k, t = p["consensus"], p["knowledge"], p["treatment"]
    ml = p["prediction"]["risk_model"]
    lines: List[str] = [f"### Kết quả đánh giá: {c['risk_label']}", ""]
    for a in c["safety_guardrails"]:
        lines.append(f"> ⚠️ **{a['message']}**")
    if c["safety_guardrails"]:
        lines.append("")
    if k["glycemia"]["findings"]:
        lines.append("**Xét nghiệm đường huyết**")
        lines += [f"- {f['test']}: {f['value']:g} {f['unit']} → {f['status_vi']} ({f['threshold']})"
                  for f in k["glycemia"]["findings"]]
        lines.append("")
    models = ml.get("models", {})
    detail = (f" — ML {models['ml']['risk_probability'] * 100:.0f}%, DL {models['dl']['risk_probability'] * 100:.0f}%"
              if models else "")
    lines.append(f"**Mô hình dự đoán (ML + DL):** nguy cơ theo yếu tố nguy cơ {ml['risk_probability'] * 100:.0f}%{detail} "
                 f"(gấp {ml['relative_risk']} lần trung bình quần thể).")
    lines += ["", "**Hội đồng đánh giá**"] + [f"- {x}" for x in c["positive_evidence_plus"] + c["negative_contradictions_minus"]]
    rec = k["graph_rag"]["recommendations"]
    if rec["screenings"] and c["glycemic_status"] == "diabetes_range":
        lines += ["", f"**Tầm soát biến chứng nên làm:** {', '.join(rec['screenings'])}."]
    lines += ["", f"**Gợi ý bước tiếp theo:** {t['recommended_action']} — {t['intervention_details']}"]
    lines += [f"- {n}" for n in t["notes"][1:]]
    missing = [FIELD_SPECS[f].label_vi for f in p.get("missing_recommended", [])]
    if missing:
        lines += ["", f"_Bổ sung thêm {', '.join(missing).lower()} để đánh giá chính xác hơn._"]
    if k["citations"]:
        lines += ["", "**Nguồn:** " + "; ".join(f"{x['title']} ({x['section']})" for x in k["citations"][:4])]
    return "\n".join(lines)


def _follow_up_template(p: Dict[str, Any]) -> str:
    lines = []
    if p.get("understood"):
        lines += ["Mình đã ghi nhận:"] + [f"- {u}" for u in p["understood"]] + [""]
    for d in p.get("documents", []):
        if d.get("error"):
            lines.append(f"⚠️ Không đọc được **{d['filename']}**: {d['error']}")
        elif not d.get("is_medical_document", True):
            lines.append(f"⚠️ **{d['filename']}** có vẻ không phải tài liệu y tế.")
    if lines and lines[-1] != "":
        lines.append("")
    questions = [FIELD_SPECS[f].question_vi for f in p["missing_required"] if FIELD_SPECS[f].question_vi]
    questions += [q["question"] for q in p.get("questions", [])]
    lines.append("Để hội đồng các tác tử đánh giá nguy cơ đái tháo đường, mình cần thêm:")
    lines += [f"{i}. {q}" for i, q in enumerate(questions, 1)]
    lines += ["", "Bạn có thể trả lời bằng tin nhắn hoặc tải lên phiếu xét nghiệm / bệnh án (PDF, ảnh, DOCX)."]
    return "\n".join(lines)


def _clarification_template(p: Dict[str, Any]) -> str:
    lines = []
    if p.get("understood"):
        lines += ["Mình đã ghi nhận:"] + [f"- {u}" for u in p["understood"]] + [""]
    lines += ["Trước khi đánh giá, mình cần hỏi thêm vì câu trả lời sẽ thay đổi cách hiểu kết quả:",
              _questions_block(p["questions"])]
    return "\n".join(lines)


def _questions_block(questions: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{i}. {q['question']}  \n   _Lý do: {q['reason']}_" for i, q in enumerate(questions, 1))


def _contextual_template(p: Dict[str, Any]) -> str:
    import re
    question = p["question"].lower()
    asked = [label for pattern, label in FIELD_KEYWORDS if re.search(pattern, question)]
    lines: List[str] = []
    if asked:
        for label in asked:
            found = [s for s in p["profile_summary"] if label.lower() in s.lower()]
            lines += found or [f"- {label}: bạn **chưa cung cấp** trong cuộc trò chuyện này."]
        lines = [l if l.startswith("-") else f"- {l}" for l in lines]
        return "Theo những gì bạn đã cung cấp:\n" + "\n".join(lines)
    last = p.get("last_assessment")
    lines = ["**Hồ sơ đã ghi nhận:**"] + ([f"- {s}" for s in p["profile_summary"]] or ["- Chưa có dữ liệu."])
    if last:
        lines += ["", f"**Kết quả lần đánh giá gần nhất:** {last['risk_label']}"]
        lines += [f"- {x}" for x in last.get("positive_evidence_plus", []) + last.get("negative_contradictions_minus", [])]
    return "\n".join(lines)


def _knowledge_template(p: Dict[str, Any]) -> str:
    if not p["passages"]:
        return "Cơ sở tri thức hiện chưa có thông tin cho câu hỏi này. Bạn nên hỏi trực tiếp bác sĩ."
    lines = ["Thông tin liên quan trong cơ sở tri thức:", ""]
    for x in p["passages"][:3]:
        lines += [f"**{x['heading']}** _(Nguồn: {x['title']})_", x["content"], ""]
    return "\n".join(lines).strip()


def _small_talk_template() -> str:
    return ("Xin chào, mình là **Diabetes Prediction** — trợ lý sàng lọc nguy cơ đái tháo đường.\n\n"
            "Bạn có thể:\n- Nhập tuổi, chiều cao, cân nặng, đường huyết lúc đói hoặc HbA1c để mình đánh giá nguy cơ\n"
            "- Tải lên phiếu xét nghiệm / bệnh án (PDF, ảnh, DOCX, CSV)\n- Hỏi kiến thức về đái tháo đường")
