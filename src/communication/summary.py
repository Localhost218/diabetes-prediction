"""
Tóm tắt một dòng cho mỗi AgentMessage, để giao diện "Phòng hội chẩn" hiển thị như lời phát biểu của tác tử.
Chỉ đọc payload, không gọi LLM; trường nào không có thì bỏ qua.
"""

from typing import Any, Dict

from src.schemas.patient_profile import FIELD_SPECS

MAX_LEN = 180

REQUEST_TEXT = {
    "EXTRACT_FROM_TEXT": "Bóc tách chỉ số và yếu tố nguy cơ từ tin nhắn giúp tôi.",
    "SCOPE_CHECK": "Hồ sơ này có nằm trong phạm vi sàng lọc không?",
    "PLAN_QUESTIONS": "Có cần hỏi lại người dùng điều gì không?",
    "PREDICT_RISK": "Ước lượng nguy cơ từ hồ sơ bằng mô hình ML + DL.",
    "RETRIEVE_EVIDENCE": "Đối chiếu ngưỡng ADA và tra đồ thị tri thức cho hồ sơ này.",
    "ANSWER_QUESTION": "Tìm tài liệu để trả lời câu hỏi kiến thức.",
    "DELIBERATE": "Mở hội đồng: tổng hợp bằng chứng (+/−) và tranh biện.",
    "RECOMMEND_TREATMENT": "Chọn bước can thiệp phù hợp với kết luận của hội đồng.",
    "CHALLENGE_ML_VS_LABS": "Mô hình và xét nghiệm đang nói khác nhau — anh giải thích thế nào?",
    "CHALLENGE_DATA_QUALITY": "Hồ sơ còn thiếu nhiều đặc trưng — dự đoán còn tin được không?",
    "CHALLENGE_MODEL_DISAGREEMENT": "ML và DL lệch nhau nhiều — vì sao?",
    "CHALLENGE_COHORT_DISAGREEMENT": "Mô hình NHANES và mô hình BRFSS 3 triệu người nói ngược nhau — vì sao?",
    "CHALLENGE_VERIFY_CRITERIA": "Xét nghiệm đã đủ tiêu chí chẩn đoán theo ADA chưa?",
    "CHALLENGE_SCREENING_PLAN": "Xét nghiệm bình thường nhưng nguy cơ dài hạn cao — nên tầm soát thế nào?",
}


def _cut(s: str) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= MAX_LEN else s[: MAX_LEN - 1].rstrip() + "…"


def _pct(x: Any) -> str:
    return f"{float(x) * 100:.0f}%"


def summarize(msg: Dict[str, Any]) -> str:
    t, perf, p = msg["msg_type"], msg["performative"], msg.get("payload") or {}
    try:
        if perf == "FAILURE":
            return _cut(f"Lỗi: {p.get('error', t)}")
        if perf in ("REQUEST", "CHALLENGE"):
            if t == "PARSE_DOCUMENT":
                return _cut(f"Đọc tệp bệnh án {p.get('filename') or ''}".strip() + ".")
            if t.startswith("COMPOSE_"):
                return "Viết câu trả lời cho người dùng dựa trên kết quả các tác tử."
            return REQUEST_TEXT.get(t, t)

        # INFORM / RESPONSE
        if "statement" in p:
            return _cut(p["statement"])
        if "profile_delta" in p:
            fields = list((p["profile_delta"] or {}).get("fields", {}))
            labels = [FIELD_SPECS[f].label_vi if f in FIELD_SPECS else f for f in fields]
            return _cut(f"Ghi nhận {len(fields)} chỉ số: {', '.join(labels)}." if fields else "Không thấy chỉ số mới.")
        if "risk_label" in p:
            plus, minus = len(p.get("positive_evidence_plus", [])), len(p.get("negative_contradictions_minus", []))
            return _cut(f"Kết luận: {p['risk_label']} · {plus} (+) / {minus} (−) · {p.get('debate_rounds', 0)} vòng tranh biện.")
        if "risk_model" in p:
            rm = p["risk_model"]
            models = rm.get("models") or {}
            parts = " · ".join(f"{k.upper()} {_pct(v['risk_probability'])}" for k, v in models.items())
            ls = rm.get("large_scale") or {}
            extra = f" BRFSS 3,17 triệu người: gấp {ls['relative_risk']} lần." if ls.get("relative_risk") else ""
            return _cut(f"Nguy cơ theo yếu tố nguy cơ {_pct(rm['risk_probability'])}"
                        + (f" ({parts})" if parts else "") + f", gấp {rm.get('relative_risk')} lần trung bình." + extra)
        if "glycemia" in p:
            g = p["glycemia"]
            paths = len((p.get("graph_rag") or {}).get("paths", []))
            return _cut(f"Đường huyết: {g.get('status_vi', g.get('status'))}; {paths} quan hệ trên đồ thị tri thức.")
        if "passages" in p:
            titles = list(dict.fromkeys(x.get("title", "") for x in p["passages"]))[:2]
            return _cut(f"Tìm được {len(p['passages'])} đoạn tài liệu: {'; '.join(titles)}.")
        if "recommended_action" in p:
            extra = " (đã bị hội đồng giới hạn)" if p.get("overridden_by_consensus") else ""
            return _cut(f"Đề xuất: {p['recommended_action']}{extra}.")
        if "issues" in p:
            n = len(p["issues"])
            return "Hồ sơ nằm trong phạm vi sàng lọc." if n == 0 else f"Phát hiện {n} vấn đề về phạm vi / dữ liệu."
        if "questions" in p:
            n = len(p["questions"])
            return "Không cần hỏi thêm." if n == 0 else _cut(f"Cần hỏi lại {n} câu: " + "; ".join(
                q.get("question", q.get("key", "")) for q in p["questions"][:2]))
        if "text" in p:
            return "Đã viết xong câu trả lời."
        if "filename" in p:
            return _cut(f"Đã đọc {p['filename']}" + (f": lỗi {p['error']}" if p.get("error") else "."))
    except (KeyError, TypeError, ValueError):
        pass
    return t
