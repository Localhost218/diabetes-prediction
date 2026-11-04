"""
Phạm vi áp dụng của hệ thống sàng lọc và các câu hỏi làm rõ.

Mô hình dự đoán và ngưỡng ADA chỉ đúng với người lớn, không mang thai, chưa được chẩn đoán, có xét nghiệm
phản ánh đúng đường huyết nền. Module này phát hiện (bằng luật, có thể kiểm thử) khi một hồ sơ:
  - CHƯA RÕ có nằm trong phạm vi hay không → sinh câu hỏi làm rõ cho người dùng;
  - ĐÃ RÕ là nằm ngoài phạm vi → sinh ghi chú (-) / cảnh báo cho hội đồng.
LLM không quyết định khi nào hỏi ở tầng này; LLM chỉ bổ sung các tình huống luật chưa lường trước (ClarificationAgent).
"""

from typing import Any, Dict, List, Tuple

from src.knowledge.clinical_criteria import evaluate_glycemia
from src.schemas.patient_profile import FIELD_SPECS, GLYCEMIC_FIELDS, PatientProfile, format_value, source_vi

MAX_QUESTIONS_PER_TURN = 2


def _question(key: str, question: str, reason: str, priority: int, blocking: bool = False,
              field: str = None) -> Dict[str, Any]:
    return {"key": key, "field": field, "question": question, "reason": reason, "priority": priority,
            "blocking": blocking, "source": "rule"}


def rule_questions(profile: PatientProfile) -> List[Dict[str, Any]]:
    """Câu hỏi làm rõ theo luật, chưa lọc theo các câu đã hỏi. priority nhỏ = quan trọng hơn."""
    g = profile.get
    gly = evaluate_glycemia(profile)
    status = gly["status"]
    questions: List[Dict[str, Any]] = []

    def ask_field(field: str, reason: str, priority: int, blocking: bool = False) -> None:
        if not profile.has(field):
            questions.append(_question(field, FIELD_SPECS[field].question_vi, reason, priority, blocking, field))

    # Thai phụ: ngưỡng chẩn đoán khác hẳn → phải biết trước khi kết luận
    age = g("age_years")
    if g("sex") == 0 and age is not None and 15 <= age <= 49 and status in ("prediabetes_range", "diabetes_range"):
        ask_field("pregnant", "Ngưỡng chẩn đoán ở thai phụ (đái tháo đường thai kỳ) khác người thường; "
                              "mô hình không được huấn luyện cho thai phụ.", 0, blocking=True)

    # Mâu thuẫn giữa hai nguồn
    for c in profile.conflicts:
        questions.append(_question(
            f"conflict:{c.field}:{c.incoming.value}",
            f"{FIELD_SPECS[c.field].label_vi} có hai giá trị: {format_value(c.field, c.existing.value)} "
            f"({source_vi(c.existing.source)}) và {format_value(c.field, c.incoming.value)} "
            f"({source_vi(c.incoming.source)}). Giá trị nào đúng?",
            "Hai nguồn dữ liệu không thống nhất.", 1, field=c.field))

    for f in gly["findings"]:
        flags = f.get("flags") or []
        if "unit_inferred" in flags:
            questions.append(_question(
                f"unit:{f['field']}:{f['value']:g}",
                f"{f['test']} \"{f.get('evidence') or f['value']}\" bạn đo theo đơn vị nào (mmol/L hay mg/dL)?",
                f"Không ghi đơn vị, hệ thống đang hiểu là {f['value']:g} {f['unit']}.", 1, field=f["field"]))
        if "fasting_unclear" in flags:
            ask_field("fasting_confirmed", "Chưa rõ glucose đo lúc đói hay sau ăn — ngưỡng chẩn đoán khác nhau.",
                      1 if f["value"] >= 100 else 3)

    # Xét nghiệm ở ngưỡng ĐTĐ: có thể người dùng đã được chẩn đoán → khi đó cần theo dõi điều trị, không phải sàng lọc
    if status == "diabetes_range":
        ask_field("diagnosed_diabetes", "Xét nghiệm ở ngưỡng đái tháo đường; nếu đã được chẩn đoán thì mục tiêu là "
                                        "kiểm soát, không phải sàng lọc.", 2)

    # HbA1c và glucose lệch nhau → tìm nguyên nhân làm sai lệch
    rank = {"normal": 0, "prediabetes_range": 1, "diabetes_range": 2}
    a1c = next((f for f in gly["findings"] if f["field"] == "hba1c_percent"), None)
    glucose = [f for f in gly["findings"] if f["field"] in GLYCEMIC_FIELDS and f["field"] != "hba1c_percent"]
    if a1c and glucose:
        gap = rank[a1c["status"]] - max(rank[f["status"]] for f in glucose)
        if gap >= 1:
            ask_field("hba1c_interference", "HbA1c cao hơn hẳn so với glucose — thiếu máu, bệnh thận, truyền máu "
                                            "có thể làm HbA1c sai lệch.", 3)
        elif gap <= -1:
            ask_field("acute_illness", "Glucose cao hơn hẳn so với HbA1c — ốm cấp, nhiễm trùng, phẫu thuật hoặc "
                                       "corticoid có thể làm glucose tăng tạm thời.", 3)
    if a1c and (g("egfr") or 999) < 30:
        ask_field("hba1c_interference", "eGFR < 30: bệnh thận nặng làm HbA1c kém tin cậy.", 3)

    return questions


def select_questions(profile: PatientProfile, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Bỏ câu đã hỏi trong phiên và câu trùng; giữ tối đa MAX_QUESTIONS_PER_TURN câu quan trọng nhất."""
    seen, out = set(profile.asked_questions), []
    for q in sorted(candidates, key=lambda q: (not q["blocking"], q["priority"])):
        if q["key"] in seen:
            continue
        seen.add(q["key"])
        out.append(q)
    return out[:MAX_QUESTIONS_PER_TURN]


def scope_notes(profile: PatientProfile) -> Tuple[List[Dict[str, str]], List[str]]:
    """Hồ sơ đã rõ là nằm ngoài phạm vi giả định của mô hình → (cảnh báo, các điểm (-)) cho hội đồng."""
    g = profile.get
    alerts: List[Dict[str, str]] = []
    minus: List[str] = []
    if g("pregnant"):
        alerts.append({"level": "scope", "message": "Đang mang thai: đái tháo đường thai kỳ có ngưỡng riêng (ADA 2024: "
                                                    "nghiệm pháp 75 g glucose ở tuần 24–28, lúc đói ≥ 92 mg/dL đã bất "
                                                    "thường). Kết quả dưới đây không áp dụng cho thai phụ — cần khám "
                                                    "sản khoa / nội tiết."})
        minus.append("[-] Ngoài phạm vi mô hình: mô hình dự đoán được huấn luyện trên người không mang thai.")
    if g("diagnosed_diabetes"):
        minus.append("[-] Người dùng đã được chẩn đoán đái tháo đường: cần theo dõi mục tiêu điều trị cùng bác sĩ; "
                     "xác suất sàng lọc của mô hình chỉ để tham khảo.")
    if g("glucose_lowering_meds"):
        minus.append("[-] Đang dùng thuốc hạ đường huyết / insulin: chỉ số hiện tại đã chịu tác động của thuốc, "
                     "mô hình và ngưỡng sàng lọc có thể đánh giá thấp mức độ bệnh.")
    if g("steroid_use"):
        minus.append("[-] Đang dùng corticoid: glucose có thể tăng do thuốc, cần bác sĩ đánh giá lại khi ngừng thuốc.")
    if g("acute_illness"):
        minus.append("[-] Xét nghiệm làm lúc đang ốm cấp: glucose có thể tăng tạm thời do stress, nên đo lại khi đã "
                     "khỏi bệnh (HbA1c ít bị ảnh hưởng hơn).")
    if g("hba1c_interference") and profile.has("hba1c_percent"):
        minus.append("[-] Có yếu tố làm HbA1c sai lệch (thiếu máu / bệnh thận / truyền máu): nên dựa vào glucose "
                     "thay vì HbA1c.")
    age = g("age_years")
    if age is not None and age < 18:
        minus.append("[-] Ngoài phạm vi mô hình: mô hình được huấn luyện trên người lớn; ở trẻ em cần phân biệt "
                     "đái tháo đường típ 1 — nên khám chuyên khoa Nhi.")
    return alerts, minus


def apply_yes_no(profile: PatientProfile, text: str) -> List[str]:
    """Câu trả lời cụt "có"/"không" cho đúng MỘT câu hỏi có/không vừa hỏi → ghi vào trường tương ứng."""
    import re
    pending = [k for k in profile.pending_questions if FIELD_SPECS.get(k) and FIELD_SPECS[k].kind == "bool"]
    if len(pending) != 1 or profile.has(pending[0]):
        return []
    t = text.strip().lower().rstrip(".!")
    if re.fullmatch(r"(không|ko|k|khong|chưa|chua|không có|chưa từng|không ạ|chưa ạ)", t):
        value = False
    elif re.fullmatch(r"(có|co|có ạ|đúng|đúng rồi|rồi|vâng|ừ|uh|yes|đã|có đang)", t):
        value = True
    else:
        return []
    profile.set(pending[0], value, "text", 0.9, f"Trả lời câu hỏi: {text.strip()}")
    profile.derive()
    return [pending[0]]
