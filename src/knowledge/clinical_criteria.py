"""
Đối chiếu chỉ số xét nghiệm với tiêu chuẩn chẩn đoán ADA 2024 / QĐ 5481/QĐ-BYT
và ánh xạ hồ sơ bệnh nhân vào các thực thể của Knowledge Graph.
"""

from typing import Any, Dict, List, Optional

from src.schemas.patient_profile import PatientProfile

# (trường, tên xét nghiệm, ngưỡng tiền ĐTĐ, ngưỡng ĐTĐ, đơn vị)
GLYCEMIC_CRITERIA = [
    ("glucose_fasting_mgdl", "Glucose huyết tương lúc đói", 100.0, 126.0, "mg/dL"),
    ("hba1c_percent", "HbA1c", 5.7, 6.5, "%"),
    ("ogtt_2h_mgdl", "Glucose 2 giờ sau OGTT", 140.0, 200.0, "mg/dL"),
    ("glucose_random_mgdl", "Glucose bất kỳ", None, 200.0, "mg/dL"),
]
STATUS_RANK = {"unknown": 0, "normal": 1, "prediabetes_range": 2, "diabetes_range": 3}
STATUS_VI = {"normal": "Bình thường", "prediabetes_range": "Ngưỡng tiền đái tháo đường",
             "diabetes_range": "Ngưỡng đái tháo đường", "unknown": "Chưa có xét nghiệm"}


def evaluate_glycemia(profile: PatientProfile) -> Dict[str, Any]:
    findings: List[Dict[str, Any]] = []
    for field, test, pre, dm, unit in GLYCEMIC_CRITERIA:
        fv = profile.fields.get(field)
        if fv is None:
            continue
        v = float(fv.value)
        if v >= dm:
            status, threshold = "diabetes_range", f"≥ {dm:g} {unit}"
        elif pre is not None and v >= pre:
            status, threshold = "prediabetes_range", f"{pre:g}–{dm:g} {unit}"
        elif field == "glucose_random_mgdl":
            continue  # glucose bất kỳ < 200 không dùng để loại trừ hay chẩn đoán tiền ĐTĐ
        else:
            status, threshold = "normal", f"< {pre:g} {unit}"
        findings.append({
            "field": field, "test": test, "value": v, "unit": unit, "status": status,
            "status_vi": STATUS_VI[status], "threshold": threshold,
            "source": fv.source, "confidence": fv.confidence, "flags": fv.flags, "evidence": fv.evidence,
            "source_doc": "ada-2024-chan-doan" if status == "diabetes_range" else "ada-2024-tien-dtd",
        })

    overall = max((f["status"] for f in findings), key=STATUS_RANK.get, default="unknown")
    if overall == "unknown" and profile.glucose_history:
        mean = sum(profile.glucose_history) / len(profile.glucose_history)
        overall = "diabetes_range" if mean >= 154 else ("prediabetes_range" if mean >= 117 else "normal")
        findings.append({"field": "glucose_history", "test": "Glucose trung bình chuỗi đo", "value": round(mean, 1),
                         "unit": "mg/dL", "status": overall, "status_vi": STATUS_VI[overall],
                         "threshold": "ước tính từ HbA1c tương đương (eAG)", "source": profile.glucose_history_source,
                         "confidence": 0.6, "source_doc": "cgm-ha-duong-huyet"})

    diabetic_tests = [f for f in findings if f["status"] == "diabetes_range" and f["field"] != "glucose_history"]
    return {
        "status": overall,
        "status_vi": STATUS_VI[overall],
        "findings": findings,
        "abnormal_diabetic_tests": len(diabetic_tests),
        # ADA: không có tăng đường huyết rõ thì cần 2 kết quả bất thường mới chẩn đoán xác định
        "needs_confirmation": overall == "diabetes_range" and len(diabetic_tests) < 2
                              and not _unequivocal_hyperglycemia(profile),
    }


def _unequivocal_hyperglycemia(profile: PatientProfile) -> bool:
    return (profile.get("glucose_random_mgdl") or 0) >= 200 and (profile.get("glucose_fasting_mgdl") or 0) >= 126 \
        or (profile.get("glucose_fasting_mgdl") or 0) >= 300


def is_central_obesity(profile: PatientProfile) -> Optional[bool]:
    waist, sex = profile.get("waist_cm"), profile.get("sex")
    if waist is None:
        return None
    return waist >= (90 if sex == 1 else 80) if sex is not None else waist >= 90


def graph_entities(profile: PatientProfile, glycemic_status: str) -> List[str]:
    """Các node trong Knowledge Graph tương ứng với tình trạng của bệnh nhân (điểm xuất phát của GraphRAG)."""
    entities = []
    if glycemic_status == "diabetes_range":
        entities.append("Đái tháo đường típ 2")
    elif glycemic_status == "prediabetes_range":
        entities.append("Tiền đái tháo đường")
    if (profile.get("bmi") or 0) >= 23:
        entities.append("Thừa cân/béo phì")
    if is_central_obesity(profile):
        entities.append("Béo bụng")
    checks = [("family_history", "Tiền sử gia đình ĐTĐ"), ("high_bp", "Tăng huyết áp"),
              ("high_chol", "Rối loạn lipid máu"), ("smoker", "Hút thuốc lá")]
    entities += [node for field, node in checks if profile.get(field)]
    if profile.get("phys_activity") is False:
        entities.append("Ít vận động")
    if profile.get("heart_disease") or profile.get("stroke"):
        entities.append("Tiền sử bệnh tim mạch")
    if (profile.get("age_years") or 0) >= 35:
        entities.append("Tuổi ≥ 35")
    if profile.get("egfr") is not None and profile.get("egfr") < 30:
        entities.append("Suy thận nặng (eGFR < 30)")
    return entities
