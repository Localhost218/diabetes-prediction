"""
Quy đổi đơn vị xét nghiệm về đơn vị chuẩn của hệ thống.

Xét nghiệm ở Việt Nam thường dùng mmol/L (glucose, lipid) và µmol/L (creatinin),
trong khi tiêu chuẩn ADA và mô hình dùng mg/dL. Khi không ghi đơn vị, đoán theo độ lớn
của giá trị và hạ độ tin cậy.
"""

from typing import List, Optional, Tuple

GLUCOSE_MMOL_TO_MGDL = 18.016
CHOL_MMOL_TO_MGDL = 38.67
TG_MMOL_TO_MGDL = 88.57
CREATININE_UMOL_TO_MGDL = 1 / 88.4

# (giá trị đã quy đổi, độ tin cậy)
Normalized = Tuple[float, float]


def _unit(unit: Optional[str]) -> str:
    return (unit or "").lower().replace(" ", "").replace("µ", "u").replace("μ", "u")


def glucose_to_mgdl(value: float, unit: Optional[str]) -> Normalized:
    u = _unit(unit)
    if "mmol" in u:
        return round(value * GLUCOSE_MMOL_TO_MGDL, 1), 1.0
    if "mg" in u:
        return round(value, 1), 1.0
    # Không có đơn vị: glucose < 35 gần như chắc chắn là mmol/L
    if value < 35:
        return round(value * GLUCOSE_MMOL_TO_MGDL, 1), 0.8
    return round(value, 1), 0.8


def hba1c_to_percent(value: float, unit: Optional[str]) -> Normalized:
    u = _unit(unit)
    if "mmol" in u or value > 20:
        # IFCC (mmol/mol) -> NGSP (%)
        return round(value / 10.929 + 2.15, 2), 0.95
    return round(value, 2), 1.0


def cholesterol_to_mgdl(value: float, unit: Optional[str]) -> Normalized:
    u = _unit(unit)
    if "mmol" in u:
        return round(value * CHOL_MMOL_TO_MGDL, 1), 1.0
    if "mg" in u:
        return round(value, 1), 1.0
    if value < 20:
        return round(value * CHOL_MMOL_TO_MGDL, 1), 0.8
    return round(value, 1), 0.8


def triglycerides_to_mgdl(value: float, unit: Optional[str]) -> Normalized:
    u = _unit(unit)
    if "mmol" in u:
        return round(value * TG_MMOL_TO_MGDL, 1), 1.0
    if "mg" in u:
        return round(value, 1), 1.0
    if value < 30:
        return round(value * TG_MMOL_TO_MGDL, 1), 0.8
    return round(value, 1), 0.8


def creatinine_to_mgdl(value: float, unit: Optional[str]) -> Normalized:
    u = _unit(unit)
    if "umol" in u or (not u and value > 20):
        return round(value * CREATININE_UMOL_TO_MGDL, 2), 0.95 if u else 0.8
    return round(value, 2), 1.0


NORMALIZERS = {
    "glucose_fasting_mgdl": glucose_to_mgdl,
    "glucose_random_mgdl": glucose_to_mgdl,
    "ogtt_2h_mgdl": glucose_to_mgdl,
    "hba1c_percent": hba1c_to_percent,
    "total_cholesterol_mgdl": cholesterol_to_mgdl,
    "ldl_mgdl": cholesterol_to_mgdl,
    "hdl_mgdl": cholesterol_to_mgdl,
    "triglycerides_mgdl": triglycerides_to_mgdl,
    "creatinine_mgdl": creatinine_to_mgdl,
}


def normalize(field: str, value: float, unit: Optional[str]) -> Normalized:
    fn = NORMALIZERS.get(field)
    return fn(value, unit) if fn else (value, 1.0)


def unit_flags(field: str, unit: Optional[str]) -> List[str]:
    """Đánh dấu khi đơn vị phải đoán theo độ lớn giá trị (ví dụ glucose '6,4' hiểu là mmol/L)."""
    return ["unit_inferred"] if field in NORMALIZERS and not _unit(unit) else []
