"""
Trích xuất chỉ số lâm sàng bằng luật/regex — nhánh dự phòng khi không có LLM.

Dùng chung cho tin nhắn chat (ConversationAgent) và văn bản bệnh án (DocumentAgent).
Chỉ ghi nhận những gì thực sự xuất hiện trong văn bản; không điền giá trị mặc định.
"""

import re
from datetime import date
from typing import Callable, List, Optional, Tuple

from src.document_processing.unit_normalizer import normalize, unit_flags
from src.schemas.patient_profile import PatientProfile

NUM = r"(\d+(?:[.,]\d+)?)"
GAP = r"[^\d\n]{0,30}?"  # chữ chen giữa tên chỉ số và giá trị (": ", "lúc đói là", ...)
NEGATION = re.compile(r"(không|chưa|ko|k|khong|chua|chẳng|hết|bỏ)\s+(\S+\s+){0,3}$")


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def _negated(text: str, start: int) -> bool:
    window = text[max(0, start - 25):start]
    return bool(NEGATION.search(window))


def _evidence(text: str, start: int, end: int) -> str:
    return text[max(0, start - 10):min(len(text), end + 10)].strip().replace("\n", " ")


# ---------------- chỉ số dạng số ----------------

GLUCOSE_RE = re.compile(
    rf"(glucose|glucoza|glu\b|đường huyết|duong huyet|đường máu|đường trong máu|fpg){GAP}{NUM}\s*(mmol\s*/\s*l|mg\s*/\s*dl)?",
    re.I,
)
HBA1C_RE = re.compile(rf"(hba1c|hb\s*a1c|a1c){GAP}{NUM}\s*(%|mmol\s*/\s*mol)?", re.I)
BMI_RE = re.compile(rf"\bbmi{GAP}{NUM}", re.I)
HEIGHT_M_RE = re.compile(r"(?:chiều cao|cao)\s*[:=]?\s*(\d)\s*(?:m|,|\.)\s*(\d{1,2})\b", re.I)
HEIGHT_CM_RE = re.compile(rf"(?:chiều cao|cao){GAP}(\d{{2,3}}(?:[.,]\d+)?)\s*cm", re.I)
WEIGHT_RE = re.compile(rf"(?:cân nặng|nặng|cân){GAP}(\d{{2,3}}(?:[.,]\d+)?)\s*(?:kg|ký|kí|ki\s*lô|cân)", re.I)
WAIST_RE = re.compile(rf"(?:vòng eo|vòng bụng|vong eo){GAP}(\d{{2,3}}(?:[.,]\d+)?)", re.I)
BP_RE = re.compile(r"(?:huyết áp|huyet ap|\bha\b|\bbp\b)[^\d\n]{0,20}?(\d{2,3})\s*/\s*(\d{2,3})", re.I)
AGE_RE = re.compile(r"(\d{1,3})\s*(?:tuổi|tuoi|years?\s*old|\bt\b)|năm nay\s*(\d{1,3})\b", re.I)
BIRTH_YEAR_RE = re.compile(r"(?:năm sinh|sinh năm|ngày sinh)[^\d\n]{0,15}?(?:\d{1,2}[/-]\d{1,2}[/-])?((?:19|20)\d{2})", re.I)
SEX_RE = re.compile(
    r"(?:giới tính|giới|gt|phái)\s*[:\-]?\s*(nam|nữ|nu)\b"
    r"|(?:bệnh nhân|tôi là|là|tuổi\s*,?)\s*(nam|nữ)\b"
    r"|(?:^|[,;]\s*)(nam|nữ)\s*(?=[,;.]|\d|$)",
    re.I | re.M,
)
CHOL_RE = re.compile(rf"(cholesterol|cholesterol toàn phần|cholesterol tp|\btc\b){GAP}{NUM}\s*(mmol\s*/\s*l|mg\s*/\s*dl)?", re.I)
LDL_RE = re.compile(rf"(ldl(?:\s*-\s*c)?){GAP}{NUM}\s*(mmol\s*/\s*l|mg\s*/\s*dl)?", re.I)
HDL_RE = re.compile(rf"(hdl(?:\s*-\s*c)?){GAP}{NUM}\s*(mmol\s*/\s*l|mg\s*/\s*dl)?", re.I)
TG_RE = re.compile(rf"(triglycerid\w*|\btg\b){GAP}{NUM}\s*(mmol\s*/\s*l|mg\s*/\s*dl)?", re.I)
CREAT_RE = re.compile(rf"(creatinin\w*){GAP}{NUM}\s*(u\s*mol\s*/\s*l|µ\s*mol\s*/\s*l|μ\s*mol\s*/\s*l|mg\s*/\s*dl)?", re.I)
EGFR_RE = re.compile(rf"(egfr|mlcT|mức lọc cầu thận){GAP}{NUM}", re.I)


def _glucose_field(context: str) -> Tuple[str, float]:
    """Xác định loại glucose từ ngữ cảnh quanh số đo."""
    c = context.lower()
    if re.search(r"ogtt|nghiệm pháp|dung nạp|2\s*h\b|2\s*giờ", c):
        return "ogtt_2h_mgdl", 0.9
    if re.search(r"bất kỳ|ngẫu nhiên|sau ăn|random", c):
        return "glucose_random_mgdl", 0.9
    if re.search(r"đói|doi|fasting|fpg|sáng sớm|lúc sáng|chưa ăn|trước ăn|nhịn ăn", c):
        return "glucose_fasting_mgdl", 1.0
    # Phiếu xét nghiệm sinh hóa thường lấy máu lúc đói
    return "glucose_fasting_mgdl", 0.7


def extract_measurements(text: str, profile: PatientProfile, source: str) -> None:
    lower = text.lower()

    for m in GLUCOSE_RE.finditer(lower):
        if "hba1c" in lower[max(0, m.start() - 5):m.end()]:
            continue
        context = lower[max(0, m.start() - 15):m.end() + 25]
        field, conf = _glucose_field(context)
        value, unit_conf = normalize(field, _num(m.group(2)), m.group(3))
        flags = unit_flags(field, m.group(3)) + (["fasting_unclear"] if conf < 0.8 else [])
        profile.set(field, value, source, conf * unit_conf, _evidence(text, m.start(), m.end()), flags)

    for m in HBA1C_RE.finditer(lower):
        value, conf = normalize("hba1c_percent", _num(m.group(2)), m.group(3))
        profile.set("hba1c_percent", value, source, conf, _evidence(text, m.start(), m.end()))

    for m in BMI_RE.finditer(lower):
        profile.set("bmi", _num(m.group(1)), source, 1.0, _evidence(text, m.start(), m.end()))

    m = HEIGHT_M_RE.search(lower)
    if m:
        cm = int(m.group(1)) * 100 + int(m.group(2).ljust(2, "0"))
        profile.set("height_cm", cm, source, 0.95, _evidence(text, m.start(), m.end()))
    else:
        m = HEIGHT_CM_RE.search(lower)
        if m:
            profile.set("height_cm", _num(m.group(1)), source, 1.0, _evidence(text, m.start(), m.end()))

    m = WEIGHT_RE.search(lower)
    if m:
        profile.set("weight_kg", _num(m.group(1)), source, 1.0, _evidence(text, m.start(), m.end()))

    m = WAIST_RE.search(lower)
    if m:
        profile.set("waist_cm", _num(m.group(1)), source, 1.0, _evidence(text, m.start(), m.end()))

    m = BP_RE.search(lower)
    if m:
        profile.set("systolic_bp", _num(m.group(1)), source, 1.0, _evidence(text, m.start(), m.end()))
        profile.set("diastolic_bp", _num(m.group(2)), source, 1.0, _evidence(text, m.start(), m.end()))

    m = AGE_RE.search(lower)
    if m:
        profile.set("age_years", int(m.group(1) or m.group(2)), source, 1.0, _evidence(text, m.start(), m.end()))
    else:
        m = BIRTH_YEAR_RE.search(lower)
        if m:
            profile.set("age_years", date.today().year - int(m.group(1)), source, 0.95, _evidence(text, m.start(), m.end()))

    m = SEX_RE.search(lower)
    if m:
        word = m.group(1) or m.group(2) or m.group(3)
        profile.set("sex", "nam" if word == "nam" else "nữ", source, 1.0, _evidence(text, m.start(), m.end()))

    for regex, field in ((CHOL_RE, "total_cholesterol_mgdl"), (LDL_RE, "ldl_mgdl"),
                         (HDL_RE, "hdl_mgdl"), (TG_RE, "triglycerides_mgdl"), (CREAT_RE, "creatinine_mgdl")):
        m = regex.search(lower)
        if m:
            value, conf = normalize(field, _num(m.group(2)), m.group(3))
            profile.set(field, value, source, conf, _evidence(text, m.start(), m.end()), unit_flags(field, m.group(3)))

    m = EGFR_RE.search(lower)
    if m:
        profile.set("egfr", _num(m.group(2)), source, 1.0, _evidence(text, m.start(), m.end()))


# ---------------- yếu tố nguy cơ dạng có/không ----------------

BOOLEAN_PATTERNS: List[Tuple[str, str, Optional[str]]] = [
    # (trường, mẫu khẳng định, mẫu phủ định riêng)
    ("high_bp", r"tăng huyết áp|cao huyết áp|huyết áp cao|tang huyet ap|\btha\b|thuốc (hạ )?huyết áp|thuốc hạ áp", None),
    ("high_chol", r"mỡ máu cao|rối loạn mỡ máu|rối loạn lipid|cholesterol cao|mo mau cao|máu nhiễm mỡ", None),
    ("family_history",
     r"\b(bố|mẹ|cha|ba|má|anh ruột|chị ruột|anh trai|chị gái|em trai|em gái|em ruột|gia đình|nhà tôi|nhà mình)\b"
     r"[^.\n]{0,30}(tiểu đường|đái tháo đường|đtđ|tieu duong)",
     r"(không ai|ko ai|chưa ai)[^.\n]{0,30}(tiểu đường|đái tháo đường|đtđ)"),
    ("smoker", r"hút thuốc|thuốc lá|hut thuoc", r"(không|ko|chưa|đã bỏ|bỏ)\s+(hút\s+)?thuốc"),
    ("heart_disease", r"bệnh tim|nhồi máu cơ tim|bệnh mạch vành|suy tim|thiếu máu cơ tim", None),
    ("stroke", r"đột quỵ|tai biến mạch máu não|tai biến", None),
    ("heavy_alcohol", r"uống nhiều rượu|nhậu nhiều|uống rượu bia nhiều|nghiện rượu", None),
]

ACTIVITY_POS = r"tập thể dục|chạy bộ|đi bộ|tập gym|chơi thể thao|đạp xe|bơi"
ACTIVITY_NEG = r"ít vận động|lười vận động|không tập|ít tập|ngồi nhiều|không vận động"


def extract_risk_factors(text: str, profile: PatientProfile, source: str) -> None:
    lower = text.lower()
    for field, pos, neg in BOOLEAN_PATTERNS:
        if neg and re.search(neg, lower):
            m = re.search(neg, lower)
            profile.set(field, False, source, 0.9, _evidence(text, m.start(), m.end()))
            continue
        m = re.search(pos, lower)
        if m:
            value = not _negated(lower, m.start())
            profile.set(field, value, source, 0.85, _evidence(text, m.start(), m.end()))

    m = re.search(ACTIVITY_NEG, lower)
    if m:
        profile.set("phys_activity", False, source, 0.85, _evidence(text, m.start(), m.end()))
    else:
        m = re.search(ACTIVITY_POS, lower)
        if m and not _negated(lower, m.start()):
            profile.set("phys_activity", True, source, 0.8, _evidence(text, m.start(), m.end()))


def extract_profile(text: str, source: str) -> PatientProfile:
    profile = PatientProfile()
    extract_measurements(text, profile, source)
    extract_risk_factors(text, profile, source)
    profile.derive()
    return profile


GLUCOSE_SERIES_HINT: Callable[[str], bool] = lambda name: bool(
    re.search(r"glu|đường|duong|bg|cgm|sugar", name.lower())
)
