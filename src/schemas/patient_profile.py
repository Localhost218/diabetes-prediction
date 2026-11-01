"""
PatientProfile — "Blackboard" dùng chung giữa các tác tử.

Mỗi trường lưu kèm nguồn (text / file:<tên> / derived) và độ tin cậy, để ConsensusCriticAgent
phân biệt được giá trị người dùng cung cấp với giá trị suy luận hoặc bị thiếu.
Hệ thống KHÔNG tự điền giá trị mặc định cho các trường lâm sàng.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class FieldSpec:
    label_vi: str
    kind: str  # number | int | bool | sex
    valid_range: Optional[Tuple[float, float]] = None
    unit: str = ""
    question_vi: str = ""


FIELD_SPECS: Dict[str, FieldSpec] = {
    "age_years": FieldSpec("Tuổi", "int", (1, 120), "tuổi", "Bạn bao nhiêu tuổi?"),
    "sex": FieldSpec("Giới tính", "sex", None, "", "Giới tính của bạn là nam hay nữ?"),
    "height_cm": FieldSpec("Chiều cao", "number", (80, 230), "cm"),
    "weight_kg": FieldSpec("Cân nặng", "number", (20, 300), "kg"),
    "bmi": FieldSpec("BMI", "number", (10, 70), "kg/m²", "Chiều cao và cân nặng của bạn (hoặc chỉ số BMI)?"),
    "waist_cm": FieldSpec("Vòng eo", "number", (40, 200), "cm", "Vòng eo của bạn khoảng bao nhiêu cm?"),
    "glucose_fasting_mgdl": FieldSpec("Đường huyết lúc đói", "number", (30, 700), "mg/dL",
                                      "Đường huyết lúc đói gần nhất của bạn (ghi rõ mmol/L hay mg/dL)?"),
    "glucose_random_mgdl": FieldSpec("Đường huyết bất kỳ", "number", (30, 900), "mg/dL"),
    "ogtt_2h_mgdl": FieldSpec("Glucose 2 giờ sau nghiệm pháp (OGTT)", "number", (30, 900), "mg/dL"),
    "hba1c_percent": FieldSpec("HbA1c", "number", (3, 20), "%", "Chỉ số HbA1c gần nhất của bạn?"),
    "systolic_bp": FieldSpec("Huyết áp tâm thu", "number", (60, 260), "mmHg"),
    "diastolic_bp": FieldSpec("Huyết áp tâm trương", "number", (30, 160), "mmHg"),
    "total_cholesterol_mgdl": FieldSpec("Cholesterol toàn phần", "number", (50, 600), "mg/dL"),
    "ldl_mgdl": FieldSpec("LDL-C", "number", (10, 400), "mg/dL"),
    "hdl_mgdl": FieldSpec("HDL-C", "number", (5, 150), "mg/dL"),
    "triglycerides_mgdl": FieldSpec("Triglycerid", "number", (20, 3000), "mg/dL"),
    "creatinine_mgdl": FieldSpec("Creatinin máu", "number", (0.1, 20), "mg/dL"),
    "egfr": FieldSpec("eGFR", "number", (1, 200), "mL/phút/1.73m²"),
    "high_bp": FieldSpec("Tăng huyết áp", "bool", None, "", "Bạn có bị tăng huyết áp không?"),
    "high_chol": FieldSpec("Rối loạn mỡ máu", "bool", None, "", "Bạn có bị mỡ máu cao không?"),
    "family_history": FieldSpec("Tiền sử gia đình mắc ĐTĐ", "bool", None, "",
                                "Bố mẹ hoặc anh chị em ruột có ai bị tiểu đường không?"),
    "smoker": FieldSpec("Hút thuốc", "bool"),
    "phys_activity": FieldSpec("Vận động thể lực thường xuyên", "bool"),
    "heart_disease": FieldSpec("Bệnh tim mạch", "bool"),
    "stroke": FieldSpec("Tiền sử đột quỵ", "bool"),
    "heavy_alcohol": FieldSpec("Uống nhiều rượu bia", "bool"),
    "gen_hlth": FieldSpec("Tự đánh giá sức khỏe (1 rất tốt – 5 kém)", "int", (1, 5)),
}

# Tối thiểu để chạy dự đoán: tuổi, BMI và ít nhất một xét nghiệm đường huyết
REQUIRED_FIELDS = ["age_years", "bmi"]
GLYCEMIC_FIELDS = ["glucose_fasting_mgdl", "hba1c_percent", "ogtt_2h_mgdl", "glucose_random_mgdl"]
# Hỏi thêm (không chặn dự đoán) để mô hình chính xác hơn
RECOMMENDED_FIELDS = ["sex", "high_bp", "family_history", "waist_cm"]


class FieldValue(BaseModel):
    value: Any
    source: str  # "text" | "file:<tên file>" | "derived" | "form"
    confidence: float = 1.0
    evidence: Optional[str] = None
    flags: List[str] = Field(default_factory=list)  # fasting_unclear | unit_inferred
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class Conflict(BaseModel):
    field: str
    existing: FieldValue
    incoming: FieldValue


class PatientProfile(BaseModel):
    fields: Dict[str, FieldValue] = Field(default_factory=dict)
    glucose_history: Optional[List[float]] = None  # chuỗi glucose (mg/dL) cho RNN
    glucose_history_source: Optional[str] = None
    fundus_features: Optional[Dict[str, float]] = None  # đặc trưng ảnh đáy mắt cho CNN
    conflicts: List[Conflict] = Field(default_factory=list)

    # ---------- truy cập ----------
    def get(self, name: str, default: Any = None) -> Any:
        fv = self.fields.get(name)
        return fv.value if fv is not None else default

    def has(self, name: str) -> bool:
        return name in self.fields

    def set(self, name: str, value: Any, source: str, confidence: float = 1.0, evidence: Optional[str] = None,
            flags: Optional[List[str]] = None) -> bool:
        """Ghi một trường sau khi kiểm tra miền giá trị. Trả về False nếu giá trị bị loại."""
        spec = FIELD_SPECS.get(name)
        if spec is None or value is None:
            return False
        value = _coerce(spec, value)
        if value is None:
            return False
        if spec.valid_range and not (spec.valid_range[0] <= float(value) <= spec.valid_range[1]):
            return False

        incoming = FieldValue(value=value, source=source, confidence=confidence, evidence=evidence, flags=flags or [])
        existing = self.fields.get(name)
        if existing is not None and existing.source != source and existing.source != "derived" \
                and not _close(existing.value, value):
            self.conflicts = [c for c in self.conflicts if c.field != name]
            self.conflicts.append(Conflict(field=name, existing=existing, incoming=incoming))
        self.fields[name] = incoming
        return True

    def merge(self, other: "PatientProfile") -> List[str]:
        """Gộp dữ liệu mới vào hồ sơ; trả về danh sách trường đã cập nhật."""
        updated = []
        for name, fv in other.fields.items():
            if self.set(name, fv.value, fv.source, fv.confidence, fv.evidence, fv.flags):
                updated.append(name)
        if other.glucose_history:
            self.glucose_history = other.glucose_history
            self.glucose_history_source = other.glucose_history_source
            updated.append("glucose_history")
        if other.fundus_features:
            self.fundus_features = other.fundus_features
            updated.append("fundus_features")
        self.derive()
        return updated

    def derive(self) -> None:
        """Suy ra các trường phụ thuộc (BMI từ chiều cao/cân nặng, tăng huyết áp từ số đo...)."""
        h, w = self.get("height_cm"), self.get("weight_kg")
        if h and w and (not self.has("bmi") or self.fields["bmi"].source == "derived"):
            self.set("bmi", round(w / (h / 100) ** 2, 1), "derived", 0.95, f"{w} kg / ({h} cm)²")

        sbp, dbp = self.get("systolic_bp"), self.get("diastolic_bp")
        if (sbp or dbp) and not self.has("high_bp"):
            hypertensive = (sbp or 0) >= 140 or (dbp or 0) >= 90
            self.set("high_bp", hypertensive, "derived", 0.8, f"Huyết áp đo được {sbp}/{dbp} mmHg")

        if not self.has("high_chol"):
            tc, ldl, tg = self.get("total_cholesterol_mgdl"), self.get("ldl_mgdl"), self.get("triglycerides_mgdl")
            if tc or ldl or tg:
                dyslipidemia = (tc or 0) >= 240 or (ldl or 0) >= 160 or (tg or 0) >= 200
                self.set("high_chol", dyslipidemia, "derived", 0.8, "Suy ra từ bộ mỡ máu")

    # ---------- đánh giá đủ dữ liệu ----------
    def missing_required(self) -> List[str]:
        missing = [f for f in REQUIRED_FIELDS if not self.has(f)]
        if not any(self.has(f) for f in GLYCEMIC_FIELDS) and not self.glucose_history:
            missing.append("glucose_fasting_mgdl")
        return missing

    def missing_recommended(self) -> List[str]:
        return [f for f in RECOMMENDED_FIELDS if not self.has(f)]

    def is_ready(self) -> bool:
        return not self.missing_required()

    def summary_vi(self) -> List[str]:
        lines = [f"{FIELD_SPECS[name].label_vi}: {format_value(name, fv.value)} ({_source_vi(fv.source)})"
                 for name, fv in self.fields.items()]
        if self.glucose_history:
            lines.append(f"Chuỗi đường huyết: {len(self.glucose_history)} lần đo ({_source_vi(self.glucose_history_source or '')})")
        return lines


def format_value(name: str, value: Any) -> str:
    spec = FIELD_SPECS[name]
    if spec.kind == "bool":
        return "Có" if value else "Không"
    if spec.kind == "sex":
        return "Nam" if value == 1 else "Nữ"
    return f"{value:g} {spec.unit}".strip() if isinstance(value, (int, float)) else str(value)


def source_vi(source: str) -> str:
    return _source_vi(source)


def _coerce(spec: FieldSpec, value: Any) -> Any:
    try:
        if spec.kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "có", "co")
            return bool(value)
        if spec.kind == "sex":
            if isinstance(value, str):
                v = value.strip().lower()
                if v in ("male", "nam", "m", "1"):
                    return 1
                if v in ("female", "nữ", "nu", "f", "0"):
                    return 0
                return None
            return 1 if int(value) == 1 else 0
        if spec.kind == "int":
            return int(round(float(value)))
        return round(float(value), 2)
    except (TypeError, ValueError):
        return None


def _close(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= max(0.02 * abs(float(a)), 0.05)
    except (TypeError, ValueError):
        return a == b


def _source_vi(source: str) -> str:
    if source.startswith("file:"):
        return f"từ file {source[5:]}"
    return {"text": "bạn cung cấp", "derived": "tính toán", "form": "nhập form"}.get(source, source)
