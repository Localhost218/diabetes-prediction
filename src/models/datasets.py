"""
Mô tả các bộ dữ liệu huấn luyện của nhánh DB → ML/DL.

- nhanes: NHANES 2011–2018 (CDC, dữ liệu thật), nhãn ĐTĐ xác định bằng xét nghiệm + chẩn đoán của bác sĩ.
- vn: vietnam_diabetes_dataset.csv (150.000 mẫu, chưa rõ nguồn, nhiều dấu hiệu là dữ liệu mô phỏng).

Cả hai dùng chung quy ước: đặc trưng chưa biết = NaN (không điền trung vị), vì người dùng chatbot thường
chỉ khai một phần thông tin.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sqlalchemy import inspect, text

from src.config import BASE_DIR, settings
from src.schemas.patient_profile import PatientProfile


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    title: str
    table: str
    csv_path: str
    features: List[str]
    feature_vi: Dict[str, str]
    core_features: List[str]          # luôn có khi chatbot chạy dự đoán (không bị che khi tăng cường dữ liệu)
    label: str
    profile_to_row: Callable[[PatientProfile], Dict[str, Optional[float]]]
    subgroup: Optional[str] = None    # cột dùng để đánh giá riêng một nhóm (ví dụ người gốc Á)


def _b(profile: PatientProfile, name: str) -> Optional[float]:
    v = profile.get(name)
    return None if v is None else float(bool(v))


def _num(profile: PatientProfile, name: str) -> Optional[float]:
    v = profile.get(name)
    return None if v is None else float(v)


# ---------------------------------------------------------------- NHANES
NHANES_FEATURES = ["age_years", "sex", "bmi", "waist_cm", "systolic_bp", "diastolic_bp", "high_bp", "high_chol",
                   "family_history", "smoker", "phys_activity", "heart_disease", "stroke"]
NHANES_VI = {
    "age_years": "Tuổi", "sex": "Giới tính nam", "bmi": "Chỉ số khối cơ thể (BMI)", "waist_cm": "Vòng eo (cm)",
    "systolic_bp": "Huyết áp tâm thu", "diastolic_bp": "Huyết áp tâm trương", "high_bp": "Tăng huyết áp",
    "high_chol": "Rối loạn mỡ máu", "family_history": "Tiền sử gia đình mắc ĐTĐ", "smoker": "Hút thuốc lá",
    "phys_activity": "Vận động thể lực", "heart_disease": "Bệnh tim mạch", "stroke": "Tiền sử đột quỵ",
}


def _nhanes_row(profile: PatientProfile) -> Dict[str, Optional[float]]:
    return {
        "age_years": _num(profile, "age_years"), "sex": _num(profile, "sex"), "bmi": _num(profile, "bmi"),
        "waist_cm": _num(profile, "waist_cm"), "systolic_bp": _num(profile, "systolic_bp"),
        "diastolic_bp": _num(profile, "diastolic_bp"), "high_bp": _b(profile, "high_bp"),
        "high_chol": _b(profile, "high_chol"), "family_history": _b(profile, "family_history"),
        "smoker": _b(profile, "smoker"), "phys_activity": _b(profile, "phys_activity"),
        "heart_disease": _b(profile, "heart_disease"), "stroke": _b(profile, "stroke"),
    }


# ---------------------------------------------------------------- bộ dữ liệu VN
VN_FEATURES = ["HighBP", "HighChol", "BMI", "Waist_cm", "CentralObesity_VN", "Smoker", "Stroke",
               "HeartDiseaseorAttack", "PhysActivity", "Veggies", "HvyAlcoholConsump", "FamilyHistory", "GenHlth",
               "PhysHlth", "Sex", "Age", "UrbanResidence", "Income_VN"]
VN_VI = {
    "HighBP": "Huyết áp cao", "HighChol": "Rối loạn mỡ máu", "BMI": "Chỉ số khối cơ thể (BMI)", "Waist_cm": "Vòng eo (cm)",
    "CentralObesity_VN": "Béo phì vùng bụng (VN)", "Smoker": "Hút thuốc lá", "Stroke": "Tiền sử đột quỵ",
    "HeartDiseaseorAttack": "Bệnh mạch vành / Tim", "PhysActivity": "Vận động thể chất", "Veggies": "Ăn rau hàng ngày",
    "HvyAlcoholConsump": "Uống nhiều rượu bia", "FamilyHistory": "Tiền sử gia đình mắc ĐTĐ",
    "GenHlth": "Sức khỏe chung tự đánh giá", "PhysHlth": "Số ngày đau ốm thể chất", "Sex": "Giới tính",
    "Age": "Nhóm tuổi", "UrbanResidence": "Nơi cư trú thành thị", "Income_VN": "Mức thu nhập",
}


def age_to_group(age_years: float) -> int:
    """Tuổi thực → nhóm tuổi 13 bậc (1: 18–24, 2: 25–29, ..., 12: 75–79, 13: ≥ 80)."""
    if age_years < 25:
        return 1
    return int(min(13, 2 + (age_years - 25) // 5))


def _vn_row(profile: PatientProfile) -> Dict[str, Optional[float]]:
    waist, sex, age = profile.get("waist_cm"), profile.get("sex"), profile.get("age_years")
    central = None if waist is None or sex is None else float(waist >= (90 if sex == 1 else 80))
    return {
        "HighBP": _b(profile, "high_bp"), "HighChol": _b(profile, "high_chol"), "BMI": _num(profile, "bmi"),
        "Waist_cm": _num(profile, "waist_cm"), "CentralObesity_VN": central, "Smoker": _b(profile, "smoker"),
        "Stroke": _b(profile, "stroke"), "HeartDiseaseorAttack": _b(profile, "heart_disease"),
        "PhysActivity": _b(profile, "phys_activity"), "Veggies": None, "HvyAlcoholConsump": _b(profile, "heavy_alcohol"),
        "FamilyHistory": _b(profile, "family_history"), "GenHlth": _num(profile, "gen_hlth"), "PhysHlth": None,
        "Sex": None if sex is None else float(sex), "Age": None if age is None else float(age_to_group(age)),
        "UrbanResidence": None, "Income_VN": None,
    }


DATASETS: Dict[str, DatasetSpec] = {
    "nhanes": DatasetSpec(
        name="nhanes", title="NHANES 2011–2018 (CDC)", table="nhanes_samples",
        csv_path=str(BASE_DIR / "data" / "nhanes" / "nhanes_diabetes.csv"),
        features=NHANES_FEATURES, feature_vi=NHANES_VI, core_features=["age_years", "bmi"],
        label="diabetes", profile_to_row=_nhanes_row, subgroup="asian",
    ),
    "vn": DatasetSpec(
        name="vn", title="vietnam_diabetes_dataset (150k, chưa rõ nguồn)", table="training_samples",
        csv_path=str(settings.dataset_csv), features=VN_FEATURES, feature_vi=VN_VI, core_features=["Age", "BMI"],
        label="Diabetes_binary", profile_to_row=_vn_row,
    ),
}


def load_dataframe(spec: DatasetSpec, engine, reload: bool = False) -> pd.DataFrame:
    """Đọc dữ liệu huấn luyện từ PostgreSQL (khối DB trên sơ đồ); nạp CSV vào DB nếu bảng chưa có."""
    has_rows = inspect(engine).has_table(spec.table) and \
        pd.read_sql(text(f"SELECT EXISTS (SELECT 1 FROM {spec.table})"), engine).iat[0, 0]
    if reload or not has_rows:
        csv = pd.read_csv(spec.csv_path)
        # chia lô để số tham số mỗi câu INSERT dưới giới hạn 65.535 của Postgres
        csv.to_sql(spec.table, engine, if_exists="replace", index=False,
                   chunksize=max(100, 60_000 // len(csv.columns)), method="multi")
        print(f"Đã nạp {len(csv):,} mẫu vào bảng {spec.table}")
    cols = spec.features + [spec.label] + ([spec.subgroup] if spec.subgroup else [])
    return pd.read_sql(text(f"SELECT {', '.join(chr(34) + c + chr(34) for c in cols)} FROM {spec.table}"), engine)


def mask_augment(X: pd.DataFrame, y: pd.Series, spec: DatasetSpec, p: float = 0.4, seed: int = 0):
    """Nhân đôi tập huấn luyện, bản sao bị che ngẫu nhiên các đặc trưng không bắt buộc — mô phỏng người dùng
    chỉ khai một phần thông tin, để mô hình học cách dự đoán khi thiếu dữ liệu."""
    rng = np.random.default_rng(seed)
    masked = X.copy()
    for col in spec.features:
        if col not in spec.core_features:
            masked.loc[rng.random(len(masked)) < p, col] = np.nan
    return pd.concat([X, masked], ignore_index=True), pd.concat([y, y], ignore_index=True)
