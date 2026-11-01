"""
Tải dữ liệu NHANES (CDC, công khai) và dựng bộ dữ liệu đái tháo đường có nhãn xác định bằng xét nghiệm.

    python scripts/fetch_nhanes.py

- 4 chu kỳ 2011–2018 (từ 2011 NHANES có nhóm "Non-Hispanic Asian" riêng — dùng để đánh giá trên người gốc Á).
- Người trưởng thành ≥ 20 tuổi, loại phụ nữ đang mang thai.
- Nhãn đái tháo đường (diagnosed + undiagnosed) theo ADA: đã được bác sĩ chẩn đoán (DIQ010) HOẶC HbA1c ≥ 6,5%
  HOẶC glucose lúc đói ≥ 126 mg/dL. Nhãn phụ: tiền đái tháo đường (HbA1c 5,7–6,4% hoặc FPG 100–125).
- Đặc trưng chỉ gồm những gì người dùng có thể tự khai với chatbot (không dùng xét nghiệm làm đặc trưng để tránh rò nhãn).

Kết quả: data/nhanes/nhanes_diabetes.csv và bảng nhanes_samples trong PostgreSQL.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.config import BASE_DIR  # noqa: E402

CYCLES = {"G": 2011, "H": 2013, "I": 2015, "J": 2017}
FILES = ["DEMO", "BMX", "BPX", "BPQ", "DIQ", "MCQ", "SMQ", "PAQ", "GHB", "GLU"]
URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/{year}/DataFiles/{name}_{suffix}.xpt"
RAW_DIR = BASE_DIR / "data" / "nhanes" / "raw"
OUT_CSV = BASE_DIR / "data" / "nhanes" / "nhanes_diabetes.csv"


def download(name: str, suffix: str, year: int) -> Path:
    path = RAW_DIR / f"{name}_{suffix}.xpt"
    if not path.exists():
        url = URL.format(year=year, name=name, suffix=suffix)
        print(f"  tải {url}")
        with urllib.request.urlopen(url, timeout=120) as r:
            data = r.read()
        if data[:6] != b"HEADER":
            raise RuntimeError(f"{url} không phải file XPT")
        path.write_bytes(data)
    return path


def yes(series: pd.Series) -> pd.Series:
    """Mã hóa câu hỏi có/không của NHANES: 1 = có, 2 = không, còn lại (7/9/thiếu) = NaN."""
    return series.map({1: 1.0, 2: 0.0})


def build_cycle(suffix: str, year: int) -> pd.DataFrame:
    tables = {name: pd.read_sas(download(name, suffix, year), format="xport", encoding="infer") for name in FILES}
    df = tables["DEMO"][["SEQN", "RIAGENDR", "RIDAGEYR", "RIDRETH3", "RIDEXPRG"]]
    for name in FILES[1:]:
        df = df.merge(tables[name], on="SEQN", how="left", suffixes=("", f"_{name}"))

    bp_sys = df[[c for c in ("BPXSY1", "BPXSY2", "BPXSY3") if c in df]].mean(axis=1)
    bp_dia = df[[c for c in ("BPXDI1", "BPXDI2", "BPXDI3") if c in df]].replace(0, np.nan).mean(axis=1)
    out = pd.DataFrame({
        "seqn": df.SEQN.astype(int),
        "cycle": f"{year}-{year + 1}",
        "age_years": df.RIDAGEYR,
        "sex": (df.RIAGENDR == 1).astype(float),  # 1 = nam
        "bmi": df.BMXBMI,
        "waist_cm": df.BMXWAIST,
        "systolic_bp": bp_sys,
        "diastolic_bp": bp_dia,
        "high_bp": yes(df.BPQ020),
        "high_chol": yes(df.BPQ080),
        "family_history": yes(df.MCQ300C),
        "smoker": df.SMQ040.map({1: 1.0, 2: 1.0, 3: 0.0}).where(df.SMQ020 == 1, 0.0).where(df.SMQ020.isin([1, 2])),
        "phys_activity": ((df.PAQ650 == 1) | (df.PAQ665 == 1)).astype(float).where(df.PAQ650.isin([1, 2])),
        "heart_disease": ((df.MCQ160B == 1) | (df.MCQ160C == 1) | (df.MCQ160E == 1)).astype(float)
                         .where(df.MCQ160C.isin([1, 2])),
        "stroke": yes(df.MCQ160F),
        "asian": (df.RIDRETH3 == 6).astype(int),
        "hba1c": df.LBXGH,
        "fasting_glucose": df.LBXGLU,
        "diagnosed": (df.DIQ010 == 1).astype(int),
        "pregnant": (df.RIDEXPRG == 1).astype(int),
    })
    return out


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for suffix, year in CYCLES.items():
        print(f"Chu kỳ {year}-{year + 1}")
        frames.append(build_cycle(suffix, year))
    df = pd.concat(frames, ignore_index=True)

    df = df[(df.age_years >= 20) & (df.pregnant == 0)]
    df = df[df.hba1c.notna() | (df.diagnosed == 1)]  # cần xét nghiệm để xác định nhãn
    df["diabetes"] = ((df.diagnosed == 1) | (df.hba1c >= 6.5) | (df.fasting_glucose >= 126)).astype(int)
    df["prediabetes"] = ((df.diabetes == 0) & ((df.hba1c >= 5.7) | (df.fasting_glucose >= 100))).astype(int)
    df["undiagnosed"] = ((df.diabetes == 1) & (df.diagnosed == 0)).astype(int)
    df = df.drop(columns=["pregnant"])

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nĐã lưu {OUT_CSV} — {len(df):,} người trưởng thành")
    print(f"  Đái tháo đường: {df.diabetes.mean():.1%} (trong đó chưa được chẩn đoán: {df.undiagnosed.sum():,} người)")
    print(f"  Tiền đái tháo đường: {df.prediabetes.mean():.1%}")
    print(f"  Người gốc Á: {df.asian.sum():,} — tỷ lệ ĐTĐ {df[df.asian == 1].diabetes.mean():.1%}")
    print("  Tỷ lệ thiếu dữ liệu:", df.drop(columns=["hba1c", "fasting_glucose"]).isna().mean().round(3)
          .loc[lambda s: s > 0].to_dict())


if __name__ == "__main__":
    main()
