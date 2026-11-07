"""
Tải và chuẩn hóa BRFSS (Behavioral Risk Factor Surveillance System, CDC) nhiều năm cho nhánh DB → ML/DL.

BRFSS là khảo sát qua điện thoại lớn nhất thế giới về sức khỏe người trưởng thành (~400–490 nghìn người/năm).
Dùng các năm LẺ 2011–2023 vì câu hỏi tăng huyết áp và cholesterol chỉ có trong năm lẻ.

    python scripts/fetch_brfss.py                     # tải 7 năm lẻ 2011–2023 → data/brfss/brfss_diabetes.csv.gz
    python scripts/fetch_brfss.py --years 2015 2017   # chỉ một số năm

Nhãn (tự khai): diabetes = 1 nếu "bác sĩ từng nói bạn bị tiểu đường" (không tính ĐTĐ thai kỳ);
prediabetes = 1 nếu tiền ĐTĐ / ranh giới. Trả lời "không biết / từ chối" bị loại.
Đặc trưng dùng đúng tên trường của PatientProfile để tác tử ánh xạ trực tiếp; giá trị thiếu giữ NaN.
"""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "download" / "brfss"
OUT = ROOT / "data" / "output" / "datasets" / "brfss_diabetes.csv.gz"
URL = "https://www.cdc.gov/brfss/annual_data/{y}/files/LLCP{y}XPT.zip"
YEARS = [2021, 2023]

# Tên biến thay đổi theo năm: lấy biến đầu tiên có trong file
CANDIDATES = {
    "diab": ["DIABETE4", "DIABETE3", "DIABETE2"],
    "sex": ["SEXVAR", "_SEX", "SEX1", "SEX"],
    "age80": ["_AGE80"],
    "agegrp": ["_AGEG5YR"],
    "bmi": ["_BMI5"],
    "high_bp": ["_RFHYPE6", "_RFHYPE5"],
    "high_chol": ["_RFCHOL3", "_RFCHOL2", "_RFCHOL1", "_RFCHOL"],
    "smoker": ["SMOKE100"],
    "stroke": ["CVDSTRK3"],
    "heart_disease": ["_MICHD"],
    "mi": ["CVDINFR4"],          # 2011 chưa có _MICHD: suy ra từ nhồi máu cơ tim / bệnh mạch vành
    "chd": ["CVDCRHD4"],
    "phys_activity": ["_TOTINDA"],
    "heavy_alcohol": ["_RFDRHV8", "_RFDRHV7", "_RFDRHV6", "_RFDRHV5", "_RFDRHV4"],
    "gen_hlth": ["GENHLTH"],
    "race": ["_RACE"],           # 4 = người gốc Á (không Hispanic)
    "imprace": ["_IMPRACE"],     # 3 = người gốc Á (2011 chỉ có biến này)
}
OPTIONAL = {"age80", "heavy_alcohol", "heart_disease", "mi", "chd", "race", "imprace"}
# Điểm giữa của 13 nhóm tuổi _AGEG5YR (dùng khi năm đó không có _AGE80)
AGE_MID = {1: 21, 2: 27, 3: 32, 4: 37, 5: 42, 6: 47, 7: 52, 8: 57, 9: 62, 10: 67, 11: 72, 12: 77, 13: 82}


def download(year: int) -> Path:
    path = RAW / f"LLCP{year}XPT.zip"
    if not path.exists():
        import httpx
        RAW.mkdir(parents=True, exist_ok=True)
        print(f"Tải BRFSS {year}…")
        with httpx.stream("GET", URL.format(y=year), follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(path.with_suffix(".part"), "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        path.with_suffix(".part").rename(path)
    return path


def _yes_no(s: pd.Series, yes: int = 1, no: int = 2) -> pd.Series:
    """Mã BRFSS 1/2 (có/không) → 1/0; mọi mã khác (7, 9, trống) → NaN."""
    return s.map({yes: 1.0, no: 0.0})


def _heart(raw: pd.DataFrame) -> pd.Series:
    if "heart_disease" in raw:
        return _yes_no(raw["heart_disease"])
    mi, chd = _yes_no(raw["mi"]), _yes_no(raw["chd"])
    return np.fmax(mi, chd).where(mi.notna() | chd.notna())


def read_year(year: int) -> pd.DataFrame:
    with zipfile.ZipFile(download(year)) as z:
        name = next(n for n in z.namelist() if n.strip().upper().endswith(".XPT"))
        parts, cols = [], None
        with z.open(name) as fh:
            reader = pd.read_sas(fh, format="xport", chunksize=100_000, encoding="infer")
            for chunk in reader:
                if cols is None:
                    upper = {c.upper(): c for c in chunk.columns}
                    cols = {k: next((upper[c] for c in v if c in upper), None) for k, v in CANDIDATES.items()}
                    missing = [k for k, c in cols.items() if c is None and k not in OPTIONAL]
                    if missing:
                        raise SystemExit(f"BRFSS {year}: không tìm thấy biến {missing}")
                parts.append(chunk[[c for c in cols.values() if c]].rename(columns={v: k for k, v in cols.items() if v}))
    raw = pd.concat(parts, ignore_index=True)

    diab = raw["diab"]
    keep = diab.isin([1, 2, 3, 4])  # 7/9 = không biết / từ chối
    raw = raw[keep].reset_index(drop=True)
    diab = raw["diab"]
    age = raw["age80"] if "age80" in raw else pd.Series(np.nan, index=raw.index)
    age = age.where(age.between(18, 99), raw["agegrp"].map(AGE_MID))

    df = pd.DataFrame({
        "year": year,
        "age_years": age,
        "sex": raw["sex"].map({1: 1.0, 2: 0.0}),                        # 1 = nam
        "bmi": (raw["bmi"] / 100).where(raw["bmi"].between(1000, 9999)),
        "high_bp": _yes_no(raw["high_bp"], yes=2, no=1),                # _RFHYPE: 1 = không, 2 = có
        "high_chol": _yes_no(raw["high_chol"], yes=2, no=1),
        "smoker": _yes_no(raw["smoker"]),
        "stroke": _yes_no(raw["stroke"]),
        "heart_disease": _heart(raw),
        "phys_activity": _yes_no(raw["phys_activity"]),
        "heavy_alcohol": _yes_no(raw["heavy_alcohol"], yes=2, no=1) if "heavy_alcohol" in raw else np.nan,
        "gen_hlth": raw["gen_hlth"].where(raw["gen_hlth"].between(1, 5)),
        "asian": (raw["race"] == 4).astype(float) if "race" in raw else (raw["imprace"] == 3).astype(float),
        "diabetes": (diab == 1).astype(int),
        "prediabetes": (diab == 4).astype(int),
    })
    print(f"BRFSS {year}: {len(df):,} người, ĐTĐ {df.diabetes.mean():.1%}, tiền ĐTĐ {df.prediabetes.mean():.1%}")
    return df


def clean_and_filter(df: pd.DataFrame, max_per_year: int = 60_000) -> pd.DataFrame:
    """Chỉ giữ các bản ghi gần đây (2021–2023), đầy đủ 100% đặc trưng (không khuyết thiếu) và BMI hợp lý."""
    df = df[df["year"] >= 2021].dropna().copy()
    df = df[df["bmi"].between(13.0, 55.0) & df["age_years"].between(18, 80)]
    if max_per_year and len(df) > max_per_year * df["year"].nunique():
        parts = []
        for _, grp in df.groupby(["year", "diabetes"], group_keys=False):
            year_total = len(df[df["year"] == grp["year"].iloc[0]])
            n_take = min(len(grp), int(round(max_per_year * len(grp) / year_total)))
            parts.append(grp.sample(n=n_take, random_state=42))
        df = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--years", type=int, nargs="+", default=YEARS)
    ap.add_argument("--max-per-year", type=int, default=60_000, help="Số mẫu sạch tối đa mỗi năm (0 = giữ tất cả)")
    args = ap.parse_args()
    df = pd.concat([read_year(y) for y in args.years], ignore_index=True)
    df = clean_and_filter(df, max_per_year=args.max_per_year)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, compression="gzip")
    print(f"\nTổng sau làm sạch: {len(df):,} người ({min(args.years)}–{max(args.years)}), ĐTĐ {df.diabetes.mean():.1%} → {OUT}")
    print("Tiếp theo: python scripts/train_model.py --dataset brfss --reload")


if __name__ == "__main__":
    sys.exit(main())
