"""Nạp và mã hóa dữ liệu (không phụ thuộc torch để các kịch bản CPU chạy song song với GPU)."""
import numpy as np
import pandas as pd

from .common import DATA

# ================================================================ dữ liệu y tế
SMOKING = ["never", "No Info", "current", "former", "ever", "not current"]
NUMERIC = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]


def diabetes_features(df):
    """Mã hóa hồ sơ: 4 cột số + giới tính + 2 cờ bệnh nền + one-hot hút thuốc (6) = 13 đặc trưng."""
    X = pd.DataFrame({c: df[c].astype(float) for c in NUMERIC})
    X["male"] = (df["gender"] == "Male").astype(float)
    X["hypertension"] = df["hypertension"].astype(float)
    X["heart_disease"] = df["heart_disease"].astype(float)
    for s in SMOKING:
        X[f"smoke_{s}"] = (df["smoking_history"] == s).astype(float)
    return X


def load_diabetes():
    df = pd.read_csv(DATA / "tabular" / "diabetes_prediction_dataset.csv")
    df = df[df["gender"] != "Other"].drop_duplicates().reset_index(drop=True)
    return df


# ================================================================ dữ liệu thị trường
SERIES = ["FPT", "HPG", "VCB", "VIC", "XAU", "USDVND", "DXY", "VN30ETF"]
EXOG = ["USDVND", "XAU", "DXY", "VN30ETF"]
TROY_PER_LUONG = 1.2057


def load_series(name):
    df = pd.read_csv(DATA / "market" / f"{name}.csv", parse_dates=["Date"], index_col="Date")
    return df["Close"].astype(float).dropna()


def market_frame():
    """Bảng giá đóng cửa trên lịch giao dịch của HOSE; chuỗi nước ngoài được điền tiến (ffill) vào ngày VN."""
    raw = {s: load_series(s) for s in SERIES}
    cal = raw["FPT"].index
    df = pd.DataFrame({s: raw[s].reindex(cal.union(raw[s].index)).ffill().reindex(cal) for s in SERIES})
    df["GOLD_VND"] = df["XAU"] * df["USDVND"] * TROY_PER_LUONG / 1e6    # triệu đồng/lượng
    return df.loc["2015-01-01":].dropna()


def gold_frame():
    """Giá vàng quy đổi theo lịch giao dịch của vàng thế giới (cập nhật tới ngày tải dữ liệu)."""
    xau, fx = load_series("XAU"), load_series("USDVND")
    cal = xau.index
    df = pd.DataFrame({"XAU": xau, "USDVND": fx.reindex(cal.union(fx.index)).ffill().reindex(cal)})
    for s in ("DXY", "VN30ETF"):
        x = load_series(s)
        df[s] = x.reindex(cal.union(x.index)).ffill().reindex(cal)
    df["GOLD_VND"] = df["XAU"] * df["USDVND"] * TROY_PER_LUONG / 1e6
    return df.loc["2015-01-01":].dropna()


def forecast_features(df, target, keep_last=False):
    """Đặc trưng mỗi ngày: lợi suất log của đích và 4 biến ngoại sinh, độ lệch so với MA20, độ biến động 20 phiên.

    keep_last=True giữ phiên mới nhất (chưa có nhãn y) cho suy luận; huấn luyện cần nhãn nên bỏ phiên đó.
    """
    r = np.log(df).diff()
    F = pd.DataFrame({"r_target": r[target]})
    for s in EXOG:
        if s != target and not (target == "GOLD_VND" and s == "XAU"):
            F[f"r_{s}"] = r[s]
    F["dev_ma20"] = df[target] / df[target].rolling(20).mean() - 1
    F["vol20"] = r[target].rolling(20).std()
    F["y"] = r[target].shift(-1)                                # lợi suất phiên kế tiếp
    F["price"] = df[target]
    return F.dropna(subset=F.columns.drop("y")) if keep_last else F.dropna()


def make_windows(F, cols, window):
    X = F[cols].to_numpy(np.float32)
    y = F["y"].to_numpy(np.float32)
    idx = np.arange(window - 1, len(F))
    Xw = np.stack([X[i - window + 1:i + 1] for i in idx])
    return Xw, y[idx], F.index[idx]
