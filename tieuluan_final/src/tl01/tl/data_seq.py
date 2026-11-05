"""Chương 4: hai tập dữ liệu chuỗi.

1. Giá cổ phiếu bốn mã trụ cột VN30 trên HoSE (FPT, VCB, HPG, VIC), 01/2012 - 09/2026, tải từ
   Yahoo Finance qua yfinance. Bài toán hồi quy: từ cửa sổ 30 phiên x 9 đặc trưng của FPT,
   dự báo giá đóng cửa phiên kế tiếp.
2. KKBox Churn: log nghe nhạc 31 ngày x 8 đặc trưng của 100.000 người dùng thật (đã dựng ở
   Assignment 06 từ tệp log 1,4 GB). Bài toán phân loại chuỗi: người dùng có rời bỏ dịch vụ không.
"""
import numpy as np
import pandas as pd

from .common import DATA

SEQ = DATA / "seq"
TICKERS = ["FPT", "VCB", "HPG", "VIC"]
START, END = "2012-01-01", "2026-10-01"
FEATURES = ["Open", "High", "Low", "Close", "Volume", "SMA20", "SMA50", "RSI14", "Vol30"]
LOOKBACK = 30
TRAIN_FRAC = 0.8
CHURN_FEATURES = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_unq", "total_secs", "active"]


def download_stocks():
    import yfinance as yf
    for t in TICKERS:
        df = yf.download(f"{t}.VN", start=START, end=END, auto_adjust=True, progress=False)
        if df.empty:
            raise RuntimeError(f"yfinance không trả dữ liệu cho {t}.VN")
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df = df[df.Volume > 0]                       # bỏ ngày nghỉ lễ mà Yahoo vẫn ghi giá đứng yên
        df.to_csv(SEQ / f"{t}.csv")


def rsi(close, n=14):
    """RSI theo Wilder: trung bình trượt hàm mũ alpha = 1/n của mức tăng và mức giảm."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / down)


def load_stock(t):
    df = pd.read_csv(SEQ / f"{t}.csv", index_col=0, parse_dates=True)
    df["SMA20"] = df.Close.rolling(20).mean()
    df["SMA50"] = df.Close.rolling(50).mean()
    df["RSI14"] = rsi(df.Close)
    df["Vol30"] = df.Close.pct_change().rolling(30).std() * 100   # độ biến động 30 phiên, %/phiên
    return df


PRICE_COLS = [0, 1, 2, 3, 5, 6]                     # Open High Low Close SMA20 SMA50 trong FEATURES


def stock_windows(t="FPT", lookback=LOOKBACK):
    """Cửa sổ trượt bước 1, chia theo thời gian 80/20, chuẩn hóa theo TỪNG cửa sổ.

    Giá FPT giai đoạn kiểm thử (11/2023 - 09/2026) lên tới 118 nghìn đồng, gần gấp đôi mức cao
    nhất của giai đoạn huấn luyện; MinMaxScaler khớp trên tập huấn luyện đẩy đầu vào kiểm thử ra
    ngoài [0, 1] và mạng phải ngoại suy (thử nghiệm đầu cho R^2 âm). Vì vậy mọi cột giá được chia
    cho giá đóng cửa cuối cửa sổ (đơn vị: % chênh lệch), và nhãn là lợi suất phiên kế tiếp (%).
    Khối lượng lấy log rồi chuẩn hóa z-score theo thống kê tập huấn luyện; RSI chia 100.
    """
    df = load_stock(t).dropna()
    vals = df[FEATURES].to_numpy(np.float64)
    close = vals[:, 3]
    n_win = len(df) - lookback
    n_train = int(n_win * TRAIN_FRAC)
    logv = np.log1p(vals[:, 4])
    v_mu, v_sd = logv[:n_train + lookback].mean(), logv[:n_train + lookback].std()
    X = np.empty((n_win, lookback, len(FEATURES)), np.float32)
    for i in range(n_win):
        w = vals[i:i + lookback].copy()
        last = w[-1, 3]
        w[:, PRICE_COLS] = (w[:, PRICE_COLS] / last - 1) * 100
        w[:, 4] = (logv[i:i + lookback] - v_mu) / v_sd
        w[:, 7] = w[:, 7] / 100
        X[i] = w
    close_now = close[lookback - 1:-1]               # giá đóng cửa ngày cuối của mỗi cửa sổ
    close_next = close[lookback:]
    y = ((close_next / close_now - 1) * 100).astype(np.float32)
    dates = df.index[lookback:]
    sl_tr, sl_te = slice(0, n_train), slice(n_train, None)
    return {"X_train": X[sl_tr], "y_train": y[sl_tr], "X_test": X[sl_te], "y_test": y[sl_te],
            "close_now_train": close_now[sl_tr], "close_now_test": close_now[sl_te],
            "close_next_test": close_next[sl_te], "dates_test": dates[sl_te], "dates_train": dates[sl_tr],
            "frame": df}


def to_price(ret_pct, close_now):
    """Đổi lợi suất dự báo (%) về giá đóng cửa phiên kế tiếp (đồng)."""
    return close_now * (1 + np.asarray(ret_pct) / 100)


def churn_data():
    """Chuẩn hóa z-score theo từng đặc trưng với thống kê của tập huấn luyện; giữ nguyên chia 70/15/15."""
    z = np.load(SEQ / "kkbox_seq.npz")
    X, y = z["X"].astype(np.float32), z["y"].astype(np.int64)
    tr, va, te = z["idx_train"], z["idx_val"], z["idx_test"]
    mu = X[tr].mean((0, 1))
    sd = X[tr].std((0, 1)) + 1e-6
    Xs = (X - mu) / sd
    return {"X_train": Xs[tr], "y_train": y[tr], "X_val": Xs[va], "y_val": y[va],
            "X_test": Xs[te], "y_test": y[te], "X_raw": X, "y": y, "mu": mu, "sd": sd}


if __name__ == "__main__":
    SEQ.mkdir(parents=True, exist_ok=True)
    download_stocks()
    for t in TICKERS:
        df = load_stock(t)
        print(t, len(df), df.index.min().date(), df.index.max().date(), round(df.Close.iloc[-1]))
    d = stock_windows()
    print("FPT windows", d["X_train"].shape, d["X_test"].shape, d["dates_test"][0].date(),
          "y%", d["y_train"].std().round(3), np.abs(d["X_train"]).max(axis=(0, 1)).round(2))
