"""Chương 4 — dữ liệu chuỗi thời gian: nạp, gộp theo ngày, tạo cửa sổ trượt, chia theo thời gian, chuẩn hoá.

Ba thực nghiệm:
  ecommerce  E-Commerce Customer Behavior & Sales Analysis -TR (Kaggle umuttuygurr), gộp đơn hàng theo ngày,
             dự báo Total_Amount ngày kế tiếp từ 30 ngày trước.
  aapl       AAPL Stock Performance (Kaggle nitirajkulkarni), dự báo giá đóng cửa phiên kế tiếp từ 30 phiên.
  gold       Gold Price Prediction Dataset — FINAL_USO.csv (Kaggle sid321axn), giá đóng cửa quỹ vàng GLD (USD)
             cùng các chỉ báo thị trường (S&P 500, Dow Jones, EUR/USD, dầu Brent, USD index, bạc, GDX, USO).
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .common import data_dir

WINDOW = 30
SPLIT = (0.70, 0.15, 0.15)   # train / validation / test theo thứ tự thời gian

KAGGLE = {
    "ecommerce": ("umuttuygurr/e-commerce-customer-behavior-and-sales-analysis-tr",
                  "ecommerce_customer_behavior_dataset.csv"),
    "aapl": ("nitirajkulkarni/aapl-stock-performance", "AAPL_stock_data.csv"),
    "gold": ("sid321axn/gold-price-prediction-dataset", "FINAL_USO.csv"),
}


def download_all():
    """Tải ba bộ dữ liệu Kaggle (kagglehub, ẩn danh) và chép tệp CSV cần dùng vào data/ch4/."""
    import shutil
    from pathlib import Path

    import kagglehub
    out = data_dir("ch4")
    for name, (slug, fname) in KAGGLE.items():
        src = Path(kagglehub.dataset_download(slug)) / fname
        shutil.copy2(src, out / fname)
        print(name, "->", out / fname)


def ecommerce_daily():
    """Gộp 5.000 đơn hàng thành một vector 9 đặc trưng cho mỗi ngày (mỗi ngày là một bước thời gian)."""
    raw = pd.read_csv(data_dir("ch4") / KAGGLE["ecommerce"][1], parse_dates=["Date"])
    g = raw.groupby("Date")
    daily = pd.DataFrame({
        "Total_Amount": g["Total_Amount"].sum(),               # doanh thu ngày — biến mục tiêu
        "Transactions": g["Order_ID"].count(),                 # số đơn hàng
        "Quantity": g["Quantity"].sum(),                       # số sản phẩm bán ra
        "Avg_Unit_Price": g["Unit_Price"].mean(),
        "Discount": g["Discount_Amount"].sum(),
        "Avg_Session": g["Session_Duration_Minutes"].mean(),
        "Avg_Pages": g["Pages_Viewed"].mean(),
        "Avg_Rating": g["Customer_Rating"].mean(),
        "Returning_Rate": g["Is_Returning_Customer"].mean(),
    })
    # ngày không có đơn (nếu có) được điền 0 cho đại lượng cộng dồn, giá trị trước đó cho đại lượng trung bình
    full = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full)
    for c in ("Total_Amount", "Transactions", "Quantity", "Discount"):
        daily[c] = daily[c].fillna(0.0)
    daily = daily.ffill()
    daily.index.name = "Date"
    return daily


def aapl_frame():
    """Bảng giá AAPL theo phiên: open, high, low, close, adjclose, volume (1980-12-12 → 2024)."""
    df = pd.read_csv(data_dir("ch4") / KAGGLE["aapl"][1], index_col=0, parse_dates=True)
    df.index.name = "Date"
    return df[["open", "high", "low", "close", "adjclose", "volume"]].astype(float).sort_index()


GOLD_COLS = {"Close": "GLD_Close", "Open": "GLD_Open", "High": "GLD_High", "Low": "GLD_Low", "Volume": "GLD_Volume",
             "SP_close": "SP500", "DJ_close": "DowJones", "EU_Price": "EURUSD", "OF_Price": "Brent",
             "USDI_Price": "USD_Index", "SF_Price": "Silver_Fut", "GDX_Close": "GDX", "USO_Close": "USO"}


def gold_frame():
    """FINAL_USO.csv: giá quỹ vàng GLD (mục tiêu GLD_Close, USD/chứng chỉ) và 12 chỉ báo liên thị trường."""
    df = pd.read_csv(data_dir("ch4") / KAGGLE["gold"][1], parse_dates=["Date"]).set_index("Date").sort_index()
    return df[list(GOLD_COLS)].rename(columns=GOLD_COLS).astype(float)


@dataclass
class Experiment:
    """Cấu hình một thực nghiệm: bảng dữ liệu, cột mục tiêu, cách chuẩn hoá."""
    name: str
    frame: pd.DataFrame
    target: str
    mode: str                       # "level": chuẩn hoá z-score fit trên train; "relative": log-tỉ số so với bước cuối cửa sổ
    level_cols: list = field(default_factory=list)   # cột không âm, khác thang giá (khối lượng) — log1p rồi z-score
    unit: str = ""


def experiment(name):
    if name == "ecommerce":
        return Experiment(name, ecommerce_daily(), "Total_Amount", "level", unit="TRY")
    if name == "aapl":
        return Experiment(name, aapl_frame(), "close", "relative", level_cols=["volume"], unit="USD")
    if name == "gold":
        return Experiment(name, gold_frame(), "GLD_Close", "relative", level_cols=["GLD_Volume"], unit="USD")
    raise KeyError(name)


def time_split(n, split=SPLIT):
    """Chỉ số cắt [0, a) train, [a, b) validation, [b, n) test — không xáo trộn, không rò rỉ tương lai."""
    a = int(round(n * split[0]))
    b = a + int(round(n * split[1]))
    return a, b


def make_windows(values, target, window=WINDOW):
    """Cửa sổ trượt: X_i = values[i-window:i] (window × F), y_i = target[i] — many-to-one."""
    X = np.stack([values[i - window:i] for i in range(window, len(values))])
    y = np.asarray(target[window:])
    return X.astype(np.float32), y.astype(np.float32)


class Prepared:
    """Dữ liệu đã chuẩn hoá và cắt cửa sổ cho một thực nghiệm, kèm hàm đổi dự báo về đơn vị gốc.

    mode="level"    (e-commerce): mọi cột z-score theo thống kê tập train; mục tiêu là giá trị chuẩn hoá ngày kế.
    mode="relative" (giá AAPL, vàng): mỗi cột giá lấy log(x_t / x_cuối-cửa-sổ) nên mọi cửa sổ cùng thang đo
                    dù giá năm 2024 gấp hàng trăm lần năm 1990; mục tiêu là log-lợi suất log(P_{t+1}/P_t).
    """

    def __init__(self, exp, window=WINDOW, split=SPLIT, mode=None):
        self.exp, self.window = exp, window
        self.mode = mode or exp.mode
        df = exp.frame
        self.dates = df.index[window:]
        n_rows = len(df)
        a_row, _ = time_split(n_rows, split)
        self.cols = list(df.columns)
        self.t_idx = self.cols.index(exp.target)
        vals = df.values.astype(np.float64)
        self.last = vals[window - 1:-1, self.t_idx]             # giá trị mục tiêu ở bước cuối mỗi cửa sổ
        self.y_true = vals[window:, self.t_idx]                 # giá trị thật cần dự báo (đơn vị gốc)
        if self.mode == "relative":
            X, y = self._relative(vals, a_row)
        elif self.mode == "minmax":
            X, y = self._minmax(vals, a_row)
        else:
            X, y = self._level(vals, a_row)
        n = len(X)
        a, b = time_split(n, split)
        self.sl = {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n)}
        self.X, self.y = X, y

    def _level(self, vals, a_row):
        self.mu, self.sd = vals[:a_row].mean(0), vals[:a_row].std(0) + 1e-9
        z = (vals - self.mu) / self.sd
        return make_windows(z, z[:, self.t_idx], self.window)

    def _minmax(self, vals, a_row):
        """Cách của bài mẫu: MinMax toàn cục fit trên train — không ngoại suy được khi giá test vượt vùng train."""
        self.lo, self.hi = vals[:a_row].min(0), vals[:a_row].max(0)
        z = (vals - self.lo) / (self.hi - self.lo + 1e-9)
        return make_windows(z, z[:, self.t_idx], self.window)

    def _relative(self, vals, a_row):
        level = [self.cols.index(c) for c in self.exp.level_cols]
        price = [i for i in range(len(self.cols)) if i not in level]
        lv = np.log1p(vals[:, level])
        self.lv_mu, self.lv_sd = lv[:a_row].mean(0), lv[:a_row].std(0) + 1e-9
        lvz = (lv - self.lv_mu) / self.lv_sd
        logp = np.log(vals[:, price])
        X = []
        for i in range(self.window, len(vals)):
            w = logp[i - self.window:i] - logp[i - 1]            # log-tỉ số so với bước cuối cửa sổ
            X.append(np.concatenate([w, lvz[i - self.window:i]], axis=1))
        X = np.stack(X)
        tgt = np.log(vals[self.window:, self.t_idx]) - np.log(vals[self.window - 1:-1, self.t_idx])
        self.ret_sd = tgt[:a_row - self.window].std() + 1e-12     # chia để mục tiêu có thang ~1 khi huấn luyện
        self.price_cols = [self.cols[i] for i in price] + [self.cols[i] for i in level]
        return X.astype(np.float32) * 10.0, (tgt / self.ret_sd).astype(np.float32)

    def to_original(self, pred, part="test"):
        """Đổi đầu ra mô hình (thang chuẩn hoá) về đơn vị gốc của biến mục tiêu."""
        pred = np.asarray(pred, dtype=np.float64).reshape(-1)
        if self.mode == "relative":
            return self.last[self.sl[part]] * np.exp(pred * self.ret_sd)
        if self.mode == "minmax":
            return pred * (self.hi[self.t_idx] - self.lo[self.t_idx]) + self.lo[self.t_idx]
        return pred * self.sd[self.t_idx] + self.mu[self.t_idx]

    def part(self, name):
        s = self.sl[name]
        return self.X[s], self.y[s]

    def truth(self, part="test"):
        return self.y_true[self.sl[part]]

    def naive(self, part="test"):
        """Dự báo ngây thơ (naive / persistence): giá trị ngày mai = giá trị hôm nay."""
        return self.last[self.sl[part]]

    def moving_average(self, k, part="test"):
        """Dự báo trung bình trượt: giá trị ngày mai = trung bình k giá trị mục tiêu gần nhất."""
        t = self.exp.frame[self.exp.target].values.astype(np.float64)
        ma = np.array([t[i - k:i].mean() for i in range(self.window, len(t))])
        return ma[self.sl[part]]

    def shapes(self):
        return {k: list(self.X[s].shape) for k, s in self.sl.items()}
