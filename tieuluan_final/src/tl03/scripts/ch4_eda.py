"""Khảo sát và phân tích khám phá (EDA) ba bộ dữ liệu chuỗi thời gian của Chương 4.

Ghi hình figures/ch4_eda_*.pdf và số liệu outputs/metrics/ch4_eda.json.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch4_data import KAGGLE, Prepared, aapl_frame, ecommerce_daily, gold_frame, time_split  # noqa: E402
from tl3.common import data_dir, save_json  # noqa: E402
from tl3.plot_style import C, SEQ, W, plt, vn_axis  # noqa: E402
from tl3.plot_style import save as _save  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402


def save(fig, name):
    """Gọn nhãn ngày trên trục x (3–6 mốc, định dạng rút gọn) rồi ghi hình vào figures/."""
    for ax in fig.axes:
        if isinstance(ax.xaxis.get_major_formatter(), (mdates.AutoDateFormatter, mdates.ConciseDateFormatter)):
            loc = mdates.AutoDateLocator(minticks=3, maxticks=6)
            ax.xaxis.set_major_locator(loc)
            ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
            ax.tick_params(axis="x", rotation=0)
    return _save(fig, name)

VN_DOW = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]


def acf(x, nlags=40):
    """Hàm tự tương quan mẫu ρ(k) = Σ(x_t − x̄)(x_{t+k} − x̄) / Σ(x_t − x̄)²."""
    x = np.asarray(x, float) - np.mean(x)
    den = np.sum(x * x)
    return np.array([np.sum(x[:len(x) - k] * x[k:]) / den for k in range(nlags + 1)])


def corr_heatmap(df, name, title, size=(W, 5.4), annot=True):
    corr = df.corr().values
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(df.columns)), df.columns, rotation=55, ha="right")
    ax.set_yticks(range(len(df.columns)), df.columns)
    ax.grid(False)
    if annot:
        for i in range(len(corr)):
            for j in range(len(corr)):
                ax.text(j, i, f"{corr[i, j]:.2f}".replace(".", ","), ha="center", va="center", fontsize=8.5,
                        color="white" if abs(corr[i, j]) > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title(title)
    save(fig, name)
    return pd.DataFrame(corr, index=df.columns, columns=df.columns)


def hist_grid(df, name, title, ncols=3, logcols=()):
    n = len(df.columns)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(W, 1.9 * nrows + 0.4))
    for ax, (c, col) in zip(axes.flat, zip(SEQ * 3, df.columns)):
        v = np.log1p(df[col]) if col in logcols else df[col]
        ax.hist(v, bins=35, color=c, alpha=0.85, edgecolor="white", linewidth=0.4)
        ax.set_title(("log1p " if col in logcols else "") + col, fontsize=10.5)
        ax.tick_params(labelsize=9)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.suptitle(title, fontweight="bold", fontsize=12.5)
    fig.tight_layout()
    save(fig, name)


def split_plot(series, n_rows, name, title, ylabel, logy=False):
    a, b = time_split(n_rows)
    fig, ax = plt.subplots(figsize=(W, 3.0))
    for (lo, hi), c, lab in (((0, a), C["blue"], "Train (70%)"), ((a, b), C["orange"], "Validation (15%)"),
                             ((b, n_rows), C["green"], "Test (15%)")):
        ax.plot(series.index[lo:hi], series.values[lo:hi], color=c, lw=1.4, label=lab)
    ax.axvline(series.index[a], color="gray", ls="--", lw=1)
    ax.axvline(series.index[b], color="gray", ls="--", lw=1)
    if logy:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    # chú giải dưới trục: chuỗi giá phủ kín cả góc trái (vàng) lẫn góc phải (AAPL)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3, frameon=False)
    save(fig, name)
    return {"train_end": str(series.index[a - 1].date()), "val_end": str(series.index[b - 1].date()),
            "test_start": str(series.index[b].date())}


def ecommerce_eda():
    raw = pd.read_csv(data_dir("ch4") / KAGGLE["ecommerce"][1], parse_dates=["Date"])
    d = ecommerce_daily()
    out = {"orders": len(raw), "customers": int(raw["Customer_ID"].nunique()), "raw_cols": raw.shape[1],
           "days": len(d), "start": str(d.index.min().date()), "end": str(d.index.max().date()),
           "categories": int(raw["Product_Category"].nunique()), "cities": int(raw["City"].nunique()),
           "missing": int(raw.isna().sum().sum()), "dup": int(raw.duplicated().sum()),
           "zero_days": int((d["Total_Amount"] == 0).sum()),
           "target": {k: float(v) for k, v in d["Total_Amount"].describe().items()},
           "orders_per_day": float(d["Transactions"].mean())}
    # 1) bốn chuỗi chính
    fig, axes = plt.subplots(4, 1, figsize=(W, 7.2), sharex=True)
    for ax, col, c, lab in zip(axes, ["Total_Amount", "Transactions", "Avg_Session", "Avg_Rating"],
                               [C["blue"], C["orange"], C["green"], C["purple"]],
                               ["Doanh thu ngày (TRY)", "Số đơn", "Phiên TB (phút)", "Đánh giá TB"]):
        ax.plot(d.index, d[col], color=c, lw=1.0, alpha=0.75)
        ax.plot(d.index, d[col].rolling(14, center=True).mean(), color="black", lw=1.6)
        ax.set_ylabel(lab, fontsize=10.5)
    vn_axis(axes[0])
    axes[0].set_title("E-commerce: chuỗi hành vi khách hàng theo ngày (đường đen: TB trượt 14 ngày)")
    fig.tight_layout()
    save(fig, "ch4_eda_ecom_series")
    hist_grid(d, "ch4_eda_ecom_hist", "Phân phối 9 đặc trưng ngày của dữ liệu e-commerce")
    corr = corr_heatmap(d, "ch4_eda_ecom_corr", "Ma trận tương quan Pearson các đặc trưng ngày")
    out["corr_amount_tx"] = float(corr.loc["Total_Amount", "Transactions"])
    out["corr_amount_qty"] = float(corr.loc["Total_Amount", "Quantity"])
    out["corr_amount_rating"] = float(corr.loc["Total_Amount", "Avg_Rating"])
    # 2) mùa vụ: thứ trong tuần, tháng; tự tương quan
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.3), gridspec_kw={"width_ratios": [1.3, 1.4, 1.1]})
    dow = d["Total_Amount"].groupby(d.index.dayofweek).mean()
    axes[0].bar(VN_DOW, dow.values / 1e3, color=SEQ[:7])
    axes[0].set_title("TB theo thứ")
    axes[0].tick_params(axis="x", labelsize=8.5)
    axes[0].set_ylabel("nghìn TRY")
    mon = d["Total_Amount"].resample("MS").sum()
    axes[1].bar(mon.index.strftime("%m/%y"), mon.values / 1e6, color=C["teal"])
    axes[1].set_title("Tổng theo tháng")
    axes[1].set_ylabel("triệu TRY")
    axes[1].tick_params(axis="x", rotation=90, labelsize=8.5)
    for i, lbl in enumerate(axes[1].get_xticklabels()):
        lbl.set_visible(i % 2 == 0)                      # một nhãn mỗi hai tháng cho dễ đọc
    r = acf(d["Total_Amount"], 30)
    axes[2].bar(range(len(r)), r, color=C["purple"], width=0.6)
    axes[2].axhline(1.96 / np.sqrt(len(d)), color="red", ls="--", lw=1)
    axes[2].axhline(-1.96 / np.sqrt(len(d)), color="red", ls="--", lw=1)
    axes[2].set_title("ACF doanh thu")
    axes[2].set_xlabel("độ trễ (ngày)")
    fig.tight_layout()
    save(fig, "ch4_eda_ecom_season")
    out["acf1"], out["acf7"] = float(r[1]), float(r[7])
    out["dow_max"] = VN_DOW[int(dow.values.argmax())]
    out["dow_min"] = VN_DOW[int(dow.values.argmin())]
    out["dow_ratio"] = float(dow.max() / dow.min())
    out["split"] = split_plot(d["Total_Amount"], len(d), "ch4_eda_ecom_split",
                              "Chia theo thời gian: doanh thu ngày e-commerce", "TRY")
    # 3) cơ cấu đơn hàng
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0))
    cat = raw.groupby("Product_Category")["Total_Amount"].sum().sort_values() / 1e6
    axes[0].barh(cat.index, cat.values, color=SEQ[:len(cat)])
    axes[0].set_xlabel("triệu TRY")
    axes[0].set_title("Doanh thu theo danh mục")
    dev = raw["Device_Type"].value_counts()
    axes[1].pie(dev.values, labels=dev.index, autopct=lambda p: f"{p:.1f}%".replace(".", ","), colors=SEQ,
                wedgeprops={"edgecolor": "white"}, textprops={"fontsize": 10})
    axes[1].set_title("Đơn theo thiết bị")
    fig.tight_layout()
    save(fig, "ch4_eda_ecom_mix")
    out["top_category"] = cat.index[-1]
    out["sample"] = raw.head(5)[["Order_ID", "Date", "Product_Category", "Quantity", "Total_Amount"]].astype(str) \
        .values.tolist()
    return out


def aapl_eda():
    d = aapl_frame()
    ret = np.log(d["close"]).diff().dropna()
    out = {"rows": len(d), "n_obs": len(d), "start": str(d.index.min().date()), "end": str(d.index.max().date()),
           "missing": int(d.isna().sum().sum()), "dup": int(d.index.duplicated().sum()),
           "close_first": float(d["close"].iloc[0]), "close_last": float(d["close"].iloc[-1]),
           "close_max": float(d["close"].max()), "growth_x": float(d["close"].iloc[-1] / d["close"].iloc[0]),
           "ret_mean_pct": float(100 * ret.mean()), "ret_sd_pct": float(100 * ret.std()),
           "ret_kurt": float(ret.kurt()), "ret_min_pct": float(100 * ret.min()), "ret_max_pct": float(100 * ret.max()),
           "ret_min_date": str(ret.idxmin().date()), "vol_mean": float(d["volume"].mean())}
    fig, axes = plt.subplots(2, 2, figsize=(W, 5.0))
    axes[0, 0].plot(d.index, d["close"], color=C["blue"], lw=1)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("Close (thang log)")
    axes[0, 1].plot(d.index[-750:], (d["close"] - d["open"]).iloc[-750:], color=C["orange"], lw=0.8)
    axes[0, 1].set_title("Close − Open (3 năm cuối)")
    axes[1, 0].plot(d.index[-750:], (d["high"] - d["low"]).iloc[-750:], color=C["green"], lw=0.8)
    axes[1, 0].set_title("High − Low (3 năm cuối)")
    axes[1, 1].plot(d.index, d["volume"] / 1e6, color=C["purple"], lw=0.5)
    axes[1, 1].set_title("Volume (triệu cổ phiếu)")
    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=30, labelsize=9)
    fig.suptitle("AAPL: tổng quan giá và khối lượng 1980–2024", fontweight="bold")
    fig.tight_layout()
    save(fig, "ch4_eda_aapl_overview")
    hist_grid(d, "ch4_eda_aapl_hist", "Phân phối sáu biến AAPL (Volume vẽ theo log1p)", logcols=("volume",))
    corr = corr_heatmap(d, "ch4_eda_aapl_corr", "Ma trận tương quan các biến AAPL", size=(W * 0.85, 4.6))
    out["corr_close_volume"] = float(corr.loc["close", "volume"])
    yearly = d["close"].resample("YE").last()
    yr = yearly.pct_change().dropna() * 100
    fig, axes = plt.subplots(2, 1, figsize=(W, 4.8), sharex=True)
    axes[0].plot(yearly.index.year, yearly.values, "o-", color=C["navy"], ms=3.5)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Close cuối năm (USD)")
    axes[0].set_title("AAPL: giá cuối năm và lợi suất năm")
    axes[1].bar(yr.index.year, yr.values, color=[C["green"] if v >= 0 else C["red"] for v in yr.values])
    axes[1].set_ylabel("Lợi suất năm (%)")
    fig.tight_layout()
    save(fig, "ch4_eda_aapl_yearly")
    out["best_year"], out["best_year_pct"] = int(yr.idxmax().year), float(yr.max())
    out["worst_year"], out["worst_year_pct"] = int(yr.idxmin().year), float(yr.min())
    # lợi suất: biến động trượt, phân phối đuôi dày
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9))
    vol = ret.rolling(60).std() * np.sqrt(252) * 100
    axes[0].plot(vol.index, vol.values, color=C["red"], lw=0.8)
    axes[0].set_title("Biến động năm hoá (60 phiên)")
    axes[0].set_ylabel("%")
    axes[0].tick_params(axis="x", rotation=30, labelsize=9)
    z = (ret - ret.mean()) / ret.std()
    axes[1].hist(z.clip(-8, 8), bins=80, density=True, color=C["sky"], alpha=0.8, label="log-lợi suất (chuẩn hoá)")
    xs = np.linspace(-8, 8, 300)
    axes[1].plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color="black", lw=1.4, label="N(0,1)")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-5, 1)
    axes[1].set_title("Đuôi dày lợi suất ngày")
    axes[1].legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "ch4_eda_aapl_returns")
    out["acf1_ret"] = float(acf(ret, 1)[1])
    out["acf1_abs"] = float(acf(ret.abs(), 1)[1])
    out["split"] = split_plot(d["close"], len(d), "ch4_eda_aapl_split",
                              "Chia theo thời gian: giá đóng cửa AAPL (thang log)", "USD", logy=True)
    out["sample"] = d.tail(5).round(3).reset_index().astype(str).values.tolist()
    return out


def gold_eda():
    d = gold_frame()
    ret = np.log(d["GLD_Close"]).diff().dropna()
    out = {"rows": len(d), "n_obs": len(d), "cols_raw": 81, "cols_used": d.shape[1], "start": str(d.index.min().date()),
           "end": str(d.index.max().date()), "missing": int(d.isna().sum().sum()),
           "target": {k: float(v) for k, v in d["GLD_Close"].describe().items()},
           "ret_sd_pct": float(100 * ret.std()), "drawdown_pct": float(100 * (d["GLD_Close"].min() /
                                                                             d["GLD_Close"].iloc[0] - 1))}
    norm = d[["GLD_Close", "SP500", "USD_Index", "Silver_Fut", "Brent"]] / d[["GLD_Close", "SP500", "USD_Index",
                                                                           "Silver_Fut", "Brent"]].iloc[0] * 100
    fig, ax = plt.subplots(figsize=(W, 3.3))
    for col, c in zip(norm.columns, SEQ):
        ax.plot(norm.index, norm[col], color=c, lw=1.6 if col == "GLD_Close" else 1.0, label=col)
    ax.axhline(100, color="gray", lw=0.8, ls=":")
    ax.set_ylabel("Chỉ số (ngày đầu = 100)")
    ax.set_title("Vàng (GLD) và các thị trường liên quan, 2011–2018")
    ax.legend(ncol=3, fontsize=9.5)
    save(fig, "ch4_eda_gold_series")
    rets = np.log(d).diff().dropna()
    corr = corr_heatmap(rets, "ch4_eda_gold_corr", "Tương quan log-lợi suất ngày giữa vàng và các chỉ báo")
    out["corr_ret"] = {c: float(corr.loc["GLD_Close", c]) for c in ("Silver_Fut", "GDX", "USD_Index", "SP500",
                                                                     "EURUSD", "Brent")}
    rc = rets["GLD_Close"].rolling(120).corr(rets["USD_Index"])
    rs = rets["GLD_Close"].rolling(120).corr(rets["SP500"])
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.plot(rc.index, rc, color=C["red"], label="GLD ~ USD Index")
    ax.plot(rs.index, rs, color=C["blue"], label="GLD ~ S&P 500")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_title("Tương quan trượt 120 phiên của lợi suất")
    ax.legend()
    save(fig, "ch4_eda_gold_rollcorr")
    hist_grid(d[["GLD_Close", "GLD_Volume", "SP500", "USD_Index", "EURUSD", "Brent"]], "ch4_eda_gold_hist",
              "Phân phối mức giá của vàng và năm chỉ báo", logcols=("GLD_Volume",))
    out["split"] = split_plot(d["GLD_Close"], len(d), "ch4_eda_gold_split",
                              "Chia theo thời gian: giá đóng cửa GLD", "USD")
    return out


def windows_info():
    """Kích thước tensor cửa sổ của ba thực nghiệm sau khi chuẩn hoá."""
    from tl3.ch4_data import experiment
    return {n: Prepared(experiment(n)).shapes() for n in ("ecommerce", "aapl", "gold")}


if __name__ == "__main__":
    save_json("ch4_eda", {"ecom": ecommerce_eda(), "aapl": aapl_eda(), "gold": gold_eda(), "shapes": windows_info()})
