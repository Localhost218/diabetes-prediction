"""Hình khám phá dữ liệu (EDA) của Chương 2, 3, 4 và số liệu thống kê đi kèm (outputs/metrics/eda.json).

    python scripts/make_eda.py [ch2 ch3 ch4]

Mọi hình vẽ ở bề rộng vùng chữ (W = 6,3 inch) để chữ trong hình có cùng cỡ với thân bài.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import C, SEQ, H1, W, plt, save, vn  # noqa: E402

from tl import data_seq as dq  # noqa: E402
from tl import data_tabular as dt  # noqa: E402
from tl.common import merge_json  # noqa: E402

EDA = "eda"
SEQ4 = [C["blue"], C["green"], C["orange"], C["red"]]


def heat(corr, names, fname, size, fs=9):
    fig, ax = plt.subplots(figsize=size)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=40, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            v = corr.values[i, j]
            ax.text(j, i, vn(v, 2), ha="center", va="center", fontsize=fs, color="white" if abs(v) > 0.55 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    save(fig, fname)


def describe(df, cols, names):
    """Thống kê mô tả cho bảng: trung bình, độ lệch chuẩn, min, tứ phân vị, max, tỉ lệ thiếu."""
    out = []
    for c, n in zip(cols, names):
        s = df[c]
        out.append([n, s.mean(), s.std(), s.min(), s.quantile(0.25), s.median(), s.quantile(0.75), s.max(), s.isna().mean()])
    return out


def label_bars(ax, bars, values, total=None, d=0, fs=10):
    for b, v in zip(bars, values):
        txt = vn(v, d) + (f"\n({vn(100 * v / total, 1)} %)" if total else "")
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), txt, ha="center", va="bottom", fontsize=fs)


# ============================================================================== Chương 2
def ch2_diabetes():
    df = dt.diabetes_frame()
    pos = df.diabetes.mean()
    fig, ax = plt.subplots(figsize=(W * 0.72, 3.3))
    cnt = df.diabetes.value_counts().sort_index()
    bars = ax.bar(["Không mắc (0)", "Mắc tiểu đường (1)"], cnt.values, color=[C["blue"], C["red"]], width=0.55)
    label_bars(ax, bars, cnt.values, len(df))
    ax.set_ylabel("Số hồ sơ")
    ax.set_ylim(0, cnt.max() * 1.25)
    save(fig, "f2_diab_target")

    from scipy.stats import gaussian_kde
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.6))
    for ax, col, lab in zip(axs.ravel(), ["blood_glucose_level", "HbA1c_level", "bmi", "age"],
                            ["Đường huyết (mg/dL)", "HbA1c (%)", "BMI", "Tuổi"]):
        grid = np.linspace(df[col].quantile(0.001), df[col].quantile(0.999), 300)
        for k, c, name in [(0, C["blue"], "Không mắc"), (1, C["red"], "Mắc bệnh")]:
            v = df.loc[df.diabetes == k, col].to_numpy(float)
            dens = gaussian_kde(v, bw_method=0.35 if col in ("blood_glucose_level", "HbA1c_level") else 0.15)(grid)
            ax.fill_between(grid, dens, color=c, alpha=0.25)
            ax.plot(grid, dens, color=c, lw=2, label=name)
        ax.set_xlabel(lab)
        ax.set_yticks([])
    axs[0, 0].legend()
    fig.tight_layout()
    save(fig, "f2_diab_kde")

    fig, axs = plt.subplots(1, 2, figsize=(W, 3.3), sharey=True)
    for ax, col, bins, lab, c in [(axs[0], "HbA1c_level", [3, 5.7, 6.0, 6.5, 7.0, 9.1], "HbA1c (%)", C["orange"]),
                                  (axs[1], "blood_glucose_level", [70, 100, 126, 160, 200, 301], "Đường huyết (mg/dL)", C["purple"])]:
        g = pd.cut(df[col], bins, right=False)
        r = df.groupby(g, observed=True).diabetes.mean() * 100
        bars = ax.bar([f"{int(i.left) if i.left >= 10 else i.left}–{int(i.right) if i.right >= 10 else i.right}" for i in r.index],
                      r.values, color=c)
        label_bars(ax, bars, r.values, d=1, fs=9)
        ax.set_xlabel(lab)
        ax.tick_params(axis="x", rotation=30)
    axs[0].set_ylabel("Tỉ lệ mắc bệnh (%)")
    axs[0].set_ylim(0, 115)
    fig.tight_layout()
    save(fig, "f2_diab_rates")

    num = df.assign(gender_male=(df.gender == "Male").astype(int),
                    smoker=df.smoking_history.isin(["current", "former", "ever", "not current"]).astype(int))
    cols = ["age", "hypertension", "heart_disease", "bmi", "HbA1c_level", "blood_glucose_level", "gender_male", "smoker", "diabetes"]
    corr = num[cols].corr()
    names = ["Tuổi", "Tăng huyết áp", "Bệnh tim", "BMI", "HbA1c", "Đường huyết", "Nam giới", "Có hút thuốc", "Tiểu đường"]
    heat(corr, names, "f2_diab_corr", (W, 5.4))
    sample = df.sample(5, random_state=7)
    merge_json(EDA, "diabetes", {
        "n": len(df), "n_features": df.shape[1] - 1, "pos_rate": pos, "n_pos": int(df.diabetes.sum()),
        "corr_glucose": corr.loc["blood_glucose_level", "diabetes"], "corr_hba1c": corr.loc["HbA1c_level", "diabetes"],
        "corr_age": corr.loc["age", "diabetes"], "corr_bmi": corr.loc["bmi", "diabetes"],
        "rate_glu200": df.loc[df.blood_glucose_level >= 200, "diabetes"].mean(),
        "rate_hba65": df.loc[df.HbA1c_level >= 6.6, "diabetes"].mean(),
        "rate_hba_low": df.loc[df.HbA1c_level < 5.7, "diabetes"].mean(),
        "noinfo_smoke": (df.smoking_history == "No Info").mean(),
        "n_glucose_levels": int(df.blood_glucose_level.nunique()), "n_hba1c_levels": int(df.HbA1c_level.nunique()),
        "dup_rows": int(df.duplicated().sum()),
        "describe": describe(df, ["age", "bmi", "HbA1c_level", "blood_glucose_level"], ["Tuổi", "BMI", "HbA1c (\\%)", "Đường huyết (mg/dL)"]),
        "sample": sample.astype(str).values.tolist(), "sample_cols": list(sample.columns)})


def ch2_house():
    df = dt.house_frame()
    df["ppm2"] = df.Price * 1000 / df.Area
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.4))
    for ax, col, bins, lab, c in [(axs[0, 0], "Price", 45, "Giá bán (tỷ đồng)", C["blue"]),
                                  (axs[0, 1], "Area", 60, "Diện tích (m²)", C["orange"]),
                                  (axs[1, 0], "log_area", 60, "log(1 + diện tích)", C["green"]),
                                  (axs[1, 1], "ppm2", 60, "Đơn giá (triệu đồng/m²)", C["purple"])]:
        v = df[col].clip(upper=df[col].quantile(0.995))
        ax.hist(v, bins=bins, color=c, alpha=0.85)
        ax.set_xlabel(lab)
        ax.set_title(f"độ lệch = {vn(df[col].skew(), 2)}", fontsize=10.5, fontweight="normal")
    axs[0, 0].set_ylabel("Số tin đăng")
    axs[1, 0].set_ylabel("Số tin đăng")
    fig.tight_layout()
    save(fig, "f2_house_dist")

    fig, ax = plt.subplots(figsize=(W, 3.8))
    s = df.sample(6000, random_state=1)
    sc = ax.scatter(s.Area, s.Price, c=s["Access Road"].clip(upper=20), s=7, alpha=0.55, cmap="viridis")
    ax.set_xscale("log")
    ax.set_xlabel("Diện tích (m², thang log)")
    ax.set_ylabel("Giá bán (tỷ đồng)")
    fig.colorbar(sc, ax=ax, label="Độ rộng đường vào (m)")
    save(fig, "f2_house_scatter")

    fig, ax = plt.subplots(figsize=(W, 3.6))
    top = df.province.value_counts().index[:8]
    data = [df.loc[df.province == p, "Price"] for p in top]
    bp = ax.boxplot(data, tick_labels=[p.replace("Hồ Chí Minh", "TP.HCM") for p in top], showfliers=False, patch_artist=True,
                    medianprops=dict(color="k", lw=1.6))
    for patch, c in zip(bp["boxes"], SEQ + [C["navy"]]):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("Giá bán (tỷ đồng)")
    ax.tick_params(axis="x", rotation=25)
    save(fig, "f2_house_prov")

    miss = df[["Frontage", "Access Road", "House direction", "Balcony direction", "Floors", "Bedrooms", "Bathrooms",
               "Legal status", "Furniture state"]].isna().mean().sort_values() * 100
    fig, ax = plt.subplots(figsize=(W, 3.3))
    names = {"Frontage": "Mặt tiền", "Access Road": "Đường vào", "House direction": "Hướng nhà",
             "Balcony direction": "Hướng ban công", "Floors": "Số tầng", "Bedrooms": "Phòng ngủ", "Bathrooms": "Phòng tắm",
             "Legal status": "Pháp lý", "Furniture state": "Nội thất"}
    bars = ax.barh([names[c] for c in miss.index], miss.values, color=[C["red"] if v > 40 else C["orange"] if v > 15 else C["green"] for v in miss.values])
    for b, v in zip(bars, miss.values):
        ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{vn(v, 1)} %", va="center", fontsize=10)
    ax.set_xlabel("Tỉ lệ thiếu (%)")
    ax.set_xlim(0, 100)
    save(fig, "f2_house_missing")

    cols = ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms", "Price"]
    corr = df[cols].corr()
    heat(corr, ["Diện tích", "Mặt tiền", "Đường vào", "Số tầng", "Phòng ngủ", "Phòng tắm", "Giá"], "f2_house_corr", (W * 0.86, 4.6), fs=10)
    sample = df.sample(5, random_state=3)
    merge_json(EDA, "house", {
        "n": len(df), "n_province": df.province.nunique(), "n_district": df.district.nunique(),
        "price_mean": df.Price.mean(), "price_median": df.Price.median(), "price_min": df.Price.min(),
        "price_max": df.Price.max(), "price_skew": df.Price.skew(), "area_skew": df.Area.skew(),
        "logarea_skew": df.log_area.skew(), "area_median": df.Area.median(), "ppm2_median": df.ppm2.median(),
        "corr_area": corr.loc["Area", "Price"], "corr_road": corr.loc["Access Road", "Price"],
        "corr_floors": corr.loc["Floors", "Price"], "corr_bed": corr.loc["Bedrooms", "Price"],
        "share_hcm_hn": df.province.isin(["Hồ Chí Minh", "Hà Nội"]).mean(),
        "median_by_prov": df.groupby("province").Price.median().loc[top[:6]].round(2).to_dict(),
        "miss_frontage": df.Frontage.isna().mean(), "miss_road": df["Access Road"].isna().mean(),
        "miss_furniture": df["Furniture state"].isna().mean(), "miss_direction": df["House direction"].isna().mean(),
        "miss_balcony": df["Balcony direction"].isna().mean(),
        "describe": describe(df, ["Area", "Frontage", "Access Road", "Floors", "Bedrooms", "Bathrooms", "Price", "ppm2"],
                             ["Diện tích (m²)", "Mặt tiền (m)", "Đường vào (m)", "Số tầng", "Phòng ngủ", "Phòng tắm",
                              "Giá (tỷ đồng)", "Đơn giá (tr.đ/m²)"]),
        "sample": [[r.Area, r.Floors, r.Bedrooms, r.Frontage, r["Access Road"], r.district, r.province, r.Price]
                   for _, r in sample.iterrows()]})


def ch2_reviews():
    df = dt.reviews_frame()
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.3))
    r = df.Rating.value_counts().sort_index()
    axs[0].bar([f"{i}★" for i in r.index], r.values, color=[C["red"], C["red"], C["gray"], C["green"], C["green"]])
    axs[0].set_ylabel("Số nhận xét")
    axs[0].set_title("Số sao đánh giá")
    s = df.sentiment.value_counts().sort_index()
    bars = axs[1].bar(["Tiêu cực", "Trung tính", "Tích cực"], s.values, color=[C["red"], C["gray"], C["green"]])
    for b, v in zip(bars, s.values):
        axs[1].text(b.get_x() + b.get_width() / 2, v, f"{vn(100 * v / len(df), 1)} %", ha="center", va="bottom", fontsize=10)
    axs[1].set_title("Nhãn cảm xúc 3 lớp")
    axs[1].set_ylim(0, s.max() * 1.15)
    fig.tight_layout()
    save(fig, "f2_rev_dist")

    fig, ax = plt.subplots(figsize=(W, 3.2))
    for k, c in zip(range(3), [C["red"], C["gray"], C["green"]]):
        ax.hist(df.loc[df.sentiment == k, "n_words"], bins=40, range=(0, 120), density=True, alpha=0.45, color=c,
                label=dt.SENT_NAMES[k])
    ax.axvline(df.n_words.mean(), color="k", ls="--", lw=1.2)
    ax.set_xlabel("Số từ của nhận xét")
    ax.set_ylabel("Mật độ")
    ax.legend()
    save(fig, "f2_rev_len")

    dep = pd.crosstab(df["Department Name"], df.sentiment, normalize="index") * 100
    dep = dep.loc[df["Department Name"].value_counts().index]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    left = np.zeros(len(dep))
    for k, c in zip(range(3), [C["red"], C["gray"], C["green"]]):
        ax.barh(dep.index, dep[k], left=left, color=c, label=dt.SENT_NAMES[k])
        for i, v in enumerate(dep[k]):
            if v > 6:
                ax.text(left[i] + v / 2, i, f"{vn(v, 0)} %", ha="center", va="center", color="white", fontsize=9.5)
        left += dep[k].to_numpy()
    ax.set_xlabel("Tỉ lệ nhãn trong nhóm sản phẩm (%)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    save(fig, "f2_rev_dept")

    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(preprocessor=dt.clean_text, stop_words="english", min_df=20)
    M = cv.fit_transform(df.text)
    voc = np.array(cv.get_feature_names_out())
    fig, axs = plt.subplots(1, 2, figsize=(W, 4.6))
    top = {}
    for ax, k, c, t in [(axs[0], 2, C["green"], "Tích cực"), (axs[1], 0, C["red"], "Tiêu cực")]:
        a = np.asarray(M[df.sentiment.to_numpy() == k].sum(0)).ravel() + 1
        b = np.asarray(M[df.sentiment.to_numpy() != k].sum(0)).ravel() + 1
        lo = np.log(a / a.sum()) - np.log(b / b.sum())             # tỉ số log-odds có làm trơn cộng 1
        idx = np.argsort(lo)[-12:]
        ax.barh(voc[idx], lo[idx], color=c)
        ax.set_title(t)
        ax.set_xlabel("log-odds")
        top[k] = voc[idx][::-1].tolist()
    fig.tight_layout()
    save(fig, "f2_rev_words")
    sample = df.groupby("sentiment").sample(2, random_state=11).sort_values("Rating")
    merge_json(EDA, "reviews", {
        "n_raw": 23486, "n": len(df), "share_pos": (df.sentiment == 2).mean(), "share_neu": (df.sentiment == 1).mean(),
        "share_neg": (df.sentiment == 0).mean(), "words_mean": df.n_words.mean(), "words_median": df.n_words.median(),
        "top_pos": top[2][:6], "top_neg": top[0][:6],
        "words_by_class": df.groupby("sentiment").n_words.mean().round(1).tolist(),
        "neg_share_by_dept": (dep[0] / 100).round(4).to_dict(),
        "sample": [[int(r.Rating), r["Review Text"][:110] + ("..." if len(r["Review Text"]) > 110 else ""), dt.SENT_NAMES[r.sentiment]]
                   for _, r in sample.head(5).iterrows()]})


# ============================================================================== Chương 3
def ch3():
    from tl import data_vision as dv
    for name in ("svhn", "eurosat"):
        d = dv.load(name)
        rng = np.random.default_rng(5)
        if name == "svhn":
            fig, axs = plt.subplots(4, 10, figsize=(W, 3.15))
            for c in range(10):
                idx = rng.choice(np.flatnonzero(d["y_train"] == c), 4, replace=False)
                for r in range(4):
                    axs[r, c].imshow(d["x_train"][idx[r]].transpose(1, 2, 0))
                    axs[r, c].axis("off")
                axs[0, c].set_title(d["classes"][c], fontsize=11, fontweight="normal")
            fig.subplots_adjust(wspace=0.06, hspace=0.06)
        else:                                              # tên lớp EuroSAT dài: mỗi lớp một hàng, nhãn bên trái
            fig, axs = plt.subplots(10, 7, figsize=(W, 9.0))
            for c in range(10):
                idx = rng.choice(np.flatnonzero(d["y_train"] == c), 7, replace=False)
                for r in range(7):
                    ax = axs[c, r]
                    ax.imshow(d["x_train"][idx[r]].transpose(1, 2, 0))
                    ax.set_xticks([])
                    ax.set_yticks([])
                    for s in ax.spines.values():
                        s.set_visible(False)
                axs[c, 0].set_ylabel(d["classes"][c], rotation=0, ha="right", va="center", fontsize=11, labelpad=6)
            fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0.24)
        save(fig, f"f3_{name}_grid")

        tr, te = np.bincount(d["y_train"], minlength=10), np.bincount(d["y_test"], minlength=10)
        fig, ax = plt.subplots(figsize=(W, 3.4))
        x = np.arange(10)
        ax.bar(x - 0.2, tr, 0.4, label="Huấn luyện", color=C["blue"])
        ax.bar(x + 0.2, te, 0.4, label="Kiểm thử", color=C["orange"])
        ax.set_xticks(x, d["classes"], rotation=35 if name == "eurosat" else 0, ha="right" if name == "eurosat" else "center")
        ax.set_ylabel("Số ảnh")
        ax.legend()
        save(fig, f"f3_{name}_classes")

        sub = d["x_train"][rng.choice(len(d["x_train"]), 4000, replace=False)]
        fig, ax = plt.subplots(figsize=(W, 2.9))
        for ch, c, lab in zip(range(3), ["#D55E00", "#009E73", "#0072B2"], ["Đỏ (R)", "Lục (G)", "Lam (B)"]):
            ax.hist(sub[:, ch].ravel(), bins=64, range=(0, 255), density=True, histtype="stepfilled", alpha=0.25, color=c)
            ax.hist(sub[:, ch].ravel(), bins=64, range=(0, 255), density=True, histtype="step", lw=2, color=c, label=lab)
        ax.set_xlabel("Cường độ điểm ảnh")
        ax.set_ylabel("Mật độ")
        ax.legend()
        save(fig, f"f3_{name}_rgb")

        rows = 1 if name == "svhn" else 2                  # EuroSAT 2x5 để tên lớp đầy đủ không chồng nhau
        fig, axs = plt.subplots(rows, 10 // rows, figsize=(W, 1.25 if rows == 1 else 3.2))
        for c, ax in zip(range(10), axs.ravel()):
            m = d["x_train"][d["y_train"] == c].mean(0).transpose(1, 2, 0) / 255
            m = (m - m.min()) / (m.max() - m.min())                # kéo giãn tương phản để thấy “dáng” trung bình
            ax.imshow(m)
            ax.axis("off")
            lab = d["classes"][c]
            if rows == 2 and len(lab) > 9:                 # tên dài xuống dòng sau từ đầu tiên
                lab = lab.replace(" ", "\n", 1)
            ax.set_title(lab, fontsize=10.5 if rows == 2 else 10, fontweight="normal")
        save(fig, f"f3_{name}_mean")
        xf = d["x_train"].astype(np.float32) / 255
        merge_json(EDA, name, {"n_train": len(d["y_train"]), "n_test": len(d["y_test"]),
                               "size": d["x_train"].shape[-1], "classes": d["classes"],
                               "mean": xf.mean((0, 2, 3)), "std": xf.std((0, 2, 3)),
                               "max_class_share": tr.max() / tr.sum(), "min_class_share": tr.min() / tr.sum(),
                               "min_class_n": int(tr.min()), "max_class_n": int(tr.max())})


# ============================================================================== Chương 4
def ch4():
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    from statsmodels.tsa.stattools import adfuller
    frames = {t: dq.load_stock(t) for t in dq.TICKERS}
    fig, axs = plt.subplots(2, 2, figsize=(W, 5.2), sharex=True)
    for ax, (t, df), c in zip(axs.ravel(), frames.items(), SEQ4):
        ax.plot(df.index, df.Close / 1000, color=c, lw=1.1, label="Giá đóng cửa")
        ax.plot(df.index, df.SMA50 / 1000, color="k", lw=0.9, alpha=0.7, label="SMA50")
        ax2 = ax.twinx()
        ax2.bar(df.index, df.Volume / 1e6, color=c, alpha=0.18, width=2)
        ax2.set_ylim(0, df.Volume.quantile(0.995) / 1e6 * 4)
        ax2.set_yticks([])
        ax2.grid(False)
        ax.set_title(f"{t}.VN")
        ax.tick_params(axis="x", rotation=30)
    axs[0, 0].set_ylabel("Nghìn đồng")
    axs[1, 0].set_ylabel("Nghìn đồng")
    axs[0, 0].legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "f4_stock_price")

    fpt = frames["FPT"]
    last = fpt.iloc[-70:]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(W, 4.6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    up = last.Close >= last.Open
    xs = np.arange(len(last))
    for col, mask in ((C["green"], up), (C["red"], ~up)):
        ax.vlines(xs[mask], last.Low[mask] / 1000, last.High[mask] / 1000, color=col, lw=1)
        ax.bar(xs[mask], (last.Close - last.Open)[mask].abs() / 1000, bottom=np.minimum(last.Open, last.Close)[mask] / 1000,
               color=col, width=0.7)
    ax.plot(xs, last.SMA20 / 1000, color=C["blue"], lw=1.6, label="SMA20")
    ax.plot(xs, last.SMA50 / 1000, color=C["orange"], lw=1.6, label="SMA50")
    ax.set_ylabel("Nghìn đồng")
    ax.legend(loc="upper right")
    axr.plot(xs, last.RSI14, color=C["purple"], lw=1.6)
    axr.axhspan(70, 100, color=C["red"], alpha=0.1)
    axr.axhspan(0, 30, color=C["green"], alpha=0.1)
    axr.set_ylim(0, 100)
    axr.set_ylabel("RSI-14")
    ticks = xs[::10]
    axr.set_xticks(ticks, [last.index[i].strftime("%d/%m") for i in ticks])
    fig.tight_layout()
    save(fig, "f4_stock_candle")

    rets = pd.DataFrame({t: df.Close.pct_change() * 100 for t, df in frames.items()}).dropna()
    fig, axs = plt.subplots(2, 1, figsize=(W, 5.4))
    for t, c in zip(dq.TICKERS, SEQ4):
        rets[t].plot.kde(ax=axs[0], color=c, lw=1.8, label=t, bw_method=0.15)
        (rets[t].rolling(30).std()).plot(ax=axs[1], color=c, lw=1.0, label=t)
    axs[0].set_xlim(-8, 8)
    axs[0].set_xlabel("Lợi suất ngày (%)")
    axs[0].set_ylabel("Mật độ")
    axs[0].legend(ncol=4)
    axs[1].set_ylabel("Biến động 30 phiên (%)")
    axs[1].set_xlabel("")
    fig.tight_layout()
    save(fig, "f4_stock_returns")
    corr = rets.corr()
    fig, ax = plt.subplots(figsize=(W * 0.55, 3.0))
    im = ax.imshow(corr.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), dq.TICKERS)
    ax.set_yticks(range(4), dq.TICKERS)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, vn(corr.values[i, j], 2), ha="center", va="center", fontsize=10,
                    color="white" if corr.values[i, j] > 0.6 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "f4_stock_corr")

    r = fpt.Close.pct_change().dropna() * 100
    fig, axs = plt.subplots(3, 1, figsize=(W, 6.6))
    plot_acf(fpt.Close, lags=60, ax=axs[0], title="ACF của giá đóng cửa FPT", markersize=3)
    plot_acf(r, lags=30, ax=axs[1], title="ACF của lợi suất ngày FPT", markersize=3, zero=False)
    plot_pacf(r, lags=30, ax=axs[2], title="PACF của lợi suất ngày FPT", markersize=3, zero=False, method="ywm")
    for a in axs:
        a.set_xlabel("Độ trễ (phiên)")
    fig.tight_layout()
    save(fig, "f4_acf")
    adf_p = adfuller(fpt.Close.dropna(), autolag="AIC")
    adf_r = adfuller(r, autolag="AIC")
    w = dq.stock_windows("FPT")
    sample = fpt.dropna().iloc[[60, 900, 1800, 2700, -1]]
    merge_json(EDA, "stock", {
        "n": {t: len(df) for t, df in frames.items()}, "n_total": sum(len(df) for df in frames.values()),
        "start": str(frames["FPT"].index.min().date()), "end": str(frames["FPT"].index.max().date()),
        "ret_std": rets.std().to_dict(), "ret_kurt": rets.kurt().to_dict(),
        "corr_min": float(corr.values[np.triu_indices(4, 1)].min()), "corr_max": float(corr.values[np.triu_indices(4, 1)].max()),
        "adf_price_stat": adf_p[0], "adf_price_p": adf_p[1], "adf_ret_stat": adf_r[0], "adf_ret_p": adf_r[1],
        "acf1_ret": float(r.autocorr(1)), "n_train": len(w["y_train"]), "n_test": len(w["y_test"]),
        "test_start": str(w["dates_test"][0].date()), "train_end": str(w["dates_train"][-1].date()),
        "train_max_close": float(fpt.Close.loc[:w["dates_train"][-1]].max()),
        "test_max_close": float(fpt.Close.loc[w["dates_test"][0]:].max()),
        "growth": float(fpt.Close.max() / fpt.Close.iloc[0]),
        "describe": describe(fpt.dropna().assign(Close=lambda x: x.Close / 1000, SMA20=lambda x: x.SMA20 / 1000,
                                                 Volume=lambda x: x.Volume / 1e6),
                             ["Close", "Volume", "SMA20", "RSI14", "Vol30"],
                             ["Close (nghìn đ)", "Volume (triệu cp)", "SMA20 (nghìn đ)", "RSI-14", "Biến động 30 phiên (\\%)"]),
        "sample": [[str(i.date()), x.Open, x.High, x.Low, x.Close, x.Volume, x.SMA20, x.RSI14] for i, x in sample.iterrows()]})

    d = dq.churn_data()
    X, y = d["X_raw"], d["y"]
    fig, axs = plt.subplots(3, 1, figsize=(W, 6.8), sharex=True)
    for ax, j, lab in [(axs[0], 6, "log(1 + số giây nghe)"), (axs[1], 5, "log(1 + số bài khác nhau)"), (axs[2], 7, "Tỉ lệ ngày có hoạt động")]:
        for k, c, name in [(0, C["green"], "Ở lại"), (1, C["red"], "Rời bỏ")]:
            m = X[y == k, :, j].mean(0)
            sd = X[y == k, :, j].std(0) / np.sqrt((y == k).sum()) * 1.96
            ax.plot(np.arange(1, 32), m, color=c, lw=2, label=name, marker="o", ms=3)
            ax.fill_between(np.arange(1, 32), m - sd, m + sd, color=c, alpha=0.2)
        ax.set_title(lab, fontsize=11)
    axs[2].set_xlabel("Ngày trong tháng 3/2017")
    axs[0].legend()
    fig.tight_layout()
    save(fig, "f4_churn_traj")

    act = X[:, :, 7]
    fig, ax = plt.subplots(figsize=(W, 3.2))
    for k, c, name in [(0, C["green"], "Ở lại"), (1, C["red"], "Rời bỏ")]:
        ax.hist(act[y == k].sum(1), bins=np.arange(0, 33) - 0.5, density=True, alpha=0.5, color=c, label=name)
    ax.set_xlabel("Số ngày có hoạt động trong 31 ngày")
    ax.set_ylabel("Mật độ")
    ax.legend()
    save(fig, "f4_churn_days")

    feats = pd.DataFrame(X.mean(1), columns=dq.CHURN_FEATURES)
    feats["churn"] = y
    corr = feats.corr()
    fig, ax = plt.subplots(figsize=(W * 0.62, 3.0))
    cnt = np.bincount(y)
    bars = ax.bar(["Ở lại (0)", "Rời bỏ (1)"], cnt, color=[C["green"], C["red"]], width=0.55)
    label_bars(ax, bars, cnt, len(y))
    ax.set_ylim(0, cnt.max() * 1.25)
    ax.set_ylabel("Số người dùng")
    save(fig, "f4_churn_rate")
    names = ["<25 %", "25-50 %", "50-75 %", "75-98,5 %", "100 %", "Bài khác nhau", "Số giây", "Hoạt động", "Rời bỏ"]
    heat(corr, names, "f4_churn_corr", (W, 5.2), fs=8.5)
    merge_json(EDA, "churn", {
        "n_users": len(y), "T": X.shape[1], "D": X.shape[2], "churn_rate": y.mean(),
        "n_train": len(d["y_train"]), "n_val": len(d["y_val"]), "n_test": len(d["y_test"]),
        "active_days_churn": act[y == 1].sum(1).mean(), "active_days_loyal": act[y == 0].sum(1).mean(),
        "secs_first5_churn": float(X[y == 1, :5, 6].mean()), "secs_last5_churn": float(X[y == 1, -5:, 6].mean()),
        "secs_first5_loyal": float(X[y == 0, :5, 6].mean()), "secs_last5_loyal": float(X[y == 0, -5:, 6].mean()),
        "corr_active": corr.loc["active", "churn"], "corr_secs": corr.loc["total_secs", "churn"]})


if __name__ == "__main__":
    which = sys.argv[1:] or ["ch2", "ch3", "ch4"]
    if "ch2" in which:
        ch2_diabetes()
        ch2_house()
        ch2_reviews()
    if "ch3" in which:
        ch3()
    if "ch4" in which:
        ch4()
