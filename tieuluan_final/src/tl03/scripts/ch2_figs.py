"""Hình dữ liệu và kết quả của ba thực nghiệm Chương 2 (EDA, so sánh mô hình, ROC/PR, phần dư, từ đặc trưng...)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import limit_threads, load_json  # noqa: E402

limit_threads(4)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve  # noqa: E402

from tl3 import ch2_data as cd  # noqa: E402
from tl3.common import OUT  # noqa: E402
from tl3.plot_style import C, SEQ, W, W2, plt, vn, vn_axis  # noqa: E402
from tl3.plot_style import save as _save  # noqa: E402

SHORT = {"baseline": "Baseline", "logreg": "LogReg", "knn": "KNN", "nb": "NB", "tree": "Cây QĐ", "rf": "RF",
         "xgb": "XGBoost", "lgbm": "LightGBM", "svm": "SVM", "mlp": "MLP", "linear": "Linear", "ridge": "Ridge",
         "lasso": "Lasso", "poly": "Poly-2", "svr": "SVR", "cnb": "ComplNB", "softmax": "Softmax"}


def save(fig, name, **kw):
    """Co lề giữa các ô trước khi ghi: nhiều hình nhiều ô ở đây, không co thì tiêu đề đè lên nhãn trục."""
    fig.tight_layout()
    return _save(fig, name, **kw)


def heat(ax, M, xt, yt, fmt="{:.2f}", cmap="Blues", vmin=None, vmax=None, fs=9.5):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(xt)), xt, rotation=40, ha="right")
    ax.set_yticks(range(len(yt)), yt)
    ax.grid(False)
    hi = np.nanmax(M) if vmax is None else vmax
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, fmt.format(M[i, j]).replace(".", ","), ha="center", va="center", fontsize=fs,
                    color="white" if abs(M[i, j]) > 0.6 * abs(hi) else "black")
    return im


# =============================================================== TN1 Diabetes
def diabetes():
    d = load_json("ch2_diabetes")
    raw = cd.load_diabetes_raw()
    df = cd.clean_diabetes(raw)
    pr = np.load(OUT / "ch2_diabetes_preds.npz")

    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.0))
    for ax, data, t in [(a, raw, "Trước làm sạch"), (b, df, "Sau làm sạch")]:
        cnt = data.diabetes.value_counts().sort_index()
        bars = ax.bar(["0: không", "1: có"], cnt.values, color=[C["blue"], C["red"]])
        for bar, v in zip(bars, cnt.values):
            ax.text(bar.get_x() + bar.get_width() / 2, v, f"{vn(v)}\n({100 * v / cnt.sum():.1f}%)".replace(".", ",", 1),
                    ha="center", va="bottom", fontsize=10.5)
        ax.set_title(t)
        ax.set_ylim(0, cnt.max() * 1.25)
        vn_axis(ax)
    save(fig, "ch2_diab_target")

    cols = cd.DIAB_NUM
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.6))
    for ax, c in zip(axes.ravel(), ["age", "bmi", "HbA1c_level", "blood_glucose_level"]):
        bins = np.histogram_bin_edges(df[c], bins=40)
        for lab, col, nm in [(0, C["blue"], "không"), (1, C["red"], "có tiểu đường")]:
            ax.hist(df[c][df.diabetes == lab], bins=bins, density=True, alpha=0.55, color=col, label=nm)
        ax.set_title(c, fontsize=11)
        ax.set_yticks([])
    axes[0, 0].legend(fontsize=9.5)
    save(fig, "ch2_diab_hist")

    fig, axes = plt.subplots(1, 4, figsize=(W, 2.9))
    for ax, c in zip(axes, ["age", "bmi", "HbA1c_level", "blood_glucose_level"]):
        parts = ax.violinplot([df[c][df.diabetes == 0], df[c][df.diabetes == 1]], showmedians=True)
        for body, col in zip(parts["bodies"], [C["blue"], C["red"]]):
            body.set_facecolor(col)
            body.set_alpha(0.6)
        ax.set_xticks([1, 2], ["0", "1"])
        ax.set_title(c.replace("_level", ""), fontsize=11)
    save(fig, "ch2_diab_violin")

    corr = df[cols + ["diabetes"]].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(W, 4.6))
    im = heat(ax, corr, [c.replace("_level", "") for c in cols + ["diabetes"]],
              [c.replace("_level", "") for c in cols + ["diabetes"]], cmap="RdBu_r", vmin=-1, vmax=1, fs=10.5)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "ch2_diab_corr")

    fig, axes = plt.subplots(2, 2, figsize=(W, 4.6))
    groups = [("smoking_history", "Tiền sử hút thuốc"), ("gender", "Giới tính"),
              ("hypertension", "Tăng huyết áp"), ("heart_disease", "Bệnh tim")]
    for ax, (c, t), col in zip(axes.ravel(), groups, [C["purple"], C["teal"], C["orange"], C["red"]]):
        r = df.groupby(c).diabetes.mean().sort_values()
        ax.barh([str(i) for i in r.index], 100 * r.values, color=col)
        for i, v in enumerate(100 * r.values):
            ax.text(v, i, f" {v:.1f}%".replace(".", ","), va="center", fontsize=10)
        ax.set_title(t, fontsize=11.5)
        ax.set_xlim(0, 100 * r.max() * 1.3)
        ax.set_xlabel("Tỷ lệ tiểu đường (%)")
    save(fig, "ch2_diab_rates")

    s = df.sample(8000, random_state=1)
    fig, ax = plt.subplots(figsize=(W, 3.6))
    for lab, col, nm in [(0, C["blue"], "không"), (1, C["red"], "có tiểu đường")]:
        m = s.diabetes == lab
        ax.scatter(s.HbA1c_level[m] + np.random.default_rng(0).normal(0, 0.05, m.sum()),
                   s.blood_glucose_level[m], s=7, alpha=0.45, color=col, label=nm)
    ax.axvline(6.5, ls="--", color="k", lw=1.2)
    ax.axhline(200, ls="--", color="k", lw=1.2)
    lab = dict(fontsize=10.5, bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.5))  # nền trắng: chữ nằm trên đám điểm
    ax.text(6.55, 285, "HbA1c ≥ 6,5%", **lab)
    ax.text(3.6, 205, "glucose ≥ 200 mg/dL", **lab)
    ax.set_xlabel("HbA1c (%)")
    ax.set_ylabel("Đường huyết (mg/dL)")
    ax.legend(markerscale=3, loc="upper left")
    save(fig, "ch2_diab_scatter")

    # tách theo nhãn để thấy ngoại lai của HbA1c/glucose nằm ở lớp có bệnh (lời văn và chú thích dựa vào điều này)
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.0))
    fig.subplots_adjust(wspace=0.35)
    for ax, c in zip(axes, ["bmi", "HbA1c_level", "blood_glucose_level"]):
        o = d["audit_raw"]["outliers"][c]
        bp = ax.boxplot([raw.loc[raw.diabetes == k, c] for k in (0, 1)], widths=0.55, patch_artist=True,
                        flierprops=dict(marker=".", markersize=2.5, alpha=0.35),
                        medianprops=dict(color="black", lw=1.4))
        for box, col in zip(bp["boxes"], (C["blue"], C["red"])):
            box.set(facecolor=col, alpha=0.55)
        ax.set_xticks([1, 2], ["không", "có bệnh"])
        ax.set_title(f"{c.replace('_level', '')}\n{vn(o['n'])} ngoại lai", fontsize=10.5)
    save(fig, "ch2_diab_outliers")

    names = [n for n in d["val"] if n != "baseline"]
    fig, ax = plt.subplots(figsize=(W, 3.4))
    x = np.arange(len(names))
    for i, (k, lab) in enumerate([("prec", "Precision"), ("rec", "Recall"), ("f1", "F1"), ("auc", "ROC-AUC")]):
        ax.bar(x + (i - 1.5) * 0.2, [d["val"][n][k] for n in names], 0.2, label=lab)
    ax.set_xticks(x, [SHORT[n] for n in names], rotation=25)
    ax.set_ylim(0.3, 1.05)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    vn_axis(ax, d=1)
    save(fig, "ch2_diab_compare")

    fig, axes = plt.subplots(3, 3, figsize=(W, 7.6))
    for k, (ax, n) in enumerate(zip(axes.ravel(), names)):
        t = d["test"][n]
        M = np.array([[t["tn"], t["fp"]], [t["fn"], t["tp"]]])
        heat(ax, M / M.sum(1, keepdims=True), ["0", "1"], ["0", "1"], fs=11)
        ax.set_xticks([0, 1], ["0", "1"], rotation=0, ha="center")
        for i in range(2):
            for j in range(2):
                ax.text(j, i + 0.3, f"({vn(M[i, j])})", ha="center", fontsize=8.5,
                        color="white" if M[i, j] / M[i].sum() > 0.6 else "black")
        ax.set_title(SHORT[n], fontsize=11.5)
        if k % 3 == 0:
            ax.set_ylabel("thật")
        if k >= 6:
            ax.set_xlabel("dự đoán")
    save(fig, "ch2_diab_confusion")

    # cao hơn vì chú giải nằm dưới trục và tight_layout của save() co trục để chứa nó
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 5.0))
    for n, col in zip(names, SEQ):
        fpr, tpr, _ = roc_curve(pr["y_te"], pr[f"{n}_te"])
        p, r, _ = precision_recall_curve(pr["y_te"], pr[f"{n}_te"])
        a.plot(fpr, tpr, color=col, lw=1.6, label=f"{SHORT[n]} {d['test'][n]['auc']:.3f}".replace(".", ","))
        b.plot(r, p, color=col, lw=1.6, label=f"{SHORT[n]} {d['test'][n]['ap']:.3f}".replace(".", ","))
    a.plot([0, 1], [0, 1], "--", color=C["gray"], lw=1)
    b.axhline(pr["y_te"].mean(), ls="--", color=C["gray"], lw=1)
    a.set(title="ROC (Test)", xlabel="FPR", ylabel="TPR")
    b.set(title="Precision–Recall (Test)", xlabel="Recall", ylabel="Precision")
    # chú giải đặt dưới trục: đặt trong khung sẽ che các đường cong (nhất là đường NB ở giữa khung PR)
    for ax, t in ((a, "AUC"), (b, "AP")):
        ax.legend(fontsize=9.5, title=t, title_fontsize=10, ncol=2, loc="upper center",
                  bbox_to_anchor=(0.5, -0.2), columnspacing=1.0, handlelength=1.4)
    save(fig, "ch2_diab_roc_pr")

    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.0))
    for n, col in zip(["logreg", "rf", "xgb", "mlp", "nb"], SEQ):
        tc = d["thr_curve"][n]
        a.plot(tc["grid"], tc["f1"], color=col, label=SHORT[n])
        a.plot(d["val"][n]["thr"], max(tc["f1"]), "o", color=col)
    a.set(title="F1 (Validation) theo ngưỡng", xlabel="Ngưỡng")
    a.legend(fontsize=9.5)
    best = d["best"]["name"]
    p, r, t = precision_recall_curve(pr["y_va"], pr[f"{best}_va"])
    b.plot(t, p[:-1], label="Precision")
    b.plot(t, r[:-1], label="Recall")
    b.axvline(d["best"]["thr"], ls="--", color="k", lw=1.2)
    b.set(title=f"{SHORT[best]}: P/R theo ngưỡng", xlabel="Ngưỡng")
    b.legend(loc="lower left", bbox_to_anchor=(0.52, 0.1))  # bên phải đường ngưỡng, dưới đường Recall
    save(fig, "ch2_diab_threshold")

    dc, kc = d["curve_depth"], d["curve_knn"]
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.plot(dc["depth"], dc["train_auc"], "o-", label="Train")
    ax.plot(dc["depth"], dc["val_auc"], "s-", label="Validation")
    ax.set(xlabel="max_depth", ylabel="ROC-AUC", title="Cây quyết định")
    ax.legend()
    save(fig, "ch2_diab_depth")
    fig, ax = plt.subplots(figsize=(W2, 3.0))
    ax.plot(kc["k"], kc["val_auc"], "o-", color=C["green"], label="AUC")
    ax.plot(kc["k"], kc["val_f1"], "s-", color=C["purple"], label="F1")
    ax.set_xscale("log")
    ax.set(xlabel="k (thang log)", title="KNN (Validation)")
    ax.legend()
    save(fig, "ch2_diab_knnk")

    lc = d["learning_curve"]
    fig, ax = plt.subplots(figsize=(W, 2.9))
    ax.plot(lc["sizes"], lc["train"], "o-", label="Train (CV)")
    ax.plot(lc["sizes"], lc["val"], "s-", label="Validation (CV)")
    ax.fill_between(lc["sizes"], np.array(lc["val"]) - lc["val_std"], np.array(lc["val"]) + lc["val_std"], alpha=0.2,
                    color=C["orange"])
    ax.set(xlabel="Số mẫu huấn luyện", ylabel="F1", title=f"Đường học của {SHORT[lc['model']]}")
    vn_axis(ax, "x")
    ax.legend()
    save(fig, "ch2_diab_learning")

    imp = d["importance"]
    perm = pd.Series({k: v["mean"] for k, v in imp["perm"].items()}).sort_values()
    gain = pd.Series(imp["xgb"]).sort_values().tail(10)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.3))
    a.barh(perm.index.str.replace("_level", ""), perm.values, color=C["teal"],
           xerr=[imp["perm"][k]["std"] for k in perm.index])
    a.set_title("Hoán vị (giảm ROC-AUC)", fontsize=11.5)
    b.barh(gain.index.str.replace("_level", ""), gain.values, color=C["orange"])
    b.set_title("XGBoost gain (top 10)", fontsize=11.5)
    save(fig, "ch2_diab_importance")

    coef = pd.Series(imp["logreg_coef"]).sort_values()
    fig, ax = plt.subplots(figsize=(W, 3.6))
    ax.barh(coef.index.str.replace("_level", ""), coef.values, color=[C["red"] if v > 0 else C["blue"] for v in coef])
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Hệ số w (đặc trưng đã chuẩn hóa; > 0 tăng log-odds)")
    save(fig, "ch2_diab_coef")

    h = pd.DataFrame(d["models"]["mlp"]["history"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.8))
    a.plot(h.epoch, h.train_loss, label="Train")
    a.plot(h.epoch, h.val_loss, label="Validation")
    a.set(title="MLP — BCE theo epoch", xlabel="epoch")
    a.legend()
    for n, col in zip(["logreg", "nb", "xgb", "mlp"], SEQ):
        fp, mp = calibration_curve(pr["y_te"], pr[f"{n}_te"], n_bins=10, strategy="quantile")
        b.plot(mp, fp, "o-", color=col, ms=3.5, lw=1.5, label=SHORT[n])
    b.plot([0, 1], [0, 1], "--", color=C["gray"])
    b.set(title="Hiệu chỉnh xác suất (Test)", xlabel="xác suất dự đoán", ylabel="tỷ lệ thực")
    b.legend(fontsize=9)
    save(fig, "ch2_diab_mlp_calib")

    cv = d["cv"]
    fig, ax = plt.subplots(figsize=(W, 2.7))
    ks = list(cv)
    for i, (m, lab) in enumerate([("test_roc_auc", "ROC-AUC"), ("test_f1", "F1")]):
        ax.errorbar(np.arange(len(ks)) + (i - 0.5) * 0.15, [cv[k][m]["mean"] for k in ks],
                    yerr=[2 * cv[k][m]["std"] for k in ks], fmt="o", capsize=5, ms=7, label=lab)
    ax.set_xticks(range(len(ks)), [SHORT[k] for k in ks])
    ax.set_title("Kiểm định chéo 5-fold trên Train (trung bình ± 2 độ lệch chuẩn)", fontsize=11.5)
    ax.legend()
    save(fig, "ch2_diab_cv")


def scratch():
    s = load_json("ch2_scratch")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.8))
    a.plot(np.arange(len(s["linreg"]["gd_mse_history"])) * 10, s["linreg"]["gd_mse_history"], color=C["green"])
    a.set(title="Linear GD: MSE (y chuẩn hóa)", xlabel="vòng lặp")
    a.set_yscale("log")
    b.plot(s["logreg"]["loss_history"], color=C["red"])
    b.set(title="Logistic GD: BCE", xlabel="epoch")
    save(fig, "ch2_scratch_loss")


# =============================================================== TN2 House
def house():
    h = load_json("ch2_house")
    df = cd.load_house_raw()
    pr = np.load(OUT / "ch2_house_preds.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9))
    a.hist(df.Price / 1e6, bins=40, color=C["blue"])
    a.axvline(df.Price.median() / 1e6, color=C["red"], ls="--", label="trung vị")
    a.set(title="Giá (triệu)", xlabel="Price / 10⁶")
    a.legend(loc="upper left", fontsize=9.5)
    a.text(0.98, 0.95, f"skew = {h['survey']['skew_price']:.2f}".replace(".", ","), transform=a.transAxes,
           ha="right", va="top", fontsize=10.5)
    pb = pd.Series(h["survey"]["price_by_bedrooms"])
    b.bar(pb.index.astype(str), pb.values / 1e6, color=C["purple"])
    b.set(title="Giá trung bình theo số phòng ngủ", xlabel="Bedrooms", ylim=(0.8, pb.max() / 1e6 * 1.05))
    save(fig, "ch2_house_price")

    fig, axes = plt.subplots(2, 4, figsize=(W, 4.3), sharey=True)
    for ax, c, col in zip(axes.ravel(), cd.HOUSE_NUM + ["Area"], SEQ):
        if c == "Area" and ax is axes.ravel()[-1]:
            ax.hexbin(df.Area, df.Price / 1e6, gridsize=25, cmap="Blues")
            ax.set_title("Area (mật độ)", fontsize=10.5)
            continue
        if df[c].nunique() <= 10:
            g = df.groupby(c).Price.median() / 1e6
            ax.plot(g.index, g.values, "o-", color=col, ms=6)
            ax.scatter(df[c] + np.random.default_rng(0).normal(0, 0.08, len(df)), df.Price / 1e6, s=3, alpha=0.15, color=col)
        else:
            ax.scatter(df[c], df.Price / 1e6, s=4, alpha=0.35, color=col)
        r = h["survey"]["corr_target"][c]
        ax.set_title(f"{c}\nr = {r:.2f}".replace(".", ","), fontsize=10)
    axes[0, 0].set_ylabel("Price (triệu)")
    axes[1, 0].set_ylabel("Price (triệu)")
    save(fig, "ch2_house_numeric")

    fig, axes = plt.subplots(2, 2, figsize=(W, 5.0))
    fig.subplots_adjust(hspace=0.75, wspace=0.25)
    for ax, c in zip(axes.ravel(), ["City", "Furnishing", "Preferred Tenant", "Water Supply"]):
        cats = df.groupby(c).Price.median().sort_values().index
        ax.boxplot([df.Price[df[c] == k] / 1e6 for k in cats], patch_artist=True, widths=0.6,
                   boxprops=dict(facecolor=C["sky"], alpha=0.5), flierprops=dict(markersize=2),
                   medianprops=dict(color=C["red"], lw=1.6))
        # neo nhãn nghiêng ở mép phải để bảy tên thành phố không chồng lên nhau
        ax.set_xticks(range(1, len(cats) + 1), cats, rotation=35, ha="right", rotation_mode="anchor",
                      fontsize=10)
        ax.set_title(c, fontsize=11.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Giá (triệu)")
    save(fig, "ch2_house_cats")

    num = df[cd.HOUSE_NUM + ["Price"]].corr().to_numpy()
    fig, ax = plt.subplots(figsize=(W, 4.4))
    im = heat(ax, num, cd.HOUSE_NUM + ["Price"], cd.HOUSE_NUM + ["Price"], cmap="RdBu_r", vmin=-1, vmax=1, fs=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    save(fig, "ch2_house_corr")

    names = [n for n in h["val"] if n != "baseline"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.3))
    order = sorted(names, key=lambda n: h["val"][n]["rmse"])
    a.barh([SHORT[n] for n in order][::-1], [h["val"][n]["rmse"] / 1e3 for n in order][::-1], color=C["orange"])
    base = h["val"]["baseline"]["rmse"] / 1e3
    a.axvline(base, color=C["red"], ls="--")
    # ghi tên ngay cạnh đường đứt nét thay cho chú giải (chú giải góc dưới đè lên thanh Cây QĐ)
    a.text(base, 4.5, "Baseline ", color=C["red"], ha="right", va="center", rotation=90, fontsize=10)
    a.set(title="RMSE Validation (nghìn)")
    b.barh([SHORT[n] for n in order][::-1], [h["val"][n]["r2"] for n in order][::-1], color=C["teal"])
    b.tick_params(labelleft=False)
    b.set(title="R² Validation")
    save(fig, "ch2_house_compare")

    best = h["best"]["name"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.1))
    for ax, n, col in zip(axes, [best, "rf"], [C["blue"], C["green"]]):
        ax.scatter(pr["y_te"] / 1e6, pr[f"{n}_te"] / 1e6, s=10, alpha=0.6, color=col)
        lim = [pr["y_te"].min() / 1e6, pr["y_te"].max() / 1e6]
        ax.plot(lim, lim, "k--", lw=1)
        ax.set(title=f"{SHORT[n]} (R²={h['test'][n]['r2']:.3f})".replace(".", ","), xlabel="Giá thật (triệu)")
    axes[0].set_ylabel("Giá dự đoán (triệu)")
    save(fig, "ch2_house_pred")

    res = pr["y_te"] - pr[f"{best}_te"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9))
    a.scatter(pr[f"{best}_te"] / 1e6, res / 1e3, s=10, alpha=0.6, color=C["purple"])
    a.axhline(0, color="k", lw=1)
    a.set(title="Phần dư theo giá dự đoán", xlabel="dự đoán (triệu)", ylabel="phần dư (nghìn)")
    b.hist(res / 1e3, bins=30, color=C["purple"], alpha=0.8)
    b.set(title="Phân bố phần dư", xlabel="phần dư (nghìn)")
    save(fig, "ch2_house_resid")

    coef = pd.Series(h["linear_coef"]).sort_values()
    # nhãn trục x ngắn: nhãn dài làm hình rộng hơn trang và bị thu nhỏ, kéo chữ tên đặc trưng xuống dưới 8pt
    fig, ax = plt.subplots(figsize=(W, 6.2))
    ax.barh(coef.index, coef.values / 1e3, color=[C["green"] if v > 0 else C["red"] for v in coef])
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("Hệ số Linear Regression (nghìn đơn vị giá)")
    ax.tick_params(axis="y", labelsize=10.5)
    ax.margins(y=0.01)
    save(fig, "ch2_house_coef")

    rp = h["reg_path"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.0), sharey=True)
    R, L = np.array(rp["ridge"]) / 1e3, np.array(rp["lasso"]) / 1e3
    top = np.argsort(-np.abs(R[0]))[:6]
    for j, col in zip(range(R.shape[1]), np.resize(SEQ, R.shape[1])):
        lw, al = (2.0, 1) if j in top else (0.8, 0.35)
        a.plot(rp["ridge_alpha"], R[:, j], color=col, lw=lw, alpha=al)
        b.plot(rp["lasso_alpha"], L[:, j], color=col, lw=lw, alpha=al, label=rp["features"][j] if j in top else None)
    for ax, t in [(a, "Ridge (L2)"), (b, "Lasso (L1)")]:
        ax.set_xscale("log")
        ax.set(title=t, xlabel="α (thang log)")
    a.set_ylabel("hệ số (nghìn)")
    # chú giải chung dưới hai khung: đặt trong khung Lasso sẽ che các đường đang co về 0
    fig.legend(*b.get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3,
               fontsize=10, handlelength=1.6, columnspacing=1.2)
    save(fig, "ch2_house_regpath")

    cv = h["cv"]
    ks = sorted(cv, key=lambda k: -cv[k]["r2_mean"])
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.errorbar(range(len(ks)), [cv[k]["r2_mean"] for k in ks], yerr=[2 * cv[k]["r2_std"] for k in ks], fmt="o",
                capsize=5, ms=8, color=C["navy"])
    ax.set_xticks(range(len(ks)), [SHORT[k] for k in ks])
    ax.set_ylabel("R² (5-fold)")
    save(fig, "ch2_house_cv")

    lc = h["learning_curve"]
    mh = pd.DataFrame(h["models"]["mlp"]["history"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.8))
    a.plot(lc["sizes"], lc["train"], "o-", label="Train")
    a.plot(lc["sizes"], lc["val"], "s-", label="Validation")
    a.set(title=f"Đường học {SHORT[lc['model']]} (R²)", xlabel="số mẫu")
    a.legend()
    b.plot(mh.epoch, mh.train_loss, label="Train")
    b.plot(mh.epoch, mh.val_loss, label="Validation")
    b.set(title="MLP — MSE (giá chuẩn hóa)", xlabel="epoch")
    b.legend()
    save(fig, "ch2_house_learning")


# =============================================================== TN3 Comments
def comments():
    c = load_json("ch2_comments")
    df = cd.load_reviews()
    pr = np.load(OUT / "ch2_comments_preds.npz")
    s = c["survey"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9))
    rc = pd.Series(s["rating_counts"])
    a.bar(rc.index.astype(str), rc.values, color=[C["red"], C["orange"], C["gray"], C["sky"], C["green"]])
    a.set(title="Số nhận xét theo Rating", xlabel="Rating")
    vn_axis(a)
    rr = pd.Series(s["rec_by_rating"])
    b.bar(rr.index.astype(str), 100 * rr.values, color=C["teal"])
    b.set(title="% giới thiệu theo Rating", xlabel="Rating")
    save(fig, "ch2_cmt_labels")

    fig, ax = plt.subplots(figsize=(W, 2.9))
    for lab, col, nm in [(1, C["green"], "Giới thiệu (1)"), (0, C["red"], "Không giới thiệu (0)")]:
        ax.hist(df.n_words[df["Recommended IND"] == lab], bins=50, alpha=0.6, color=col, density=True, label=nm)
    ax.set(xlabel="Số từ mỗi nhận xét", ylabel="mật độ")
    ax.legend()
    save(fig, "ch2_cmt_length")

    from wordcloud import WordCloud
    from tl3.ch2_text import normalize
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.6))
    for ax, lab, cmap, t in [(axes[0], 1, "Greens", "Giới thiệu"), (axes[1], 0, "Reds", "Không giới thiệu")]:
        text = " ".join(df.text[df["Recommended IND"] == lab].map(normalize))
        wc = WordCloud(width=900, height=600, background_color="white", colormap=cmap, max_words=120,
                       random_state=0).generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(t)
    save(fig, "ch2_cmt_wordcloud", ext="png", dpi=220)

    tt = c["top_terms"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 4.2))
    a.barh([t for t, _ in tt["pos"]][::-1], [v for _, v in tt["pos"]][::-1], color=C["green"])
    a.set_title("Đẩy về 'giới thiệu'", fontsize=11.5)
    b.barh([t for t, _ in tt["neg"]][::-1], [-v for _, v in tt["neg"]][::-1], color=C["red"])
    b.set_title("Đẩy về 'không giới thiệu'", fontsize=11.5)
    for ax in (a, b):
        ax.tick_params(axis="y", labelsize=10)
        ax.set_xlabel("|hệ số|")
    save(fig, "ch2_cmt_terms")

    names = [n for n in c["val"]]
    fig, ax = plt.subplots(figsize=(W, 3.0))
    x = np.arange(len(names))
    for i, (k, lab) in enumerate([("acc", "Accuracy"), ("macro_f1", "Macro-F1"), ("f1_neg", "F1 lớp 0"),
                                  ("auc", "ROC-AUC")]):
        ax.bar(x + (i - 1.5) * 0.2, [c["val"][n].get(k, 0) for n in names], 0.2, label=lab)
    ax.set_xticks(x, [SHORT[n] for n in names])
    ax.set_ylim(0, 1.08)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    save(fig, "ch2_cmt_compare")

    best = c["best"]["name"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.1), gridspec_kw={"width_ratios": [1, 1.4]})
    M = np.array(c["confusion"][best])
    heat(a, M / M.sum(1, keepdims=True), ["dự đoán 0", "dự đoán 1"], ["thật 0", "thật 1"], fs=11)
    a.set_title(SHORT[best] + " (Test)", fontsize=11.5)
    R = np.array(c["rating"]["confusion"])
    heat(b, R / R.sum(1, keepdims=True), ["1", "2", "3", "4", "5"], ["1", "2", "3", "4", "5"], fs=9)
    b.set_title("Softmax: Rating 1–5", fontsize=11.5)
    b.set_xlabel("dự đoán")
    save(fig, "ch2_cmt_confusion")

    g = df.groupby("Department Name").agg(n=("text", "size"), rec=("Recommended IND", "mean")).sort_values("n")
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.barh(g.index, g.n, color=C["sky"])
    for i, (n, r) in enumerate(zip(g.n, g.rec)):
        ax.text(n, i, f" {100 * r:.0f}% giới thiệu", va="center", fontsize=10)
    ax.set_xlim(0, g.n.max() * 1.35)
    vn_axis(ax, "x")
    save(fig, "ch2_cmt_dept")


# =============================================================== Không giám sát
def unsup():
    u = load_json("ch2_unsup")
    z = np.load(OUT / "ch2_unsup.npz")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.8))
    a.plot(u["scan"]["k"], u["scan"]["inertia"], "o-")
    a.axvline(u["k_final"], ls="--", color=C["red"])
    a.set(title="Elbow: inertia theo k", xlabel="k")
    b.plot(u["scan"]["k"], u["scan"]["silhouette"], "s-", color=C["green"])
    b.set(title="Silhouette theo k", xlabel="k")
    save(fig, "ch2_unsup_elbow")
    P, lab, y = z["P"], z["labels"], z["y"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.1), sharey=True)
    cols = [C["blue"], C["orange"], C["green"], C["purple"]]
    a.scatter(P[:, 0], P[:, 1], c=[cols[i] for i in lab], s=4, alpha=0.5)
    a.set(title="PCA tô theo cụm K-means", xlabel=f"PC1 ({100 * u['pca']['explained'][0]:.0f}%)",
          ylabel=f"PC2 ({100 * u['pca']['explained'][1]:.0f}%)")
    b.scatter(P[y == 0, 0], P[y == 0, 1], s=3, alpha=0.3, color=C["gray"], label="không")
    b.scatter(P[y == 1, 0], P[y == 1, 1], s=5, alpha=0.6, color=C["red"], label="tiểu đường")
    b.set(title="PCA tô theo nhãn thật", xlabel="PC1")
    b.legend(markerscale=3)
    save(fig, "ch2_unsup_pca")
    pf = pd.DataFrame(u["profiles"])
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.bar([f"Cụm {i}\n(n={vn(n)})" for i, n in zip(pf.cluster, pf.n)], 100 * pf.diabetes_rate, color=cols)
    ax.set_ylabel("% tiểu đường trong cụm")
    save(fig, "ch2_unsup_profiles")


if __name__ == "__main__":
    for f in (diabetes, scratch, house, comments, unsup):
        f()
