"""Hình kết quả thực nghiệm của Chương 2, 3, 4, đọc từ outputs/metrics (không huấn luyện lại gì).

    python scripts/make_figures.py [ch2 ch3 ch4]

Ma trận nhầm lẫn được lưu mỗi mô hình một tệp, rộng nửa trang, để báo cáo xếp hai hình một hàng
mà chữ vẫn đủ lớn.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_style import C, W, plt, save, vn  # noqa: E402

from tl.common import METRICS, load_json  # noqa: E402

FW = {"scratch": ("NumPy", C["gray"]), "keras": ("Keras", C["blue"]), "torch": ("PyTorch", C["red"])}
ARCH = {"custom": ("Custom CNN", C["gray"]), "vgg11": ("VGG-11", C["blue"]),
        "mobilenetv2": ("MobileNetV2", C["orange"]), "resnet18": ("ResNet-18", C["green"])}
CELL = {"rnn": ("Simple RNN", C["gray"]), "lstm": ("LSTM", C["blue"]), "gru": ("GRU", C["orange"]),
        "bilstm": ("BiLSTM", C["green"])}
HALF = (W / 2, W / 2 * 0.95)


def cm_fig(cm, labels, title, name, size=HALF, cmap="Blues"):
    cm = np.asarray(cm, float)
    show = cm / cm.sum(1, keepdims=True)
    n = len(labels)
    fig, ax = plt.subplots(figsize=size)
    ax.imshow(show, cmap=cmap, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(n), labels, rotation=40 if n > 3 else 0, ha="right" if n > 3 else "center")
    ax.set_yticks(range(n), labels)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_title(title)
    fs = 8.5 if n > 5 else 11
    for i in range(n):
        for j in range(n):
            txt = f"{vn(100 * show[i, j], 0)}" if n > 3 else f"{vn(int(cm[i, j]))}\n{vn(100 * show[i, j], 1)} %"
            ax.text(j, i, txt, ha="center", va="center", fontsize=fs, color="white" if show[i, j] > 0.6 else "black")
    save(fig, name)


def bars(ax, groups, series, colors, ylim=None):
    n = len(series)
    w = 0.84 / n
    x = np.arange(len(groups))
    for k, (name, vals) in enumerate(series.items()):
        ax.bar(x + (k - (n - 1) / 2) * w, vals, w, label=name, color=colors[k])
    ax.set_xticks(x, groups)
    if ylim:
        ax.set_ylim(*ylim)


# ================================================================= Chương 2
def ch2():
    sk = load_json("ch2_sklearn")
    mlp = {fw: load_json(f"ch2_mlp_{fw}") for fw in FW}
    fig, axs = plt.subplots(3, 1, figsize=(W, 7.6))
    for ax, ds, t in zip(axs, ["diabetes", "house", "reviews"],
                         ["Tiểu đường: entropy chéo nhị phân", "Giá nhà: MSE trên nhãn chuẩn hóa", "Cảm xúc: entropy chéo 3 lớp"]):
        for fw, (lab, c) in FW.items():
            h = mlp[fw][ds]["hist"]
            ep = np.arange(1, len(h["loss"]) + 1)
            ax.plot(ep, h["loss"], color=c, ls="--", lw=1.5)
            ax.plot(ep, h["val_loss"], color=c, lw=2.2, label=lab, marker="o", ms=3)
        ax.set_title(t, fontsize=11.5)
        ax.set_ylabel("Mất mát")
    axs[-1].set_xlabel("Epoch")
    axs[0].legend(title="liền: kiểm định · đứt: huấn luyện", title_fontsize=9.5, ncol=3)
    fig.tight_layout()
    save(fig, "f2_loss_mlp")

    p = np.load(METRICS / "ch2_pred_house_sklearn.npz")
    y = p["y_test"]
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.4), sharey=True)
    for ax, (name, yp, c) in zip(axs, [("Random Forest", p["rf"], C["orange"]),
                                       ("MLP PyTorch", np.load(METRICS / "ch2_pred_house_mlp_torch.npy"), C["red"])]):
        ax.scatter(y, yp, s=4, alpha=0.25, color=c)
        ax.plot([0, 12], [0, 12], color="k", lw=1.2, ls="--")
        r = sk["house/rf"] if name == "Random Forest" else mlp["torch"]["house"]
        ax.set_title(f"{name}\nR² = {vn(r['r2'], 3)}, MAE = {vn(r['mae'], 2)} tỷ", fontsize=11)
        ax.set_xlabel("Giá thực tế (tỷ đồng)")
        ax.set_xlim(0, 12)
        ax.set_ylim(0, 12)
    axs[0].set_ylabel("Giá dự báo (tỷ đồng)")
    fig.tight_layout()
    save(fig, "f2_house_pred")

    cols6 = [C["sky"], C["purple"], C["orange"], C["gray"], C["blue"], C["red"]]
    ser = {"Logistic": sk["diabetes/logreg"], "SVM RBF": sk["diabetes/svm"], "Random Forest": sk["diabetes/rf"],
           "MLP NumPy": mlp["scratch"]["diabetes"], "MLP Keras": mlp["keras"]["diabetes"], "MLP PyTorch": mlp["torch"]["diabetes"]}
    keys = ["acc", "precision", "recall", "f1", "auc"]
    fig, ax = plt.subplots(figsize=(W, 3.8))
    bars(ax, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"], {k: [v[m] for m in keys] for k, v in ser.items()}, cols6, (0, 1.0))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9.5)
    save(fig, "f2_bar_diab")

    ser = {"OLS": sk["house/ols"], "Ridge": sk["house/ridge"], "SVR": sk["house/svr"], "Random Forest": sk["house/rf"],
           "MLP NumPy": mlp["scratch"]["house"], "MLP Keras": mlp["keras"]["house"], "MLP PyTorch": mlp["torch"]["house"]}
    cols7 = [C["sky"], C["yellow"], C["purple"], C["orange"], C["gray"], C["blue"], C["red"]]
    fig, axs = plt.subplots(1, 2, figsize=(W, 3.6), gridspec_kw={"width_ratios": [2, 1]})
    bars(axs[0], ["MAE (tỷ)", "RMSE (tỷ)"], {k: [v["mae"], v["rmse"]] for k, v in ser.items()}, cols7, (0, 1.6))
    bars(axs[1], ["R²"], {k: [v["r2"]] for k, v in ser.items()}, cols7, (0, 0.7))
    axs[0].legend(ncol=4, loc="lower center", bbox_to_anchor=(0.75, 1.0), fontsize=9)
    fig.tight_layout()
    save(fig, "f2_bar_house")

    ser = {"Logistic": sk["reviews/logreg"], "Linear SVM": sk["reviews/linsvm"], "Random Forest": sk["reviews/rf"],
           "MLP NumPy": mlp["scratch"]["reviews"], "MLP Keras": mlp["keras"]["reviews"], "MLP PyTorch": mlp["torch"]["reviews"]}
    fig, ax = plt.subplots(figsize=(W, 3.8))
    bars(ax, ["Accuracy", "Precision", "Recall", "F1 (macro)"], {k: [v[m] for m in keys[:4]] for k, v in ser.items()}, cols6, (0, 1.0))
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=9.5)
    save(fig, "f2_bar_rev")

    for key, t, cmap in [("logreg", "Logistic Regression", "Purples"), ("rf", "Random Forest", "Oranges")]:
        cm_fig(sk[f"diabetes/{key}"]["confusion"], ["Không mắc", "Mắc"], t, f"f2_cm_diab_{key}", cmap=cmap)
    cm_fig(mlp["torch"]["diabetes"]["confusion"], ["Không mắc", "Mắc"], "MLP PyTorch", "f2_cm_diab_mlp", cmap="Reds")
    cm_fig(sk["diabetes/svm"]["confusion"], ["Không mắc", "Mắc"], "SVM nhân RBF", "f2_cm_diab_svm", cmap="Greens")
    lab = ["Tiêu cực", "Trung tính", "Tích cực"]
    cm_fig(sk["reviews/logreg"]["confusion"], lab, "Logistic Regression", "f2_cm_rev_logreg", cmap="Purples")
    cm_fig(mlp["torch"]["reviews"]["confusion"], lab, "MLP PyTorch", "f2_cm_rev_mlp", cmap="Reds")


# ================================================================= Chương 3
def ch3():
    eda = load_json("eda")
    bench = load_json("bench")
    for ds in ("svhn", "eurosat"):
        r = load_json(f"ch3_{ds}")
        fig, axs = plt.subplots(2, 1, figsize=(W, 6.0), sharex=True)
        for a, (lab, c) in ARCH.items():
            h = r[a]["hist"]
            ep = np.arange(1, len(h["loss"]) + 1)
            axs[0].plot(ep, h["loss"], color=c, lw=2, label=lab)
            axs[1].plot(ep, np.array(h["val_acc"]) * 100, color=c, lw=2, label=lab, marker="o", ms=3)
        h = r["custom_keras"]["hist"]
        ep = np.arange(1, len(h["loss"]) + 1)
        axs[0].plot(ep, h["loss"], color=C["purple"], lw=1.6, ls="--", label="Custom CNN (Keras)")
        axs[1].plot(ep, np.array(h["val_acc"]) * 100, color=C["purple"], lw=1.6, ls="--", label="Custom CNN (Keras)")
        axs[0].set_ylabel("Mất mát huấn luyện")
        axs[1].set_ylabel("Độ chính xác kiểm thử (%)")
        axs[1].set_ylim(85 if ds == "svhn" else 75, 100)
        axs[1].set_xlabel("Epoch")
        axs[0].legend(ncol=2, fontsize=9.5)
        fig.tight_layout()
        save(fig, f"f3_curves_{ds}")
        size = (W * 0.8, W * 0.74) if ds == "svhn" else (W * 0.92, W * 0.84)
        cm_fig(r["resnet18"]["confusion"], eda[ds]["classes"], f"ResNet-18 trên {'SVHN' if ds == 'svhn' else 'EuroSAT'} (% theo hàng)",
               f"f3_cm_{ds}", size=size, cmap="Greens")
    fig, axs = plt.subplots(2, 2, figsize=(W, 6.0))
    x = np.arange(4)
    names = [v[0].replace(" CNN", "") for v in ARCH.values()]
    cols = [v[1] for v in ARCH.values()]
    for ax, key, t, log in [(axs[0, 0], "gpu_ms", "Độ trễ GPU, lô 1 (ms)", False), (axs[0, 1], "cpu_ms", "Độ trễ CPU 4 luồng, lô 1 (ms)", False),
                            (axs[1, 0], "fps", "Thông lượng GPU, lô 256 (ảnh/s)", True)]:
        for ds, off, hatch in (("svhn", -0.2, None), ("eurosat", 0.2, "//")):
            ax.bar(x + off, [bench[f"ch3/{ds}/{a}"][key] for a in ARCH], 0.4, color=cols, hatch=hatch, edgecolor="white")
        ax.set_xticks(x, names, rotation=20)
        ax.set_title(t, fontsize=11)
        if log:
            ax.set_yscale("log")
    ax = axs[1, 1]
    for ds, mk in (("svhn", "o"), ("eurosat", "s")):
        r = load_json(f"ch3_{ds}")
        for a, (lab, c) in ARCH.items():
            ax.scatter(r[a]["params"] / 1e6, r[a]["acc"] * 100, s=40 + 60 * r[a]["gflops"], color=c, marker=mk, alpha=0.85,
                       edgecolor="k")
    ax.set_xlabel("Tham số (triệu)")
    ax.set_ylabel("Độ chính xác (%)")
    ax.set_title("Độ chính xác – kích thước\n(tròn: SVHN, vuông: EuroSAT)", fontsize=11)
    fig.text(0.5, 0.005, "cột đặc: SVHN 32×32 · cột gạch chéo: EuroSAT 64×64", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    save(fig, "f3_latency")


# ================================================================= Chương 4
def ch4():
    t, k = load_json("ch4_torch"), load_json("ch4_keras")
    bench = load_json("bench")
    fig, axs = plt.subplots(2, 1, figsize=(W, 6.0))
    for c, (lab, col) in CELL.items():
        for ax, task in zip(axs, ("stock", "churn")):
            h = t[f"{task}/{c}"]["hist"]
            ep = np.arange(1, len(h["loss"]) + 1)
            ax.plot(ep, h["loss"], color=col, lw=2, label=lab)
            ax.plot(ep, h["val_loss"], color=col, lw=2, ls="--")
    axs[0].set_title("FPT: MSE trên lợi suất (%²)", fontsize=11.5)
    axs[1].set_title("KKBox: entropy chéo nhị phân", fontsize=11.5)
    axs[1].set_xlabel("Epoch")
    axs[0].legend(title="liền: huấn luyện · đứt: kiểm định", title_fontsize=9.5, ncol=2)
    fig.tight_layout()
    save(fig, "f4_loss")

    p = np.load(METRICS / "ch4_stock_pred_torch.npz")
    d = np.array(p["dates"], dtype="datetime64[D]")
    fig, axs = plt.subplots(2, 1, figsize=(W, 6.6))
    for ax, sl, title in zip(axs, (slice(None), slice(-60, None)), ("Toàn bộ giai đoạn kiểm thử", "Phóng to 60 phiên cuối")):
        ax.plot(d[sl], p["y_true"][sl], color="k", lw=1.8, label="Thực tế")
        ax.plot(d[sl], p["naive"][sl], color=C["gray"], lw=1.4, ls=":", label="Giữ nguyên giá")
        for c in ("rnn", "lstm", "bilstm"):
            ax.plot(d[sl], p[c][sl], color=CELL[c][1], lw=1.3, alpha=0.9, label=CELL[c][0])
        ax.set_ylabel("Nghìn đồng")
        ax.set_title(title, fontsize=11.5)
        ax.tick_params(axis="x", rotation=20)
    axs[0].legend(ncol=3, fontsize=9.5)
    fig.tight_layout()
    save(fig, "f4_pred_stock")

    for c, cmap in (("lstm", "Blues"), ("bilstm", "Greens")):
        cm_fig(t[f"churn/{c}"]["confusion"], ["Ở lại", "Rời bỏ"], f"{CELL[c][0]} (ngưỡng {vn(t[f'churn/{c}']['threshold'], 2)})",
               f"f4_cm_churn_{c}", cmap=cmap)

    fig, axs = plt.subplots(2, 2, figsize=(W, 6.0))
    x = np.arange(4)
    panels = [("RMSE FPT (nghìn đồng)", lambda r, c: r[f"stock/{c}"]["rmse"]),
              ("F1 lớp rời bỏ (KKBox)", lambda r, c: r[f"churn/{c}"]["f1"]),
              ("Số tham số (mô hình FPT)", lambda r, c: r[f"stock/{c}"]["params"]),
              ("Độ trễ CPU, lô 1 (ms)", None)]
    for ax, (title, get) in zip(axs.ravel(), panels):
        if get is None:
            vt = [bench[f"ch4/stock/{c}_torch"]["cpu_ms"] for c in CELL]
            vk = [bench[f"ch4/stock/{c}_keras"]["cpu_ms"] for c in CELL]
        else:
            vt = [get(t, c) for c in CELL]
            vk = [get(k, c) for c in CELL]
        ax.bar(x - 0.2, vt, 0.4, color="#EE4C2C", label="PyTorch")
        ax.bar(x + 0.2, vk, 0.4, color=C["blue"], label="Keras")
        ax.set_xticks(x, ["RNN", "LSTM", "GRU", "BiLSTM"])
        ax.set_title(title, fontsize=11)
        if title.startswith("RMSE"):
            ax.axhline(t["stock/naive"]["rmse"], color="k", ls=":", lw=1.5, label="Giữ nguyên giá")
            ax.set_ylim(1.5, max(vt + vk) * 1.02)
    axs[0, 0].legend(fontsize=9)
    fig.tight_layout()
    save(fig, "f4_fw")


if __name__ == "__main__":
    for w in sys.argv[1:] or ["ch2", "ch3", "ch4"]:
        globals()[w]()
