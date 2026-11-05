"""Vẽ mọi hình dữ liệu Chương 3 vào Report/TieuLuan 03/figures/ch3_*.pdf|png.

    python scripts/ch3_figures.py [--part data|results|viz|all] [--exp mnist fashion ...]

data    : lưới ảnh mẫu, phân bố lớp, ảnh trung bình theo lớp, histogram cường độ, ví dụ tăng cường
results : đường cong học, ma trận nhầm lẫn, so sánh chỉ số, F1 theo lớp, ảnh đoán sai, tổng hợp thời gian/tham số
viz     : bộ lọc và bản đồ đặc trưng tầng conv1 (Basic CNN), Grad-CAM (Improved CNN) — chạy PyTorch trên CPU
Hình nào thiếu mô hình/kết quả thì bỏ qua và báo, chạy lại sau khi huấn luyện xong.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch3_common import CFG, MODEL_IDS, MODEL_NAME, MODELS, has_result, load_result, resize_nhwc  # noqa: E402
from tl3.ch3_data import CLASSES, TITLE, load, load_raw  # noqa: E402
from tl3.common import limit_threads  # noqa: E402
from tl3.plot_style import C, SEQ, W, plt, save, vn, vn_axis  # noqa: E402

EXPS = ["mnist", "fashion", "rps", "cifar10"]
MCOL = {"scratch": C["gray"], "torch_basic": C["blue"], "tf_basic": C["orange"],
        "torch_improved": C["green"], "tf_improved": C["red"]}
XNAME = {"mnist": "MNIST", "fashion": "Fashion-MNIST", "rps": "RPS", "cifar10": "CIFAR-10"}
SHORT = {"scratch": "NumPy", "torch_basic": "PT Basic", "tf_basic": "TF Basic",
         "torch_improved": "PT Improved", "tf_improved": "TF Improved"}


def models_of(exp):
    return [m for m in MODEL_IDS if has_result(exp, m)]


def show(ax, img):
    ax.imshow(img[..., 0] if img.shape[-1] == 1 else img, cmap="gray" if img.shape[-1] == 1 else None,
              vmin=0, vmax=1 if img.dtype != np.uint8 else 255)
    ax.set_xticks([]), ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


# ------------------------------------------------------------------ dữ liệu
def fig_samples(exp):
    xtr, ytr, _, _ = load_raw(exp)
    names, k = CLASSES[exp], len(CLASSES[exp])
    per = 8 if k == 10 else 6
    rng = np.random.default_rng(1)
    left = 0.06 if exp == "mnist" else 0.22
    fig, axes = plt.subplots(k, per, figsize=(W, W * (1 - left) / per * k * 1.04 + 0.45))
    for c in range(k):
        idx = rng.choice(np.flatnonzero(ytr == c), per, replace=False)
        for j, i in enumerate(idx):
            show(axes[c, j], xtr[i])
        axes[c, 0].set_ylabel(names[c], rotation=0, ha="right", va="center", fontsize=11)
        axes[c, 0].yaxis.set_visible(True)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=left, right=1, bottom=0, top=1 - 0.45 / fig.get_figheight())
    fig.suptitle(f"{TITLE[exp]}: {per} ảnh ngẫu nhiên mỗi lớp", fontweight="bold", y=1.0)
    save(fig, f"ch3_{exp}_samples", "png")


def fig_dist(exp):
    _, ytr, _, yte = load_raw(exp)
    names, k = CLASSES[exp], len(CLASSES[exp])
    ctr, cte = np.bincount(ytr, minlength=k), np.bincount(yte, minlength=k)
    fig, ax = plt.subplots(figsize=(W, 3.6))
    x = np.arange(k)
    b1 = ax.bar(x - 0.2, ctr, 0.4, label=f"Huấn luyện ({vn(len(ytr))})", color=C["blue"])
    b2 = ax.bar(x + 0.2, cte, 0.4, label=f"Kiểm tra ({vn(len(yte))})", color=C["orange"])
    for bars in (b1, b2):
        for r in bars:
            if k == 10:     # mười lớp: ghi số dọc bên trong cột cho khỏi chồng chú giải
                ax.text(r.get_x() + r.get_width() / 2, r.get_height() / 2, vn(r.get_height()), ha="center",
                        va="center", fontsize=9, rotation=90, color="white", fontweight="bold")
            else:
                ax.text(r.get_x() + r.get_width() / 2, r.get_height(), vn(r.get_height()), ha="center", va="bottom",
                        fontsize=11)
    ax.set_xticks(x, names, rotation=30 if k == 10 and exp != "mnist" else 0, ha="right" if k == 10 and exp != "mnist" else "center")
    ax.set_ylabel("Số ảnh")
    vn_axis(ax)
    ax.set_ylim(0, ctr.max() * 1.3)
    ax.legend(ncol=2, loc="upper center")
    ax.set_title(f"{TITLE[exp]}: phân bố số ảnh theo lớp")
    save(fig, f"ch3_{exp}_dist")


def fig_mean(exp):
    xtr, ytr, _, _ = load_raw(exp)
    names, k = CLASSES[exp], len(CLASSES[exp])
    cols = 5 if k == 10 else 3
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(W, W / cols * rows * 1.12 + 0.5))
    axes = np.atleast_2d(axes)
    for c in range(k):
        m = xtr[ytr == c].astype(np.float32).mean(0) / 255.0
        ax = axes.flat[c]
        if m.shape[-1] == 1:
            ax.imshow(m[..., 0], cmap="magma", vmin=0, vmax=1)
        else:
            ax.imshow((m - m.min()) / (m.max() - m.min()))
        ax.set_title(names[c], fontsize=11)
        ax.axis("off")
    for a in axes.flat[k:]:
        a.axis("off")
    fig.subplots_adjust(top=1 - 0.85 / fig.get_figheight(), hspace=0.3, wspace=0.08)
    fig.suptitle(f"{TITLE[exp]}: ảnh trung bình của từng lớp", fontweight="bold", y=0.99)
    save(fig, f"ch3_{exp}_mean", "png")


def fig_pixels(exp):
    xtr, ytr, _, _ = load_raw(exp)
    sub = xtr[np.random.default_rng(0).choice(len(xtr), min(5000, len(xtr)), replace=False)]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.3))
    if sub.shape[-1] == 1:
        axes[0].hist(sub.ravel(), bins=64, color=C["navy"], log=True)
        axes[0].set_title("Histogram (thang log)")
    else:
        for ch, col, nm in zip(range(3), (C["red"], C["green"], C["blue"]), "RGB"):
            axes[0].hist(sub[..., ch].ravel(), bins=64, histtype="step", lw=2, color=col, label=f"kênh {nm}")
        axes[0].legend()
        axes[0].set_title("Histogram theo kênh")
    axes[0].set_xlabel("Giá trị điểm ảnh (0–255)")
    f = sub.astype(np.float32) / 255.0
    stats = [(f[..., ch].mean(), f[..., ch].std()) for ch in range(f.shape[-1])]
    labels = ["xám"] if len(stats) == 1 else ["R", "G", "B"]
    cols = [C["gray"]] if len(stats) == 1 else [C["red"], C["green"], C["blue"]]
    xs = np.arange(len(stats))
    axes[1].bar(xs - 0.18, [s[0] for s in stats], 0.36, color=cols, label="trung bình $\\mu$")
    axes[1].bar(xs + 0.18, [s[1] for s in stats], 0.36, color=cols, alpha=0.45, hatch="//", label="độ lệch $\\sigma$")
    for x_, (m, s) in zip(xs, stats):
        axes[1].text(x_ - 0.18, m, vn(m, 3), ha="center", va="bottom", fontsize=10)
        axes[1].text(x_ + 0.18, s, vn(s, 3), ha="center", va="bottom", fontsize=10)
    axes[1].set_xticks(xs, labels)
    axes[1].set_ylim(0, 0.75)
    axes[1].legend(loc="upper right", fontsize=9.5)
    axes[1].set_title(r"$\mu$, $\sigma$ trên thang [0, 1]")
    vn_axis(axes[1], d=1)
    fig.subplots_adjust(wspace=0.32)
    fig.suptitle(f"{TITLE[exp]}: phân bố cường độ điểm ảnh", fontweight="bold", y=1.03)
    save(fig, f"ch3_{exp}_pixels")


def fig_aug(exp):
    import torch
    from tl3.ch3_torch import gpu_augment, to_tensor
    (xtr, ytr), _, _ = load(exp)
    cfg = CFG[exp]
    rng = np.random.default_rng(5)
    pick = [rng.choice(np.flatnonzero(ytr == c)) for c in rng.permutation(int(ytr.max()) + 1)[:4]]
    pick += list(rng.choice(len(ytr), 4 - len(pick), replace=False))
    x = resize_nhwc(xtr[pick], cfg["improved_size"])
    torch.manual_seed(3)
    n_aug = 6
    fig, axes = plt.subplots(4, n_aug + 1, figsize=(W, W / (n_aug + 1) * 4 * 1.03 + 0.7))
    for r in range(4):
        show(axes[r, 0], x[r])
        t = to_tensor(np.repeat(x[r:r + 1], n_aug, 0))
        a = gpu_augment(t, cfg).numpy().transpose(0, 2, 3, 1)
        for j in range(n_aug):
            show(axes[r, j + 1], a[j])
    axes[0, 0].set_title("Gốc", fontsize=11, color=C["navy"], fontweight="bold")
    for j in range(n_aug):
        axes[0, j + 1].set_title(f"#{j + 1}", fontsize=11)
    fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1 - 0.7 / fig.get_figheight())
    fig.suptitle(f"{TITLE[exp]}: tăng cường ngẫu nhiên (Improved CNN)", fontweight="bold", y=1.0)
    save(fig, f"ch3_{exp}_aug", "png")


# ------------------------------------------------------------------ kết quả
def fig_curves(exp):
    ms = models_of(exp)
    if not ms:
        return print("   bỏ qua curves", exp)
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.7))
    for m in ms:
        h = load_result(exp, m)["history"]
        ep = np.arange(1, len(h["acc"]) + 1)
        axes[0].plot(ep, h["acc"], "--", color=MCOL[m], lw=1.6, alpha=0.8)
        axes[0].plot(ep, h["val_acc"], "-", color=MCOL[m], marker="o", ms=3.5, label=SHORT[m])
        axes[1].plot(ep, h["val_loss"], "-", color=MCOL[m], marker="o", ms=3.5, label=SHORT[m])
        axes[1].plot(ep, h["loss"], "--", color=MCOL[m], lw=1.6, alpha=0.8)
    axes[0].set_title("Độ chính xác")
    axes[1].set_title("Hàm mất mát (thang log)")
    fig.subplots_adjust(wspace=0.28)
    for a in axes:
        a.set_xlabel("Epoch")
    lo = min(min(load_result(exp, m)["history"]["val_acc"]) for m in ms)
    axes[0].set_ylim(max(0.0, lo - 0.03), 1.0)
    axes[1].set_yscale("log")
    axes[0].legend(fontsize=9.5, loc="lower right")
    fig.suptitle(f"{TITLE[exp]}: đường cong học (nét liền: validation, nét đứt: huấn luyện)", fontweight="bold",
                 fontsize=12, y=1.02)
    save(fig, f"ch3_{exp}_curves")


def _best(exp, among=None):
    ms = [m for m in (among or models_of(exp)) if has_result(exp, m)]
    return max(ms, key=lambda m: load_result(exp, m)["accuracy"]) if ms else None


def _cm_ax(ax, cm, names, title, cmap):
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
    k = len(names)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7.3 if k == 10 else 13,
                    color="white" if cmn[i, j] > 0.55 else "black")
    long = max(len(n) for n in names) > 2
    ax.set_xticks(range(k), names, rotation=60 if long else 0, ha="right" if long else "center", fontsize=9.5)
    ax.set_yticks(range(k), names, fontsize=9.5)
    ax.set_xlabel("Nhãn dự đoán")
    ax.set_title(title, fontsize=11.5)
    ax.grid(False)


def fig_cm(exp):
    basic = _best(exp, ["torch_basic", "tf_basic", "scratch"])
    imp = _best(exp, ["torch_improved", "tf_improved"])
    pair = [m for m in (basic, imp) if m]
    if not pair:
        return print("   bỏ qua cm", exp)
    names = CLASSES[exp]
    fig, axes = plt.subplots(1, len(pair), figsize=(W * (1.0 if len(pair) == 2 else 0.6), 3.9 if len(names) == 10 else 3.0))
    axes = np.atleast_1d(axes)
    for ax, m, cmap in zip(axes, pair, ("Blues", "Greens")):
        r = load_result(exp, m)
        _cm_ax(ax, np.array(r["cm"]), names, f"{MODEL_NAME[m]}\nacc = {vn(100 * r['accuracy'], 2)} %", cmap)
    axes[0].set_ylabel("Nhãn thật")
    fig.tight_layout()
    save(fig, f"ch3_{exp}_cm")


def fig_metrics(exp):
    ms = models_of(exp)
    if not ms:
        return
    keys, labels = ["accuracy", "precision", "recall", "f1"], ["Accuracy", "Precision", "Recall", "F1 (macro)"]
    vals = np.array([[load_result(exp, m)[k] for k in keys] for m in ms])
    fig, ax = plt.subplots(figsize=(W, 3.7))
    wdt = 0.8 / len(ms)
    for i, m in enumerate(ms):
        xs = np.arange(4) + (i - (len(ms) - 1) / 2) * wdt
        bars = ax.bar(xs, 100 * vals[i], wdt, color=MCOL[m], label=MODEL_NAME[m])
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.05, vn(b.get_height(), 2), ha="center",
                    va="bottom", rotation=90, fontsize=8)
    ax.set_xticks(range(4), labels)
    lo = 100 * vals.min()
    ax.set_ylim(max(0, lo - (8 if lo < 90 else 3)), 100.6 if lo > 90 else 102)
    ax.set_ylabel("%")
    ax.legend(ncol=3, fontsize=8.8, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.set_title(f"{TITLE[exp]}: chỉ số trên tập kiểm tra")
    save(fig, f"ch3_{exp}_metrics")


def fig_f1class(exp):
    ms = models_of(exp)
    if not ms:
        return
    names = CLASSES[exp]
    fig, ax = plt.subplots(figsize=(W, 3.8))
    wdt = 0.8 / len(ms)
    for i, m in enumerate(ms):
        f = np.array(load_result(exp, m)["f1_per_class"])
        ax.bar(np.arange(len(names)) + (i - (len(ms) - 1) / 2) * wdt, f, wdt, color=MCOL[m], label=SHORT[m])
    allf = np.concatenate([load_result(exp, m)["f1_per_class"] for m in ms])
    ax.set_ylim(max(0, allf.min() - 0.05), 1.005)
    rot = 30 if exp in ("fashion", "cifar10") else 0
    ax.set_xticks(range(len(names)), names, rotation=rot, ha="right" if rot else "center")
    ax.set_ylabel("F1 của lớp")
    ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_title(f"{TITLE[exp]}: F1 theo từng lớp", pad=30)
    save(fig, f"ch3_{exp}_f1class")


def fig_wrong(exp):
    m = _best(exp)
    if not m:
        return
    _, _, xte, yte = load_raw(exp)
    p = np.load(MODELS / "results" / f"{exp}__{m}_proba.npy")
    pred = p.argmax(1)
    wrong = np.flatnonzero(pred != yte)
    wrong = wrong[np.argsort(-p[wrong, pred[wrong]])][:12]       # sai mà tự tin nhất trước
    names = CLASSES[exp]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(W, W / cols * rows * 1.5 + 0.5), squeeze=False)
    for ax, i in zip(axes.flat, wrong):
        show(ax, xte[i])
        ax.set_title(f"thật: {names[yte[i]]}\nđoán: {names[pred[i]]}\n({vn(100 * p[i, pred[i]], 1)} %)",
                     fontsize=9.5, color=C["red"])
    for ax in axes.flat[len(wrong):]:
        ax.axis("off")
    fig.subplots_adjust(hspace=0.36, wspace=0.12, left=0, right=1, bottom=0, top=1 - 0.75 / fig.get_figheight())
    fig.suptitle(f"{TITLE[exp]}: ảnh {MODEL_NAME[m]} đoán sai tự tin nhất\n(nhãn thật → nhãn đoán, xác suất)",
                 fontweight="bold", fontsize=11.5, y=1.0)
    save(fig, f"ch3_{exp}_wrong", "png")


def fig_overview():
    ms = MODEL_IDS
    acc = np.full((len(EXPS), len(ms)), np.nan)
    tt = np.full_like(acc, np.nan)
    pr = np.full_like(acc, np.nan)
    inf = np.full_like(acc, np.nan)
    for i, e in enumerate(EXPS):
        for j, m in enumerate(ms):
            if has_result(e, m):
                r = load_result(e, m)
                acc[i, j], tt[i, j], pr[i, j], inf[i, j] = 100 * r["accuracy"], r["train_time"], r["params"], r["infer_ms"]
    # bản đồ nhiệt độ chính xác
    fig, ax = plt.subplots(figsize=(W, 3.4))
    im = ax.imshow(acc, cmap="YlGn", vmin=np.nanmin(acc) - 5, vmax=100, aspect="auto")
    for i in range(len(EXPS)):
        for j in range(len(ms)):
            ax.text(j, i, "—" if np.isnan(acc[i, j]) else vn(acc[i, j], 2), ha="center", va="center", fontsize=12,
                    fontweight="bold" if not np.isnan(acc[i, j]) and acc[i, j] == np.nanmax(acc[i]) else "normal")
    ax.set_xticks(range(len(ms)), [SHORT[m].replace(" ", "\n") for m in ms])
    ax.set_yticks(range(len(EXPS)), [TITLE[e] for e in EXPS])
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="Accuracy (%)", shrink=0.9)
    ax.set_title("Độ chính xác trên tập kiểm tra: 4 bộ dữ liệu × 5 mô hình")
    save(fig, "ch3_overview_acc")
    # thời gian huấn luyện + suy luận
    for arr, name, lab, title in ((tt, "ch3_overview_time", "Thời gian huấn luyện (giây, thang log)",
                                   "Thời gian huấn luyện"),
                                  (inf, "ch3_overview_infer", "ms / ảnh (thang log)", "Thời gian suy luận mỗi ảnh"),
                                  (pr, "ch3_overview_params", "Số tham số (thang log)", "Số tham số của mô hình")):
        fig, ax = plt.subplots(figsize=(W, 3.6))
        wdt = 0.16
        for j, m in enumerate(ms):
            xs = np.arange(len(EXPS)) + (j - 2) * wdt
            ax.bar(xs, np.nan_to_num(arr[:, j]), wdt, color=MCOL[m], label=SHORT[m])
            for x_, v in zip(xs, arr[:, j]):
                if not np.isnan(v):
                    s = vn(v, 0) if v >= 10 else vn(v, 2 if v >= 0.1 else 3)
                    if arr is pr:
                        s = vn(v / 1e3, 0) + "K" if v < 1e6 else vn(v / 1e6, 2) + "M"
                    ax.text(x_, v * 1.08, s, ha="center", va="bottom", rotation=90, fontsize=8)
        ax.set_yscale("log")
        ax.set_ylim(np.nanmin(arr) / 2, np.nanmax(arr) * 6)
        ax.set_xticks(range(len(EXPS)), [XNAME[e] for e in EXPS])
        ax.set_ylabel(lab, fontsize=10.5)
        ax.legend(ncol=5, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09))
        ax.set_title(title)
        save(fig, name)


# ------------------------------------------------------------------ trực quan hoá mô hình (PyTorch, CPU)
def _torch_model(exp, mid):
    import torch
    from tl3.ch3_torch import BasicCNN, ImprovedCNN
    f = MODELS / f"{exp}_{mid}.pt"
    if not f.exists():
        return None, None
    meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
    m = (ImprovedCNN(meta["in_ch"], meta["n_classes"], tuple(meta["widths"])) if mid == "torch_improved"
         else BasicCNN(meta["in_ch"], meta["n_classes"], meta["size"]))
    m.load_state_dict(torch.load(f, map_location="cpu"))
    return m.eval(), meta


def fig_filters_fmaps(exp):
    import torch
    from tl3.ch3_torch import to_tensor
    m, meta = _torch_model(exp, "torch_basic")
    if m is None:
        return print("   bỏ qua fmaps", exp)
    w = m.features[0].weight.detach().numpy()               # [32, C, 3, 3]
    fig, axes = plt.subplots(4, 8, figsize=(W, W * 0.5))
    for i, ax in enumerate(axes.flat):
        f = w[i].transpose(1, 2, 0)
        f = (f - f.min()) / (f.max() - f.min() + 1e-8)
        ax.imshow(f[..., 0] if f.shape[2] == 1 else f, cmap="RdBu_r" if f.shape[2] == 1 else None,
                  interpolation="nearest")
        ax.axis("off")
    fig.suptitle(f"{TITLE[exp]}: 32 bộ lọc 3×3 đã học của tầng Conv1 (Basic CNN – PyTorch)", fontweight="bold")
    save(fig, f"ch3_{exp}_filters", "png")
    _, _, (xte, yte) = load(exp)
    x = resize_nhwc(xte[:1], meta["size"])
    with torch.no_grad():
        a1 = m.features[1](m.features[0](to_tensor(x)))[0].numpy()
        a2 = m.features[4](m.features[3](m.features[2](torch.from_numpy(a1[None]))))[0].numpy()
    fig = plt.figure(figsize=(W, W * 0.66))
    gs = fig.add_gridspec(5, 8, wspace=0.06, hspace=0.12, height_ratios=[1, 1, 0.28, 1, 1],
                          left=0.0, right=1.0, bottom=0.0, top=0.80)
    for i in range(16):
        ax = fig.add_subplot(gs[i // 8, i % 8])
        ax.imshow(a1[i], cmap="viridis")
        ax.axis("off")
    for i in range(16):
        ax = fig.add_subplot(gs[3 + i // 8, i % 8])
        ax.imshow(a2[i], cmap="inferno")
        ax.axis("off")
    ax = fig.add_axes([0.0, 0.82, 0.12, 0.17])
    show(ax, x[0])
    fig.text(0.14, 0.94, f"{TITLE[exp]}: bản đồ đặc trưng của Basic CNN (PyTorch)", fontsize=12, fontweight="bold")
    fig.text(0.14, 0.885, f"Hai hàng trên — Conv1 + ReLU: 16/32 kênh, {a1.shape[1]}×{a1.shape[2]}", fontsize=11,
             color=C["teal"], fontweight="bold")
    fig.text(0.14, 0.835, f"Hai hàng dưới — Conv2 + ReLU: 16/64 kênh, {a2.shape[1]}×{a2.shape[2]}", fontsize=11,
             color=C["red"], fontweight="bold")
    save(fig, f"ch3_{exp}_fmaps", "png")


def fig_gradcam(exp):
    import torch
    from tl3.ch3_torch import GradCAM, to_tensor
    m, meta = _torch_model(exp, "torch_improved")
    if m is None:
        return print("   bỏ qua gradcam", exp)
    _, _, xte_raw, yte = load_raw(exp)
    _, _, (xte, _) = load(exp)
    rng = np.random.default_rng(7)
    idx = np.concatenate([rng.choice(np.flatnonzero(yte == c), 1) for c in range(len(CLASSES[exp]))])[:10]
    if len(idx) < 6:
        idx = np.concatenate([rng.choice(np.flatnonzero(yte == c), 2, replace=False) for c in range(len(CLASSES[exp]))])
    x = resize_nhwc(xte[idx], meta["size"])
    t = to_tensor(x)
    mean = torch.tensor(meta["mean"]).view(1, -1, 1, 1)
    std = torch.tensor(meta["std"]).view(1, -1, 1, 1)
    layer = [b for b in m.blocks if b.__class__.__name__ == "ResBlock"][-1]
    cams, cls = GradCAM(m, layer)((t - mean) / std)
    names = CLASSES[exp]
    n = len(idx)
    cols = 5 if n >= 10 else 6
    rows = 2 * int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(W, W / cols * rows * 1.18 + 0.6))
    for k in range(n):
        r, c = 2 * (k // cols), k % cols
        show(axes[r, c], x[k])
        axes[r, c].set_title(names[yte[idx[k]]].split(" (")[0], fontsize=9.5)
        show(axes[r + 1, c], x[k])
        axes[r + 1, c].imshow(cams[k], cmap="jet", alpha=0.5, vmin=0, vmax=1)
        ok = cls[k] == yte[idx[k]]
        axes[r + 1, c].set_title(names[cls[k]].split(" (")[0], fontsize=9.5, color=C["green"] if ok else C["red"])
    fig.suptitle(f"{TITLE[exp]}: Grad-CAM của Improved CNN – PyTorch (khối dư cuối)\n"
                 "hàng ảnh: nhãn thật · hàng bản đồ nhiệt: nhãn đoán (xanh đúng, đỏ sai)", fontweight="bold",
                 fontsize=11.5, y=1.0)
    fig.subplots_adjust(hspace=0.3, wspace=0.06, left=0, right=1, bottom=0, top=1 - 0.8 / fig.get_figheight())
    save(fig, f"ch3_{exp}_gradcam", "png")


PARTS = {"data": [fig_samples, fig_dist, fig_mean, fig_pixels, fig_aug],
         "results": [fig_curves, fig_cm, fig_metrics, fig_f1class, fig_wrong],
         "viz": [fig_filters_fmaps, fig_gradcam]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="all", choices=["all", *PARTS])
    ap.add_argument("--exp", nargs="*", default=EXPS)
    a = ap.parse_args()
    limit_threads(4)
    parts = list(PARTS) if a.part == "all" else [a.part]
    for p in parts:
        for e in a.exp:
            for f in PARTS[p]:
                f(e)
    if "results" in parts:
        fig_overview()


if __name__ == "__main__":
    main()
