"""Hình kết quả Chương 4: đường học, dự báo trên test, phần dư, so sánh framework, thí nghiệm bổ trợ.

Đọc outputs/metrics/ch4_<exp>_<fw>.json, outputs/preds/ch4_<exp>_<fw>.npz, ch4_extras.json; ghi figures/ch4_*.pdf
và outputs/metrics/ch4_summary.json (mô hình tốt nhất, baseline trung bình trượt, tổng hợp so sánh).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch4_data import Prepared, experiment  # noqa: E402
from tl3.ch4_metrics import forecast_metrics  # noqa: E402
from tl3.common import OUT, load_json, save_json  # noqa: E402
from tl3.plot_style import C, W, plt, vn_axis  # noqa: E402
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

ARCHS = ("rnn", "lstm", "gru", "bilstm")
AN = {"rnn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}
AC = {"rnn": C["orange"], "lstm": C["blue"], "gru": C["green"], "bilstm": C["purple"]}
FW = {"torch": "PyTorch", "keras": "Keras"}
EXPN = {"ecommerce": "E-commerce", "aapl": "AAPL", "gold": "Vàng GLD"}
UNIT = {"ecommerce": "Total_Amount (TRY)", "aapl": "Close (USD)", "gold": "GLD Close (USD)"}
ZOOM = {"ecommerce": 30, "aapl": 60, "gold": 60}


def load(exp, fw):
    return load_json(f"ch4_{exp}_{fw}"), dict(np.load(OUT / "preds" / f"ch4_{exp}_{fw}.npz"))


def fig_loss(exp, fw, m):
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.6), sharex=False)
    for ax, a in zip(axes.flat, ARCHS):
        h = m["history"][a]
        ax.plot(h["loss"], color=AC[a], lw=1.6, label="train")
        ax.plot(h["val_loss"], color="black", lw=1.4, ls="--", label="validation")
        best = int(np.argmin(h["val_loss"]))
        ax.scatter([best], [h["val_loss"][best]], color=C["red"], zorder=5, s=28, label="epoch tốt nhất")
        ax.set_title(AN[a])
        ax.set_xlabel("epoch", fontsize=10)
        ax.tick_params(labelsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9.5, frameon=False)
    fig.suptitle(f"{EXPN[exp]} — đường học MSE ({FW[fw]})", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))                 # chú giải nằm dưới, không che đường học
    save(fig, f"ch4_{exp}_loss_{fw}")


def fig_pred(exp, fw, p):
    d = pd.to_datetime(p["dates"])
    fig, ax = plt.subplots(figsize=(W, 3.4))
    ax.plot(d, p["y_true"], color="black", lw=1.8, label="Thực tế")
    for a in ARCHS:
        ax.plot(d, p[a], color=AC[a], lw=1.1, alpha=0.9, label=AN[a])
    ax.set_ylabel(UNIT[exp])
    ax.set_title(f"{EXPN[exp]} — dự báo 1 bước trên test ({FW[fw]})")
    ax.legend(ncol=5, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.tick_params(axis="x", rotation=25, labelsize=9.5)
    if exp == "ecommerce":
        vn_axis(ax)
    save(fig, f"ch4_{exp}_pred_{fw}")


def best_model(mt, mk):
    cands = [(fw, a, m[a]["val"]["rmse"]) for fw, m in (("torch", mt), ("keras", mk)) for a in ARCHS]
    return min(cands, key=lambda c: c[2])[:2]


def fig_zoom(exp, pt, pk, data):
    k = ZOOM[exp]
    d = pd.to_datetime(pt["dates"])[-k:]
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.plot(d, pt["y_true"][-k:], "o-", color="black", lw=1.8, ms=3, label="Thực tế")
    ax.plot(d, pt["naive"][-k:], color=C["gray"], lw=1.2, ls=":", label="Naive (hôm qua)")
    if exp == "ecommerce":
        ax.plot(d, data.moving_average(7)[-k:], color=C["brown"], lw=1.2, ls="-.", label="TB trượt 7 ngày")
    for fw, pp, ls in (("torch", pt, "-"), ("keras", pk, "--")):
        ax.plot(d, pp["lstm"][-k:], color=C["blue"], ls=ls, lw=1.4, label=f"LSTM {FW[fw]}")
        ax.plot(d, pp["gru"][-k:], color=C["green"], ls=ls, lw=1.4, label=f"GRU {FW[fw]}")
    ax.set_ylabel(UNIT[exp])
    ax.set_title(f"{EXPN[exp]} — {k} {'ngày' if exp == 'ecommerce' else 'phiên'} cuối của test")
    ax.legend(ncol=3, fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), frameon=False)
    ax.tick_params(axis="x", rotation=25, labelsize=9.5)
    if exp == "ecommerce":
        vn_axis(ax)
    save(fig, f"ch4_{exp}_zoom")


def fig_resid(exp, fw, a, p):
    r = p["y_true"] - p[a]
    d = pd.to_datetime(p["dates"])
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9), gridspec_kw={"width_ratios": [1.7, 1]})
    axes[0].plot(d, r, color=C["red"], lw=0.9)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title(f"Phần dư ({AN[a]} {FW[fw]})")
    axes[0].tick_params(axis="x", rotation=25, labelsize=9)
    axes[1].hist(r, bins=30, color=C["sky"], edgecolor="white")
    axes[1].axvline(0, color="black", lw=0.8)
    axes[1].set_title("Phân phối phần dư")
    fig.tight_layout()
    save(fig, f"ch4_{exp}_resid")


def fig_bars(exp, mt, mk, extra):
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0))
    x = np.arange(len(ARCHS))
    for ax, key, lab in ((axes[0], "rmse", "RMSE"), (axes[1], "mae", "MAE")):
        ax.bar(x - 0.2, [mt[a]["test"][key] for a in ARCHS], 0.4, color=C["blue"], label="PyTorch")
        ax.bar(x + 0.2, [mk[a]["test"][key] for a in ARCHS], 0.4, color=C["orange"], label="Keras")
        ax.axhline(mt["naive"]["test"][key], color="black", ls="--", lw=1.2, label="Naive")
        if "ma7" in extra:
            ax.axhline(extra["ma7"][key], color=C["brown"], ls=":", lw=1.4, label="TB trượt 7")
        lo = min(min(mt[a]["test"][key] for a in ARCHS), min(mk[a]["test"][key] for a in ARCHS),
                 mt["naive"]["test"][key])
        hi = max(max(mt[a]["test"][key] for a in ARCHS), max(mk[a]["test"][key] for a in ARCHS),
                 mt["naive"]["test"][key], extra.get("ma7", {}).get(key, 0))
        ax.set_ylim(lo * 0.9 if exp != "ecommerce" else 0, hi * 1.06)
        ax.set_xticks(x, [AN[a].replace("SimpleRNN", "S.RNN") for a in ARCHS], fontsize=9)
        ax.set_title(f"{lab} trên test")
        if exp == "ecommerce":
            vn_axis(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9, frameon=False)
    fig.suptitle(f"{EXPN[exp]} — so sánh kiến trúc × framework", fontweight="bold")
    fig.tight_layout(rect=(0, 0.07, 1, 1))                 # chú giải nằm dưới, không che cột
    save(fig, f"ch4_{exp}_bars")


def fig_overview(allm):
    """Bản đồ nhiệt RMSE / RMSE naive cho 3 thực nghiệm × 8 mô hình."""
    rows, labels = [], []
    for exp in ("ecommerce", "aapl", "gold"):
        mt, mk = allm[exp]
        rows.append([m[a]["test"]["rmse"] / mt["naive"]["test"]["rmse"] for m in (mt, mk) for a in ARCHS])
        labels.append(EXPN[exp])
    M = np.array(rows)
    fig, ax = plt.subplots(figsize=(W, 2.6))
    im = ax.imshow(M, cmap="RdYlGn_r", vmin=0.6, vmax=1.1, aspect="auto")
    ax.set_xticks(range(8), [f"{AN[a].replace('SimpleRNN', 'S.RNN')}\n{FW[fw]}" for fw in ("torch", "keras")
                           for a in ARCHS], fontsize=8.8)
    ax.set_yticks(range(3), labels)
    ax.grid(False)
    for i in range(3):
        for j in range(8):
            ax.text(j, i, f"{M[i, j]:.3f}".replace(".", ","), ha="center", va="center", fontsize=9.5)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    ax.set_title("Tỉ số RMSE mô hình / RMSE naive trên test (< 1 là thắng naive)")
    save(fig, "ch4_overview_ratio")


def fig_cost(allm):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.9))
    x = np.arange(len(ARCHS))
    mt, mk = allm["aapl"]
    axes[0].bar(x - 0.2, [mt[a]["params"] for a in ARCHS], 0.4, color=C["blue"], label="PyTorch")
    axes[0].bar(x + 0.2, [mk[a]["params"] for a in ARCHS], 0.4, color=C["orange"], label="Keras")
    axes[0].set_xticks(x, [AN[a] for a in ARCHS], fontsize=9)
    axes[0].set_title("Số tham số (AAPL, 6 đặc trưng)")
    axes[0].set_ylim(0, 1.3 * max(max(mt[a]["params"] for a in ARCHS), max(mk[a]["params"] for a in ARCHS)))
    axes[0].legend(fontsize=9, ncol=2, loc="upper left")
    vn_axis(axes[0])
    axes[1].bar(x - 0.2, [mt[a]["sec_per_epoch"] for a in ARCHS], 0.4, color=C["blue"], label="PyTorch GPU")
    axes[1].bar(x + 0.2, [mk[a]["sec_per_epoch"] for a in ARCHS], 0.4, color=C["orange"], label="Keras CPU")
    axes[1].set_xticks(x, [AN[a] for a in ARCHS], fontsize=9)
    axes[1].set_title("Giây / epoch (AAPL)")
    axes[1].set_ylim(0, 1.3 * max(max(mt[a]["sec_per_epoch"] for a in ARCHS),
                                  max(mk[a]["sec_per_epoch"] for a in ARCHS)))
    axes[1].legend(fontsize=9, ncol=2, loc="upper left")
    fig.tight_layout()
    save(fig, "ch4_cost")


def fig_extras(ex):
    # 1) dòng chảy gradient
    cur = ex["gradient"]["curve"]
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.6))
    for k, c, lab in (("rnn", C["orange"], "SimpleRNN"), ("lstm", C["blue"], "LSTM"), ("gru", C["green"], "GRU"),
                      ("lstm_fb3", C["purple"], "LSTM, bias quên = 3")):
        axes[0].plot(cur[k], color=c, label=lab)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("khoảng cách T − t")
    axes[0].set_ylabel(r"$\|\partial h_T / \partial x_t\|$")
    axes[0].set_title("Khởi tạo mặc định")
    axes[0].legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False)
    for sc, c in zip((0.5, 0.9, 1.1, 1.5), (C["sky"], C["green"], C["orange"], C["red"])):
        axes[1].plot(cur[f"lin_rho{sc}"], color=c, label=rf"$\rho(W_{{hh}})$ = {sc}".replace(".", ","))
    axes[1].set_yscale("log")
    axes[1].set_xlabel("khoảng cách T − t")
    axes[1].set_title("RNN-ReLU theo bán kính phổ")
    axes[1].legend(fontsize=9, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.24), frameon=False)
    fig.tight_layout()
    save(fig, "ch4_gradflow")
    # 2) tầm dự báo
    fig, axes = plt.subplots(1, 3, figsize=(W, 3.1))
    for ax, exp in zip(axes, ("ecommerce", "aapl", "gold")):
        h = ex["horizon"][exp]
        hs = np.arange(1, len(h["rmse_h"]) + 1)
        ax.plot(hs, h["rmse_h"], "o-", color=C["blue"], ms=3.5, label="LSTM đệ quy")
        ax.plot(hs, h["naive_h"], "s--", color=C["gray"], ms=3, label="Naive")
        ax.set_title(EXPN[exp], fontsize=11)
        ax.set_xlabel("tầm h", fontsize=10)
        ax.tick_params(labelsize=8.5)
        if exp == "ecommerce":
            vn_axis(ax)
    axes[0].set_ylabel("RMSE")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))                 # chú giải nằm dưới, không che đường RMSE
    save(fig, "ch4_horizon")
    # 3) độ dài cửa sổ
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.8), sharey=False)
    ws = (10, 20, 30, 60, 90)
    for ax, exp in zip(axes, ("ecommerce", "aapl", "gold")):
        for a, c in (("lstm", C["blue"]), ("gru", C["green"])):
            ax.plot(ws, [ex["window"][exp][f"{a}_w{w}"]["ratio"] for w in ws], "o-", color=c, ms=4, label=AN[a])
        ax.axhline(1, color="black", ls="--", lw=1)
        ax.set_title(EXPN[exp], fontsize=11)
        ax.set_xlabel("độ dài cửa sổ", fontsize=10)
        ax.tick_params(labelsize=8.5)
    axes[0].set_ylabel("RMSE / naive")
    axes[0].legend(fontsize=8.5)
    fig.tight_layout()
    save(fig, "ch4_window")
    # 4) MinMax toàn cục vs chuẩn hoá tương đối
    p = dict(np.load(OUT / "preds" / "ch4_minmax.npz"))
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.5))
    for ax, exp in zip(axes, ("aapl", "gold")):
        ax.plot(p[f"{exp}_truth"], color="black", lw=1.5, label="Thực tế")
        ax.plot(p[f"{exp}_minmax_lstm"], color=C["red"], lw=1.2, label="LSTM + MinMax toàn cục")
        ax.plot(p[f"{exp}_relative_lstm"], color=C["blue"], lw=1.0, ls="--", label="LSTM + chuẩn hoá tương đối")
        ax.axhline(ex["minmax"][exp]["train_max"], color=C["orange"], ls=":", lw=1.3, label="max của train")
        ax.set_title(EXPN[exp])
        ax.set_xlabel("chỉ số phiên trong test", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9.5, frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    save(fig, "ch4_minmax")


def main():
    allm, summary = {}, {}
    for exp in ("ecommerce", "aapl", "gold"):
        mt, pt = load(exp, "torch")
        mk, pk = load(exp, "keras")
        allm[exp] = (mt, mk)
        data = Prepared(experiment(exp))
        extra = {}
        if exp == "ecommerce":
            for k in (7, 30):
                extra[f"ma{k}"] = forecast_metrics(data.truth(), data.moving_average(k), data.naive())
        for fw, m, p in (("torch", mt, pt), ("keras", mk, pk)):
            fig_loss(exp, fw, m)
            fig_pred(exp, fw, p)
        fig_zoom(exp, pt, pk, data)
        bfw, ba = best_model(mt, mk)
        fig_resid(exp, bfw, ba, pt if bfw == "torch" else pk)
        fig_bars(exp, mt, mk, extra)
        bm = (mt if bfw == "torch" else mk)[ba]["test"]
        tbest = min(ARCHS, key=lambda a: mt[a]["test"]["rmse"])
        kbest = min(ARCHS, key=lambda a: mk[a]["test"]["rmse"])
        summary[exp] = {"best_fw": FW[bfw], "best_arch": AN[ba], "best": bm,
                        "best_ratio": bm["rmse"] / mt["naive"]["test"]["rmse"],
                        "torch_best": AN[tbest], "keras_best": AN[kbest],
                        "torch_best_rmse": mt[tbest]["test"]["rmse"], "keras_best_rmse": mk[kbest]["test"]["rmse"],
                        "torch_mean_rmse": float(np.mean([mt[a]["test"]["rmse"] for a in ARCHS])),
                        "keras_mean_rmse": float(np.mean([mk[a]["test"]["rmse"] for a in ARCHS])),
                        "n_beat_naive": int(sum(m[a]["test"]["rmse"] < mt["naive"]["test"]["rmse"]
                                                for m in (mt, mk) for a in ARCHS)),
                        **extra}
    fig_overview(allm)
    fig_cost(allm)
    try:
        fig_extras(load_json("ch4_extras"))
    except FileNotFoundError:
        print("chưa có ch4_extras.json")
    save_json("ch4_summary", summary)


if __name__ == "__main__":
    main()
