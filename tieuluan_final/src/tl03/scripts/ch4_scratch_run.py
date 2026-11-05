"""RNN NumPy từ đầu: kiểm tra gradient, đối chiếu với torch.nn.RNN, huấn luyện trên chuỗi sin nhiễu.

Ghi outputs/metrics/ch4_scratch.json và hình figures/ch4_scratch_*.pdf.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch4_data import make_windows  # noqa: E402
from tl3.ch4_scratch import RNNScratch, grad_check, sgd_train  # noqa: E402
from tl3.common import limit_threads, save_json  # noqa: E402
from tl3.plot_style import C, W, plt, save  # noqa: E402


def sine_series(n=600, seed=0):
    """Chuỗi tổng hai sóng sin chu kỳ 25 và 60 bước cộng nhiễu Gauss σ = 0,1."""
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return np.sin(2 * np.pi * t / 25) + 0.5 * np.sin(2 * np.pi * t / 60) + 0.1 * rng.standard_normal(n)


def torch_equivalence(model, x):
    """Nạp cùng trọng số vào torch.nn.RNN và so trạng thái ẩn cuối cùng."""
    import torch
    rnn = torch.nn.RNN(x.shape[1], model.p["Whh"].shape[0], batch_first=True)
    with torch.no_grad():
        rnn.weight_ih_l0.copy_(torch.tensor(model.p["Wxh"]))
        rnn.weight_hh_l0.copy_(torch.tensor(model.p["Whh"]))
        rnn.bias_ih_l0.copy_(torch.tensor(model.p["bh"]))
        rnn.bias_hh_l0.zero_()
        out, _ = rnn(torch.tensor(x[None], dtype=torch.float64).float())
    model.forward(x)
    return float(np.max(np.abs(out[0, -1].double().numpy() - model.hs[-1])))


def main():
    limit_threads(2)
    s = sine_series()
    X, y = make_windows(s[:, None], s, 20)
    X, y = X.astype(np.float64), y.astype(np.float64)
    a, b = 400, 490
    model = RNNScratch(1, 16, seed=1)
    rel = grad_check(model, X[0], y[0])
    diff = torch_equivalence(model, X[0])
    hist = sgd_train(model, X[:a], y[:a], X[a:b], y[a:b], epochs=40)
    pred = np.array([model.forward(x) for x in X[b:]])
    naive = X[b:, -1, 0]
    rmse = float(np.sqrt(np.mean((pred - y[b:]) ** 2)))
    rmse_naive = float(np.sqrt(np.mean((naive - y[b:]) ** 2)))
    # chuẩn ‖δ_t‖ khi lan truyền ngược qua 20 bước
    model.backward(model.forward(X[b]), y[b])
    dh = model.dh_norms
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.4))
    axes[0].plot(hist["loss"], color=C["blue"], label="train")
    axes[0].plot(hist["val_loss"], color=C["orange"], label="validation")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("epoch")
    axes[0].set_title("MSE của RNN NumPy")
    axes[1].plot(y[b:b + 80], color="black", lw=1.6, label="thực tế")
    axes[1].plot(pred[:80], color=C["red"], lw=1.4, ls="--", label="RNN NumPy")
    axes[1].set_title("Dự báo 1 bước trên test")
    axes[1].set_xlabel("bước")
    # chú giải đặt dưới trục: ở trong khung chúng che đường dự báo
    for ax in axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False)
    fig.tight_layout()
    save(fig, "ch4_scratch_train")
    fig, ax = plt.subplots(figsize=(W, 2.6))
    ax.bar(range(1, len(dh) + 1), dh, color=C["purple"])
    ax.set_xlabel("số bước đã lan truyền ngược (T − t)")
    ax.set_ylabel("‖δ_t‖")
    ax.set_title("Độ lớn gradient theo trạng thái ẩn khi đi ngược thời gian")
    save(fig, "ch4_scratch_dh")
    save_json("ch4_scratch", {"grad_rel_err": rel, "max_rel_err": max(rel.values()), "torch_diff": diff,
                              "rmse": rmse, "rmse_naive": rmse_naive, "n_train": a, "n_val": b - a,
                              "n_test": len(X) - b, "hidden": 16, "window": 20, "epochs": 40,
                              "final_train": hist["loss"][-1], "final_val": hist["val_loss"][-1],
                              "dh_first": float(dh[0]), "dh_last": float(dh[-1]),
                              "params": int(sum(v.size for v in model.p.values()))})
    print("rel err", rel, "torch diff", diff, "rmse", rmse, rmse_naive)


if __name__ == "__main__":
    main()
