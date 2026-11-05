"""Chương 4 — RNN viết từ đầu bằng NumPy: lan truyền xuôi, lan truyền ngược qua thời gian (BPTT), kiểm tra gradient.

h_t = tanh(W_xh x_t + W_hh h_{t-1} + b_h),   ŷ = W_hy h_T + b_y,   L = ½ (ŷ − y)²   (many-to-one)
"""
import numpy as np


class RNNScratch:
    """RNN một lớp, đầu ra hồi quy từ trạng thái ẩn cuối cùng."""

    def __init__(self, n_in, n_hidden, seed=0):
        rng = np.random.default_rng(seed)
        s = 1.0 / np.sqrt(n_hidden)
        self.p = {"Wxh": rng.uniform(-s, s, (n_hidden, n_in)),
                  "Whh": rng.uniform(-s, s, (n_hidden, n_hidden)),
                  "bh": np.zeros(n_hidden),
                  "Why": rng.uniform(-s, s, (1, n_hidden)),
                  "by": np.zeros(1)}

    def forward(self, x):
        """x: (T, n_in). Lưu mọi h_t để dùng lại khi lan truyền ngược."""
        p = self.p
        hs = [np.zeros(p["Whh"].shape[0])]
        for t in range(len(x)):
            hs.append(np.tanh(p["Wxh"] @ x[t] + p["Whh"] @ hs[-1] + p["bh"]))
        self.x, self.hs = x, hs
        return (p["Why"] @ hs[-1] + p["by"])[0]

    def backward(self, yhat, y):
        """BPTT: δ_T = W_hyᵀ (ŷ − y); δ_{t−1} = W_hhᵀ [(1 − h_t²) ⊙ δ_t]; cộng dồn gradient của trọng số dùng chung."""
        p, hs, x = self.p, self.hs, self.x
        g = {k: np.zeros_like(v) for k, v in p.items()}
        dy = yhat - y
        g["Why"] = dy * hs[-1][None, :]
        g["by"] = np.array([dy])
        dh = p["Why"][0] * dy
        self.dh_norms = []
        for t in range(len(x), 0, -1):
            self.dh_norms.append(np.linalg.norm(dh))
            da = (1.0 - hs[t] ** 2) * dh            # qua đạo hàm tanh
            g["Wxh"] += np.outer(da, x[t - 1])
            g["Whh"] += np.outer(da, hs[t - 1])
            g["bh"] += da
            dh = p["Whh"].T @ da                    # truyền về bước t − 1
        return g

    def loss(self, x, y):
        return 0.5 * (self.forward(x) - y) ** 2


def grad_check(model, x, y, eps=1e-5, n_checks=12, seed=0):
    """So gradient BPTT với sai phân trung tâm (L(θ+ε) − L(θ−ε)) / 2ε trên các phần tử ngẫu nhiên."""
    rng = np.random.default_rng(seed)
    g = model.backward(model.forward(x), y)
    rel = {}
    for k, w in model.p.items():
        errs = []
        for _ in range(n_checks):
            idx = tuple(rng.integers(0, s) for s in w.shape)
            old = w[idx]
            w[idx] = old + eps
            lp = model.loss(x, y)
            w[idx] = old - eps
            lm = model.loss(x, y)
            w[idx] = old
            num = (lp - lm) / (2 * eps)
            errs.append(abs(num - g[k][idx]) / max(1e-12, abs(num) + abs(g[k][idx])))
        rel[k] = float(np.max(errs))
    return rel


def sgd_train(model, X, Y, Xv, Yv, epochs=30, lr=0.02, clip=5.0, seed=0):
    """SGD từng mẫu với cắt chuẩn gradient; trả lịch sử MSE train/validation."""
    rng = np.random.default_rng(seed)
    hist = {"loss": [], "val_loss": []}
    for _ in range(epochs):
        for i in rng.permutation(len(X)):
            g = model.backward(model.forward(X[i]), Y[i])
            norm = np.sqrt(sum(np.sum(v ** 2) for v in g.values()))
            scale = min(1.0, clip / (norm + 1e-12))
            for k in model.p:
                model.p[k] -= lr * scale * g[k]
        hist["loss"].append(float(np.mean([(model.forward(x) - y) ** 2 for x, y in zip(X, Y)])))
        hist["val_loss"].append(float(np.mean([(model.forward(x) - y) ** 2 for x, y in zip(Xv, Yv)])))
    return hist
