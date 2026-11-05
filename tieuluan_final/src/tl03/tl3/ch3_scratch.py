"""Chương 3 — Basic CNN viết tay hoàn toàn bằng NumPy: im2col, tích chập, ReLU, Max Pooling, Softmax–Cross-Entropy,
lan truyền ngược và tối ưu Adam. Không dùng thư viện học sâu nào; mọi đạo hàm được tính tường minh.

Kiến trúc (giống hệt Basic CNN bằng PyTorch và TensorFlow để so sánh công bằng):
    Conv(3×3, 32, pad 1) → ReLU → MaxPool 2×2 → Conv(3×3, 64, pad 1) → ReLU → MaxPool 2×2
    → Flatten → FC(128) → ReLU → FC(K) → Softmax
"""
import time

import numpy as np


def im2col(X, kh, kw, stride=1, pad=0):
    """X [N, C, H, W] -> ma trận cột [N·Ho·Wo, C·kh·kw]: mỗi hàng là một cửa sổ (receptive field) đã duỗi thẳng.

    Thay vì 4 vòng lặp lồng nhau, chỉ lặp kh·kw lần (9 lần với kernel 3×3); mỗi lần cắt một lát ảnh dịch chuyển."""
    N, C, H, W = X.shape
    Ho = (H + 2 * pad - kh) // stride + 1
    Wo = (W + 2 * pad - kw) // stride + 1
    Xp = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad))) if pad else X
    cols = np.empty((N, C, kh, kw, Ho, Wo), dtype=X.dtype)
    for i in range(kh):
        for j in range(kw):
            cols[:, :, i, j] = Xp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride]
    return cols.transpose(0, 4, 5, 1, 2, 3).reshape(N * Ho * Wo, -1), Ho, Wo


def col2im(dcols, X_shape, kh, kw, Ho, Wo, stride=1, pad=0):
    """Phép ngược của im2col cho lan truyền ngược: cộng dồn gradient của các cửa sổ chồng lấn về đúng vị trí điểm ảnh."""
    N, C, H, W = X_shape
    d = dcols.reshape(N, Ho, Wo, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    dXp = np.zeros((N, C, H + 2 * pad, W + 2 * pad), dtype=dcols.dtype)
    for i in range(kh):
        for j in range(kw):
            dXp[:, :, i:i + stride * Ho:stride, j:j + stride * Wo:stride] += d[:, :, i, j]
    return dXp[:, :, pad:pad + H, pad:pad + W] if pad else dXp


def conv_forward(X, W, b, pad=1):
    """Z = X * W + b. W [F, C, kh, kw]; trả Z [N, F, Ho, Wo] và cache cho lan truyền ngược."""
    F, C, kh, kw = W.shape
    cols, Ho, Wo = im2col(X, kh, kw, 1, pad)
    Z = cols @ W.reshape(F, -1).T + b                       # [N·Ho·Wo, F]
    Z = Z.reshape(X.shape[0], Ho, Wo, F).transpose(0, 3, 1, 2)
    return Z, (X.shape, cols, W, pad, Ho, Wo)


def conv_backward(dZ, cache, need_dx=True):
    """Từ dL/dZ tính dL/dW = dZᵀ·cols, dL/db = Σ dZ và dL/dX = col2im(dZ·W).
    Tầng tích chập đầu tiên không cần dL/dX (đầu vào là ảnh) nên bỏ qua để tiết kiệm bộ nhớ và thời gian."""
    X_shape, cols, W, pad, Ho, Wo = cache
    F, C, kh, kw = W.shape
    dZ2 = dZ.transpose(0, 2, 3, 1).reshape(-1, F)            # [N·Ho·Wo, F]
    dW = (dZ2.T @ cols).reshape(W.shape)
    db = dZ2.sum(axis=0)
    dX = col2im(dZ2 @ W.reshape(F, -1), X_shape, kh, kw, Ho, Wo, 1, pad) if need_dx else None
    return dX, dW, db


def relu_forward(Z):
    return np.maximum(Z, 0), Z


def relu_backward(dA, Z):
    return dA * (Z > 0)


def maxpool_forward(X, size=2):
    """Max Pooling size×size, bước = size. Lưu mặt nạ vị trí cực đại để chuyển gradient về đúng ô."""
    N, C, H, W = X.shape
    Xr = X[:, :, :H - H % size, :W - W % size].reshape(N, C, H // size, size, W // size, size)
    out = Xr.max(axis=(3, 5))
    mask = Xr == out[:, :, :, None, :, None]
    mask = mask / mask.sum(axis=(3, 5), keepdims=True)       # chia đều nếu có nhiều cực đại bằng nhau
    return out, (X.shape, mask, size)


def maxpool_backward(dout, cache):
    X_shape, mask, size = cache
    N, C, H, W = X_shape
    dXr = mask * dout[:, :, :, None, :, None]
    dX = np.zeros(X_shape, dtype=dout.dtype)
    dX[:, :, :H - H % size, :W - W % size] = dXr.reshape(N, C, (H // size) * size, (W // size) * size)
    return dX


def softmax_cross_entropy(logits, y):
    """Softmax ổn định số học (trừ max) + cross-entropy; gradient theo logits là (p − onehot)/N."""
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    N = len(y)
    loss = -np.log(p[np.arange(N), y] + 1e-12).mean()
    dlogits = p.copy()
    dlogits[np.arange(N), y] -= 1
    return loss, p, dlogits / N


class ScratchCNN:
    """Basic CNN hai khối tích chập, khởi tạo He, tối ưu Adam; mọi tham số nằm trong dict self.p."""

    def __init__(self, in_shape, n_classes, c1=32, c2=64, hidden=128, seed=42):
        C, H, W = in_shape
        rng = np.random.default_rng(seed)
        he = lambda shape, fan_in: (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)
        flat = c2 * (H // 4) * (W // 4)
        self.p = {
            "W1": he((c1, C, 3, 3), C * 9), "b1": np.zeros(c1, np.float32),
            "W2": he((c2, c1, 3, 3), c1 * 9), "b2": np.zeros(c2, np.float32),
            "W3": he((flat, hidden), flat), "b3": np.zeros(hidden, np.float32),
            "W4": he((hidden, n_classes), hidden), "b4": np.zeros(n_classes, np.float32),
        }
        self.m = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: np.zeros_like(v) for k, v in self.p.items()}
        self.t = 0

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def forward(self, X):
        """X [N, C, H, W] -> logits [N, K]; giữ cache từng tầng cho backward."""
        p, c = self.p, {}
        z1, c["conv1"] = conv_forward(X, p["W1"], p["b1"])
        a1, c["relu1"] = relu_forward(z1)
        h1, c["pool1"] = maxpool_forward(a1)
        z2, c["conv2"] = conv_forward(h1, p["W2"], p["b2"])
        a2, c["relu2"] = relu_forward(z2)
        h2, c["pool2"] = maxpool_forward(a2)
        flat = h2.reshape(len(X), -1)
        z3 = flat @ p["W3"] + p["b3"]
        a3, c["relu3"] = relu_forward(z3)
        logits = a3 @ p["W4"] + p["b4"]
        c.update(flat=flat, a3=a3, h2_shape=h2.shape)
        self.cache = c
        return logits

    def backward(self, dlogits):
        """Lan truyền ngược: Softmax → FC2 → ReLU → FC1 → MaxPool → ReLU → Conv2 → MaxPool → ReLU → Conv1."""
        p, c, g = self.p, self.cache, {}
        g["W4"] = c["a3"].T @ dlogits
        g["b4"] = dlogits.sum(0)
        da3 = dlogits @ p["W4"].T
        dz3 = relu_backward(da3, c["relu3"])
        g["W3"] = c["flat"].T @ dz3
        g["b3"] = dz3.sum(0)
        dh2 = (dz3 @ p["W3"].T).reshape(c["h2_shape"])
        da2 = maxpool_backward(dh2, c["pool2"])
        dz2 = relu_backward(da2, c["relu2"])
        dh1, g["W2"], g["b2"] = conv_backward(dz2, c["conv2"])
        da1 = maxpool_backward(dh1, c["pool1"])
        dz1 = relu_backward(da1, c["relu1"])
        _, g["W1"], g["b1"] = conv_backward(dz1, c["conv1"], need_dx=False)
        return g

    def adam_step(self, g, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        """Adam: trung bình động bậc một/bậc hai của gradient, hiệu chỉnh độ lệch theo bước t."""
        self.t += 1
        for k in self.p:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t)
            vh = self.v[k] / (1 - b2 ** self.t)
            self.p[k] -= (lr * mh / (np.sqrt(vh) + eps)).astype(np.float32)

    def predict_proba(self, X, batch=200):
        out = []
        for i in range(0, len(X), batch):
            logits = self.forward(X[i:i + batch])
            z = np.exp(logits - logits.max(1, keepdims=True))
            out.append(z / z.sum(1, keepdims=True))
        self.cache = None
        return np.concatenate(out)


def train_scratch(model, xtr, ytr, xva, yva, epochs, batch=64, lr=1e-3, seed=42, log=print):
    """Vòng huấn luyện mini-batch: xáo trộn, forward, loss, backward, Adam; cuối mỗi epoch đo trên validation."""
    rng = np.random.default_rng(seed)
    hist = {"loss": [], "acc": [], "val_loss": [], "val_acc": [], "time": []}
    for ep in range(epochs):
        t0, idx = time.perf_counter(), rng.permutation(len(xtr))
        tot_loss = tot_ok = 0.0
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            logits = model.forward(xtr[b])
            loss, prob, dlog = softmax_cross_entropy(logits, ytr[b])
            model.adam_step(model.backward(dlog), lr)
            tot_loss += loss * len(b)
            tot_ok += (prob.argmax(1) == ytr[b]).sum()
        pv = model.predict_proba(xva)
        hist["loss"].append(tot_loss / len(idx))
        hist["acc"].append(tot_ok / len(idx))
        hist["val_loss"].append(float(-np.log(pv[np.arange(len(yva)), yva] + 1e-12).mean()))
        hist["val_acc"].append(float((pv.argmax(1) == yva).mean()))
        hist["time"].append(time.perf_counter() - t0)
        log(f"epoch {ep + 1}/{epochs} loss {hist['loss'][-1]:.4f} acc {hist['acc'][-1]:.4f} "
            f"val_loss {hist['val_loss'][-1]:.4f} val_acc {hist['val_acc'][-1]:.4f} {hist['time'][-1]:.0f}s")
    return hist


def gradient_check(seed=0, eps=1e-6):
    """Kiểm tra đạo hàm số: so gradient giải tích với sai phân trung tâm trên một lô nhỏ (float64).

    eps phải nhỏ (1e-6): với eps lớn (1e-3) bước nhiễu có thể vượt qua điểm gãy của ReLU/MaxPool và sai số giả tăng vọt."""
    rng = np.random.default_rng(seed)
    m = ScratchCNN((1, 8, 8), 3, c1=2, c2=3, hidden=5, seed=seed)
    m.p = {k: v.astype(np.float64) for k, v in m.p.items()}
    X = rng.standard_normal((4, 1, 8, 8))
    y = rng.integers(0, 3, 4)
    _, _, d = softmax_cross_entropy(m.forward(X), y)
    g = m.backward(d)
    errs = {}
    for k in ("W1", "W2", "W3", "W4"):
        num = np.zeros_like(m.p[k]).ravel()
        flat = m.p[k].ravel()
        for i in range(min(flat.size, 20)):
            old = flat[i]
            flat[i] = old + eps
            lp = softmax_cross_entropy(m.forward(X), y)[0]
            flat[i] = old - eps
            lm = softmax_cross_entropy(m.forward(X), y)[0]
            flat[i] = old
            num[i] = (lp - lm) / (2 * eps)
        a, n = g[k].ravel()[:20], num[:20]
        errs[k] = float(np.abs(a - n).max() / (np.abs(a).max() + np.abs(n).max() + 1e-12))
    return errs
