"""Mục 2.4.2: mạng MLP viết tay bằng NumPy - lan truyền tiến, lan truyền ngược và Adam.

Kiến trúc giống hệt bản Keras và PyTorch: Dense(128) - ReLU - Dropout - Dense(64) - ReLU - Dropout -
đầu ra. Đầu ra và hàm mất mát đổi theo bài toán:
    'binary'  : sigmoid + entropy chéo nhị phân
    'multi'   : softmax + entropy chéo
    'reg'     : tuyến tính + MSE
Với cả ba cặp, đạo hàm của mất mát theo z ở tầng ra đều rút gọn thành (y_hat - y) / m.
"""
import numpy as np
import scipy.sparse as sp


class MLPScratch:
    def __init__(self, input_dim, hidden=(128, 64), output_dim=1, task="binary", lr=1e-3,
                 dropout=0.2, seed=42):
        self.rng = np.random.default_rng(seed)
        self.task, self.lr, self.p = task, lr, dropout
        dims = [input_dim, *hidden, output_dim]
        # Khởi tạo He: phương sai 2 / fan_in giữ biên độ kích hoạt ổn định qua các tầng ReLU
        self.W = [self.rng.standard_normal((a, b)).astype(np.float32) * np.sqrt(2.0 / a)
                  for a, b in zip(dims[:-1], dims[1:])]
        self.b = [np.zeros((1, b), np.float32) for b in dims[1:]]
        self.m = [np.zeros_like(w) for w in self.W + self.b]   # mômen bậc 1 của Adam
        self.v = [np.zeros_like(w) for w in self.W + self.b]   # mômen bậc 2 của Adam
        self.t = 0

    # ---------------------------------------------------------------- lan truyền tiến
    def _out(self, z):
        if self.task == "binary":
            return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
        if self.task == "multi":
            e = np.exp(z - z.max(1, keepdims=True))
            return e / e.sum(1, keepdims=True)
        return z

    def forward(self, X, train=False):
        a, self.cache = X, []
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = a @ W + b
            last = i == len(self.W) - 1
            if last:
                a = self._out(z)
                mask = None
            else:
                a = np.maximum(z, 0)
                # Dropout ngược: chia cho (1 - p) lúc huấn luyện để lúc suy luận không phải nhân lại
                mask = (self.rng.random(a.shape) >= self.p) / (1 - self.p) if train and self.p > 0 else None
                if mask is not None:
                    a = a * mask
            self.cache.append((z, a, mask))
        self.X_in = X
        return a

    def loss(self, y_hat, y):
        eps = 1e-7
        if self.task == "binary":
            y_hat = np.clip(y_hat, eps, 1 - eps)
            return float(-np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)))
        if self.task == "multi":
            return float(-np.mean(np.log(y_hat[np.arange(len(y)), y] + eps)))
        return float(np.mean((y_hat - y) ** 2))

    # ---------------------------------------------------------------- lan truyền ngược
    def backward(self, y):
        m = len(y)
        y_hat = self.cache[-1][1]
        if self.task == "multi":
            delta = y_hat.copy()
            delta[np.arange(m), y] -= 1
        else:
            delta = y_hat - y.reshape(-1, 1)
        if self.task == "reg":
            delta = 2 * delta                                    # đạo hàm của (y_hat - y)^2
        delta /= m
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            a_prev = self.X_in if i == 0 else self.cache[i - 1][1]
            grads_W[i] = a_prev.T @ delta if not sp.issparse(a_prev) else (a_prev.T @ delta)
            grads_b[i] = delta.sum(0, keepdims=True)
            if i > 0:
                z_prev, _, mask = self.cache[i - 1]
                delta = (delta @ self.W[i].T) * (z_prev > 0)    # quy tắc dây chuyền qua ReLU
                if mask is not None:
                    delta = delta * mask
        self._adam(grads_W + grads_b)

    def _adam(self, grads, b1=0.9, b2=0.999, eps=1e-8):
        self.t += 1
        params = self.W + self.b
        for k, (p, g) in enumerate(zip(params, grads)):
            g = np.asarray(g, dtype=np.float32)
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * g * g
            m_hat = self.m[k] / (1 - b1 ** self.t)
            v_hat = self.v[k] / (1 - b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)       # cập nhật tại chỗ

    # ---------------------------------------------------------------- vòng huấn luyện
    def fit(self, X, y, X_val, y_val, epochs=30, batch=256, patience=8, log=None):
        hist = {"loss": [], "val_loss": []}
        best, best_state, wait = np.inf, None, 0
        n = X.shape[0]
        for ep in range(epochs):
            order = self.rng.permutation(n)
            total = 0.0
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                xb = X[idx]
                xb = xb.toarray() if sp.issparse(xb) else xb
                y_hat = self.forward(xb, train=True)
                total += self.loss(y_hat, y[idx].reshape(y_hat.shape) if self.task != "multi" else y[idx]) * len(idx)
                self.backward(y[idx])
            val = self.evaluate_loss(X_val, y_val)
            hist["loss"].append(total / n)
            hist["val_loss"].append(val)
            if log:
                log(f"  epoch {ep + 1}: loss={total / n:.4f} val={val:.4f}")
            if val < best - 1e-5:
                best, wait = val, 0
                best_state = ([w.copy() for w in self.W], [b.copy() for b in self.b])
            else:
                wait += 1
                if wait >= patience:
                    break
        self.W, self.b = best_state
        return hist

    def predict_raw(self, X, batch=4096):
        outs = []
        for s in range(0, X.shape[0], batch):
            xb = X[s:s + batch]
            outs.append(self.forward(xb.toarray() if sp.issparse(xb) else xb))
        return np.vstack(outs)

    def evaluate_loss(self, X, y):
        y_hat = self.predict_raw(X)
        return self.loss(y_hat, y.reshape(y_hat.shape) if self.task != "multi" else y)

    def save(self, path):
        np.savez(path, **{f"W{i}": w for i, w in enumerate(self.W)}, **{f"b{i}": b for i, b in enumerate(self.b)})
