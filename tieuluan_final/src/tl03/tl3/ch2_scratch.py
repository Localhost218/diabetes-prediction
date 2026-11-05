"""Chương 2 — cài đặt từ đầu bằng NumPy các thuật toán nền tảng để đối chiếu với scikit-learn."""
import numpy as np


def pla(X, y, max_epochs=100):
    """Perceptron Learning Algorithm (Rosenblatt): y ∈ {-1, +1}; gặp điểm phân loại sai thì w ← w + y·x.
    Trả w (đã gộp bias) và lịch sử số điểm sai mỗi epoch."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    errors = []
    for _ in range(max_epochs):
        wrong = 0
        for xi, yi in zip(Xb, y):
            if yi * (xi @ w) <= 0:
                w += yi * xi
                wrong += 1
        errors.append(wrong)
        if wrong == 0:
            break
    return w, errors


def linreg_normal_equation(X, y):
    """Bình phương tối thiểu dạng đóng: w = (XᵀX)⁻¹Xᵀy (dùng lstsq cho ổn định số)."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    return np.linalg.lstsq(Xb, y, rcond=None)[0]


def linreg_gd(X, y, lr=0.05, epochs=500):
    """Hồi quy tuyến tính bằng Gradient Descent trên MSE: ∇w = (2/n)·Xᵀ(Xw − y)."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    hist = []
    for _ in range(epochs):
        r = Xb @ w - y
        w -= lr * 2 / len(y) * (Xb.T @ r)
        hist.append(float(np.mean(r ** 2)))
    return w, hist


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def logreg_gd(X, y, lr=0.5, epochs=400, l2=1e-4, batch=4096, seed=0):
    """Hồi quy logistic bằng mini-batch Gradient Descent trên Binary Cross-Entropy + L2:
    ∇w = Xᵀ(σ(Xw) − y)/m + λw. Trả w và lịch sử mất mát toàn tập."""
    rng = np.random.default_rng(seed)
    Xb = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(Xb.shape[1])
    hist = []
    for _ in range(epochs):
        idx = rng.permutation(len(y))
        for s in range(0, len(y), batch):
            b = idx[s:s + batch]
            g = Xb[b].T @ (sigmoid(Xb[b] @ w) - y[b]) / len(b) + l2 * np.r_[0, w[1:]]
            w -= lr * g
        p = sigmoid(Xb @ w)
        hist.append(float(-np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))))
    return w, hist


def logreg_predict(w, X):
    return sigmoid(np.hstack([np.ones((len(X), 1)), X]) @ w)


def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(1, keepdims=True)


def softmax_gd(X, y, k, lr=0.5, epochs=300):
    """Softmax Regression K lớp: P(y=j|x) = exp(w_jᵀx)/Σ exp(w_iᵀx); gradient Xᵀ(P − Y_onehot)/n."""
    Xb = np.hstack([np.ones((len(X), 1)), X])
    W = np.zeros((Xb.shape[1], k))
    Y = np.eye(k)[y]
    for _ in range(epochs):
        W -= lr * Xb.T @ (softmax(Xb @ W) - Y) / len(y)
    return W


def softmax_predict(W, X):
    return softmax(np.hstack([np.ones((len(X), 1)), X]) @ W).argmax(1)
