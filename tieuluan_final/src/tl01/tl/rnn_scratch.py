"""Mục 4.3.1: tế bào LSTM viết tay bằng NumPy, xếp chồng nhiều tầng như nn.LSTM.

Thứ tự bốn khối trọng số theo PyTorch là (i, f, g, o): cổng vào, cổng quên, ứng viên ô nhớ, cổng ra.
Nạp trọng số của mô hình PyTorch đã huấn luyện vào đây rồi so sánh đầu ra là cách kiểm chứng
rằng các công thức (4.6)-(4.11) được hiện thực đúng.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class LSTMCellScratch:
    def __init__(self, input_dim, hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        k = 1 / np.sqrt(hidden_dim)
        self.H = hidden_dim
        self.W_ih = rng.uniform(-k, k, (4 * hidden_dim, input_dim))   # trọng số đầu vào của 4 khối
        self.W_hh = rng.uniform(-k, k, (4 * hidden_dim, hidden_dim))  # trọng số hồi quy của 4 khối
        self.b = np.zeros(4 * hidden_dim)

    def step(self, x_t, h_prev, c_prev):
        """x_t: [B, D], h_prev, c_prev: [B, H] -> (h_t, c_t)."""
        z = x_t @ self.W_ih.T + h_prev @ self.W_hh.T + self.b         # [B, 4H]
        H = self.H
        i = sigmoid(z[:, 0:H])                                        # cổng vào
        f = sigmoid(z[:, H:2 * H])                                    # cổng quên
        g = np.tanh(z[:, 2 * H:3 * H])                                # ứng viên ô nhớ
        o = sigmoid(z[:, 3 * H:4 * H])                                # cổng ra
        c_t = f * c_prev + i * g                                      # cập nhật cộng: đường cao tốc gradient
        h_t = o * np.tanh(c_t)
        return h_t, c_t

    def forward(self, X):
        """X: [B, T, D] -> chuỗi trạng thái ẩn [B, T, H]."""
        B, T, _ = X.shape
        h = np.zeros((B, self.H))
        c = np.zeros((B, self.H))
        out = np.empty((B, T, self.H))
        for t in range(T):
            h, c = self.step(X[:, t, :], h, c)
            out[:, t] = h
        return out


class LSTMRegressorScratch:
    """Bản NumPy của SeqModel('lstm'): các tầng LSTM xếp chồng + Dense(32) - ReLU - Dense(1)."""

    def __init__(self, cells, W1, b1, W2, b2):
        self.cells, self.W1, self.b1, self.W2, self.b2 = cells, W1, b1, W2, b2

    @classmethod
    def from_torch(cls, model):
        sd = {k: v.detach().cpu().double().numpy() for k, v in model.state_dict().items()}
        cells, layer = [], 0
        while f"rnn.weight_ih_l{layer}" in sd:
            W_ih = sd[f"rnn.weight_ih_l{layer}"]
            cell = LSTMCellScratch(W_ih.shape[1], W_ih.shape[0] // 4)
            cell.W_ih, cell.W_hh = W_ih, sd[f"rnn.weight_hh_l{layer}"]
            cell.b = sd[f"rnn.bias_ih_l{layer}"] + sd[f"rnn.bias_hh_l{layer}"]
            cells.append(cell)
            layer += 1
        return cls(cells, sd["fc.0.weight"], sd["fc.0.bias"], sd["fc.3.weight"], sd["fc.3.bias"])

    def predict(self, X):
        h = np.asarray(X, np.float64)
        for cell in self.cells:
            h = cell.forward(h)
        last = h[:, -1, :]
        z = np.maximum(last @ self.W1.T + self.b1, 0)
        return (z @ self.W2.T + self.b2).ravel()
