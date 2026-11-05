"""Mục 4.3.3: mạng hồi quy many-to-one bằng PyTorch, một lớp cho cả bốn loại tế bào.

Cấu hình chung: 2 tầng hồi quy, 64 đơn vị ẩn, dropout 0,2 giữa hai tầng, rồi Dense(32) - ReLU -
Dropout - Dense(đầu ra). BiLSTM ghép trạng thái ẩn hai chiều nên tầng Dense nhận 128 chiều.
"""
import numpy as np
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CELLS = ["rnn", "lstm", "gru", "bilstm"]
LABELS = {"rnn": "Simple RNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}


class SeqModel(nn.Module):
    def __init__(self, cell="lstm", input_dim=9, hidden=64, num_layers=2, out_dim=1, dropout=0.2):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU, "bilstm": nn.LSTM}[cell]
        kw = {"nonlinearity": "tanh"} if cell == "rnn" else {}
        self.rnn = rnn_cls(input_dim, hidden, num_layers=num_layers, batch_first=True, dropout=dropout,
                           bidirectional=cell == "bilstm", **kw)
        d = hidden * (2 if cell == "bilstm" else 1)
        self.fc = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(dropout), nn.Linear(32, out_dim))

    def forward(self, x):                      # x: [B, T, D]
        out, _ = self.rnn(x)                    # out: [B, T, H] (BiLSTM: [B, T, 2H])
        return self.fc(out[:, -1, :])           # many-to-one: chỉ dùng bước thời gian cuối


def train_epoch(model, X, y, criterion, opt, batch, max_grad_norm=1.0):
    model.train()
    order = torch.randperm(len(X), device=DEVICE)
    total = 0.0
    for s in range(0, len(X), batch):
        idx = order[s:s + batch]
        opt.zero_grad()
        loss = criterion(model(X[idx]), y[idx])
        loss.backward()
        # Cắt chuẩn gradient: giữ hướng, giới hạn độ dài bước cập nhật khi gradient bùng nổ qua BPTT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        opt.step()
        total += loss.item() * len(idx)
    return total / len(X)


@torch.no_grad()
def predict(model, X, batch=4096):
    model.eval()
    X = torch.as_tensor(X, device=DEVICE)
    return torch.cat([model(X[s:s + batch]) for s in range(0, len(X), batch)]).float().cpu().numpy().ravel()


def fit(model, X, y, Xv, yv, loss="mse", epochs=80, batch=64, lr=1e-3, patience=12, log=None):
    model.to(DEVICE)
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=DEVICE)  # noqa: E731
    Xt, yt, Xvt, yvt = T(X), T(y).view(-1, 1), T(Xv), T(yv).view(-1, 1)
    crit = nn.MSELoss() if loss == "mse" else nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, best, wait, state = {"loss": [], "val_loss": []}, np.inf, 0, None
    for ep in range(epochs):
        tr = train_epoch(model, Xt, yt, crit, opt, batch)
        model.eval()
        with torch.no_grad():
            val = sum(crit(model(Xvt[s:s + 4096]), yvt[s:s + 4096]).item() * len(yvt[s:s + 4096])
                      for s in range(0, len(Xvt), 4096)) / len(Xvt)
        hist["loss"].append(tr)
        hist["val_loss"].append(val)
        if log:
            log(f"  epoch {ep + 1}: loss={tr:.5f} val={val:.5f}")
        if val < best - 1e-7:
            best, wait = val, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    return hist
