"""Chương 4 — mô hình hồi quy chuỗi (SimpleRNN, LSTM, GRU, BiLSTM) và vòng huấn luyện PyTorch (GPU)."""
import copy
import time

import numpy as np
import torch
from torch import nn

ARCHS = ("rnn", "lstm", "gru", "bilstm")
ARCH_NAMES = {"rnn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}


class SeqRegressor(nn.Module):
    """Một lớp hồi tiếp (hidden=64) → Dropout → Linear(1): dự báo many-to-one từ trạng thái ẩn cuối."""

    def __init__(self, n_features, arch="lstm", hidden=64, dropout=0.2):
        super().__init__()
        cell = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU, "bilstm": nn.LSTM}[arch]
        kw = {"nonlinearity": "tanh"} if arch == "rnn" else {}
        self.rnn = cell(n_features, hidden, batch_first=True, bidirectional=arch == "bilstm", **kw)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden * (2 if arch == "bilstm" else 1), 1)

    def forward(self, x):                      # x: (batch, thời gian, đặc trưng)
        out, _ = self.rnn(x)                   # out: (batch, thời gian, hidden × số chiều)
        return self.head(self.drop(out[:, -1])).squeeze(-1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())


def train_epoch(model, loader, opt, loss_fn, clip=1.0):
    """Một epoch: lan truyền xuôi, MSE, lan truyền ngược qua thời gian (BPTT), cắt chuẩn gradient, Adam."""
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / n


@torch.no_grad()
def predict(model, X, device, batch=1024):
    model.eval()
    out = [model(torch.as_tensor(X[i:i + batch], device=device)).cpu().numpy() for i in range(0, len(X), batch)]
    return np.concatenate(out)


def fit(arch, data, device, epochs=150, patience=15, lr=1e-3, batch=64, hidden=64, seed=42, verbose=True):
    """Huấn luyện với dừng sớm theo validation loss, khôi phục trọng số tốt nhất; trả (model, history, giây)."""
    torch.manual_seed(seed)
    Xtr, ytr = data.part("train")
    Xva, yva = data.part("val")
    model = SeqRegressor(Xtr.shape[-1], arch, hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=6)
    loss_fn = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.as_tensor(Xtr, device=device), torch.as_tensor(ytr, device=device))
    g = torch.Generator(device="cpu").manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, generator=g)
    xv, yv = torch.as_tensor(Xva, device=device), torch.as_tensor(yva, device=device)
    hist = {"loss": [], "val_loss": []}
    best, best_state, bad = float("inf"), None, 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        tr = train_epoch(model, loader, opt, loss_fn)
        model.eval()
        with torch.no_grad():
            va = loss_fn(model(xv), yv).item()
        sched.step(va)
        hist["loss"].append(tr)
        hist["val_loss"].append(va)
        if va < best - 1e-6:
            best, best_state, bad = va, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    secs = time.perf_counter() - t0
    if verbose:
        print(f"  torch {arch}: {len(hist['loss'])} epoch, best val {best:.4f}, {secs:.1f}s", flush=True)
    return model, hist, secs
