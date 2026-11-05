"""Chương 2 — mạng nơ-ron truyền thẳng (MLP) bằng PyTorch cho dữ liệu bảng, chạy GPU (phần VRAM nhỏ)."""
import copy
import time

import numpy as np
import torch
from torch import nn

from .common import SEED, device


class TabularMLP(nn.Module):
    """MLP: [Linear -> BatchNorm -> ReLU -> Dropout] x len(hidden) -> Linear(1).
    Đầu ra là logit (phân loại nhị phân) hoặc giá trị chuẩn hóa (hồi quy)."""

    def __init__(self, n_in, hidden=(128, 64, 32), dropout=0.2):
        super().__init__()
        layers, d = [], n_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_epoch(model, loader, loss_fn, opt, dev):
    """Một epoch: lan truyền thuận, tính mất mát, lan truyền ngược, cập nhật Adam; trả mất mát trung bình."""
    model.train()
    total, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(dev), yb.to(dev)
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        total += loss.item() * len(xb)
        n += len(xb)
    return total / n


@torch.no_grad()
def predict_raw(model, X, dev, bs=8192):
    model.eval()
    out = [model(torch.as_tensor(X[i:i + bs], dtype=torch.float32, device=dev)).cpu().numpy()
           for i in range(0, len(X), bs)]
    return np.concatenate(out)


def fit_mlp(X_tr, y_tr, X_va, y_va, task="clf", hidden=(128, 64, 32), dropout=0.2, lr=1e-3, wd=1e-4,
            epochs=60, bs=512, patience=8, pos_weight=None):
    """Huấn luyện với dừng sớm theo mất mát Validation; trả (mô hình tốt nhất, lịch sử từng epoch).
    task='clf': BCEWithLogitsLoss (pos_weight để bù mất cân bằng); task='reg': MSELoss trên y đã chuẩn hóa."""
    torch.manual_seed(SEED)
    dev = device()
    model = TabularMLP(X_tr.shape[1], hidden, dropout).to(dev)
    if task == "clf":
        pw = None if pos_weight is None else torch.tensor(pos_weight, device=dev)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    ds = torch.utils.data.TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32),
                                        torch.as_tensor(y_tr, dtype=torch.float32))
    loader = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True)
    Xv = torch.as_tensor(X_va, dtype=torch.float32, device=dev)
    yv = torch.as_tensor(y_va, dtype=torch.float32, device=dev)
    hist, best, best_state, wait, t0 = [], np.inf, None, 0, time.perf_counter()
    for ep in range(1, epochs + 1):
        tr = train_epoch(model, loader, loss_fn, opt, dev)
        model.eval()
        with torch.no_grad():
            va = loss_fn(model(Xv), yv).item()
        sched.step(va)
        hist.append({"epoch": ep, "train_loss": tr, "val_loss": va})
        if va < best - 1e-5:
            best, best_state, wait = va, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_loss": best, "epochs_run": len(hist),
                   "train_s": time.perf_counter() - t0, "n_params": sum(p.numel() for p in model.parameters())}


def proba(model, X):
    """Xác suất lớp dương = sigmoid(logit)."""
    return 1 / (1 + np.exp(-predict_raw(model, X, device())))
