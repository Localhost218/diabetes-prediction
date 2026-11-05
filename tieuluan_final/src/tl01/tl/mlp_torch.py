"""Mục 2.4.4: mạng MLP bằng PyTorch, huấn luyện trên GPU.

Toàn bộ tập dữ liệu (đã tiền xử lý) được đưa lên GPU một lần rồi chia lô bằng chỉ số, tránh chi
phí chép từng lô qua PCIe của DataLoader với dữ liệu bảng nhỏ.
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TorchMLP(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, output_dim))   # logit / giá trị thô; sigmoid, softmax nằm trong hàm mất mát
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


LOSSES = {"binary": nn.BCEWithLogitsLoss, "multi": nn.CrossEntropyLoss, "reg": nn.MSELoss}


def _tensor(X):
    X = X.toarray() if sp.issparse(X) else X
    return torch.as_tensor(np.asarray(X, np.float32), device=DEVICE)


def _target(y, task):
    t = torch.as_tensor(y, device=DEVICE)
    return t.long() if task == "multi" else t.float().view(-1, 1)


def train_epoch(model, X, y, criterion, optimizer, batch=256):
    model.train()
    order = torch.randperm(len(X), device=DEVICE)
    total = 0.0
    for s in range(0, len(X), batch):
        idx = order[s:s + batch]
        optimizer.zero_grad()
        loss = criterion(model(X[idx]), y[idx])
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    return total / len(X)


@torch.no_grad()
def predict_raw(model, X, batch=8192):
    model.eval()
    X = X if torch.is_tensor(X) else _tensor(X)
    return torch.cat([model(X[s:s + batch]) for s in range(0, len(X), batch)]).cpu().numpy()


def fit(model, X, y, X_val, y_val, task, epochs=30, batch=256, lr=1e-3, patience=8, log=None):
    model.to(DEVICE)
    Xt, yt, Xv, yv = _tensor(X), _target(y, task), _tensor(X_val), _target(y_val, task)
    crit = LOSSES[task]()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    hist, best, wait, state = {"loss": [], "val_loss": []}, np.inf, 0, None
    for ep in range(epochs):
        tr = train_epoch(model, Xt, yt, crit, opt, batch)
        model.eval()
        with torch.no_grad():
            val = crit(model(Xv), yv).item()
        hist["loss"].append(tr)
        hist["val_loss"].append(val)
        if log:
            log(f"  epoch {ep + 1}: loss={tr:.4f} val={val:.4f}")
        if val < best - 1e-5:
            best, wait = val, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(state)
    return hist
