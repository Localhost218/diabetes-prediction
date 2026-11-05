"""Chương 4 (mạch tiểu đường) — dự báo glucose CGM 30 phút tới bằng RNN, cài đặt bằng ba cách: NumPy, Keras, PyTorch.

Dữ liệu: ShanghaiT1DM/T2DM (Zhao và cộng sự, Scientific Data 2023), 112 bệnh nhân, cảm biến 15 phút/lần, đã giải nén
ở lab/data/shanghai (dùng lại pipeline của lab/ch4_rnn.py). Mỗi mẫu: 12 điểm gần nhất (3 giờ) × 3 đặc trưng
[glucose chuẩn hóa, sin(giờ), cos(giờ)] → glucose sau 30 phút. Chia theo BỆNH NHÂN 70/15/15, phân tầng T1/T2.

Mọi mô hình dự báo PHẦN CHÊNH so với điểm cuối (residual) trong thang chuẩn hóa — cùng mục tiêu cho ba framework,
nên chênh lệch kết quả chỉ đến từ cách cài đặt. Khung NumPy (RNNScratch, BPTT) kế thừa tl3.ch4_scratch (nhóm 07).
"""
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]                # MultiAgent/
SHANGHAI = ROOT / "lab" / "data" / "shanghai"
LOOKBACK, AHEAD, STEP_MIN = 12, 2, 15                     # 12 điểm vào, dự báo bước thứ 2 = 30 phút


# ------------------------------------------------------------------ dữ liệu
def load_long() -> pd.DataFrame:
    cache = SHANGHAI / "cgm_long.csv"
    if cache.exists():
        df = pd.read_csv(cache, parse_dates=["time"])
    else:
        rows = []
        for f in sorted(glob.glob(str(SHANGHAI / "extracted" / "raw" / "Shanghai_T*" / "*.xls*"))):
            if "~$" in f:
                continue
            d = pd.read_excel(f)
            col = next(c for c in d.columns if str(c).startswith("CGM"))
            name = Path(f).stem
            rows.append(pd.DataFrame({"patient": name.split("_")[0], "record": name,
                                      "type": "T1" if "T1DM" in f else "T2", "time": pd.to_datetime(d["Date"]),
                                      "cgm": pd.to_numeric(d[col], errors="coerce")}))
        df = pd.concat(rows, ignore_index=True).dropna(subset=["cgm"])
        df.to_csv(cache, index=False)
    return df.sort_values(["record", "time"]).reset_index(drop=True)


def split_patients(df, seed=0):
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for t in ["T1", "T2"]:
        pts = sorted(df[df.type == t].patient.unique())
        rng.shuffle(pts)
        n = max(2, round(0.15 * len(pts)))
        out["test"] += pts[:n]
        out["val"] += pts[n:2 * n]
        out["train"] += pts[2 * n:]
    return out


def windows(df, patients):
    """Cắt chuỗi tại khoảng trống > 20 phút rồi trượt cửa sổ; trả X (N,12) mg/dL, giờ của điểm cuối, y (N,) mg/dL."""
    X, H, Y = [], [], []
    for _, rec in df[df.patient.isin(patients)].groupby("record"):
        gap = rec.time.diff().dt.total_seconds().div(60).fillna(STEP_MIN) > 20
        for _, seg in rec.groupby(gap.cumsum()):
            v = seg.cgm.to_numpy(np.float32)
            hour = (seg.time.dt.hour + seg.time.dt.minute / 60).to_numpy(np.float32)
            for i in range(len(v) - LOOKBACK - AHEAD + 1):
                X.append(v[i:i + LOOKBACK])
                H.append(hour[i + LOOKBACK - 1])
                Y.append(v[i + LOOKBACK + AHEAD - 1])
    return np.array(X), np.array(H, np.float32), np.array(Y, np.float32)


class CGMData:
    """Ba tập (train/val/test) đã tạo đặc trưng; mu/sd lấy từ tập train."""

    def __init__(self, seed=0):
        df = load_long()
        self.split = split_patients(df, seed)
        raw = {s: windows(df, p) for s, p in self.split.items()}
        self.mu, self.sd = float(raw["train"][0].mean()), float(raw["train"][0].std())
        self.raw = raw
        self.n_patients = {s: len(p) for s, p in self.split.items()}

    def features(self, X, H):
        g = (X - self.mu) / self.sd
        ang = 2 * np.pi * H / 24
        tod = np.stack([np.sin(ang), np.cos(ang)], 1)[:, None, :].repeat(X.shape[1], 1)
        return np.concatenate([g[..., None], tod], -1).astype(np.float32)          # (N, 12, 3)

    def part(self, s):
        """(X đặc trưng, y = (glucose sau 30 phút − điểm cuối) / sd)."""
        X, H, Y = self.raw[s]
        return self.features(X, H), ((Y - X[:, -1]) / self.sd).astype(np.float32)

    def to_mgdl(self, s, pred_std):
        X, _, _ = self.raw[s]
        return X[:, -1] + np.asarray(pred_std).reshape(-1) * self.sd


# ------------------------------------------------------------------ đánh giá
def clarke_ab(ref, pred):
    """Tỷ lệ vùng A và A+B của lưới sai số Clarke (rút gọn theo định nghĩa gốc 1987)."""
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    a = (np.abs(pred - ref) <= 0.2 * ref) | ((ref < 70) & (pred < 70))
    e = ((ref <= 70) & (pred >= 180)) | ((ref >= 180) & (pred <= 70))
    c = ((ref >= 70) & (ref <= 290) & (pred >= ref + 110)) | ((ref >= 130) & (ref <= 180) & (pred <= 7 / 5 * ref - 182))
    d = ((ref >= 240) & (pred >= 70) & (pred <= 180)) | ((ref <= 175 / 3) & (pred >= 70) & (pred <= 180)) | \
        ((ref >= 175 / 3) & (ref <= 70) & (pred >= 6 / 5 * ref))
    b = ~(a | c | d | e)
    return float(a.mean()), float((a | b).mean())


def scores(ref, pred):
    ref, pred = np.asarray(ref, float), np.asarray(pred, float)
    za, zab = clarke_ab(ref, pred)
    return {"rmse": float(np.sqrt(np.mean((pred - ref) ** 2))), "mae": float(np.mean(np.abs(pred - ref))),
            "mard": float(np.mean(np.abs(pred - ref) / ref) * 100), "clarke_a": za, "clarke_ab": zab}


# ------------------------------------------------------------------ NumPy (kế thừa tl3.ch4_scratch)
def fit_numpy(data, hidden=16, n_train=12000, epochs=8, lr=0.01, seed=0, log=print):
    """RNN một lớp viết bằng NumPy, BPTT, SGD từng mẫu; huấn luyện trên tập con vì vòng lặp Python chậm."""
    from tl3.ch4_scratch import RNNScratch, sgd_train
    Xtr, ytr = data.part("train")
    Xva, yva = data.part("val")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))[:n_train]
    vidx = rng.permutation(len(Xva))[:3000]
    model = RNNScratch(3, hidden, seed=seed)
    t0 = time.perf_counter()
    hist = sgd_train(model, Xtr[idx].astype(float), ytr[idx].astype(float), Xva[vidx].astype(float),
                     yva[vidx].astype(float), epochs=epochs, lr=lr, seed=seed)
    secs = time.perf_counter() - t0
    log(f"  numpy rnn: {epochs} epoch trên {len(idx):,} mẫu, val {hist['val_loss'][-1]:.4f}, {secs:.0f}s")
    n_params = sum(v.size for v in model.p.values())
    return model, hist, secs, n_params, len(idx)


def predict_numpy(model, X):
    return np.array([model.forward(x.astype(float)) for x in X])


# ------------------------------------------------------------------ Keras
def build_keras(arch, hidden=64, layers_n=1, dropout=0.2, lr=2e-3):
    import keras
    from keras import layers
    cell = {"rnn": layers.SimpleRNN, "lstm": layers.LSTM, "gru": layers.GRU}[arch]
    stack = [keras.Input(shape=(LOOKBACK, 3))]
    for i in range(layers_n):
        stack.append(cell(hidden, return_sequences=i < layers_n - 1))
    stack += [layers.Dropout(dropout), layers.Dense(1)]
    model = keras.Sequential(stack, name=f"keras_{arch}{layers_n}")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0), loss="mse")
    return model


def fit_keras(data, arch, hidden=64, layers_n=1, epochs=40, patience=6, batch=512, seed=0, log=print):
    import keras
    keras.utils.set_random_seed(seed)
    Xtr, ytr = data.part("train")
    Xva, yva = data.part("val")
    model = build_keras(arch, hidden, layers_n)
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)]
    t0 = time.perf_counter()
    h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=epochs, batch_size=batch, callbacks=cbs, verbose=0)
    secs = time.perf_counter() - t0
    hist = {"loss": [float(v) for v in h.history["loss"]], "val_loss": [float(v) for v in h.history["val_loss"]]}
    log(f"  keras {arch}×{layers_n}: {len(hist['loss'])} epoch, best val {min(hist['val_loss']):.4f}, {secs:.0f}s")
    return model, hist, secs, int(model.count_params())


# ------------------------------------------------------------------ PyTorch
def torch_model(arch, hidden=64, layers_n=1, dropout=0.2):
    import torch.nn as nn

    class SeqReg(nn.Module):
        def __init__(self):
            super().__init__()
            rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[arch]
            self.rnn = rnn(3, hidden, num_layers=layers_n, batch_first=True, dropout=dropout if layers_n > 1 else 0.0)
            self.drop = nn.Dropout(dropout)
            self.head = nn.Linear(hidden, 1)

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(self.drop(out[:, -1])).squeeze(-1)
    return SeqReg()


def fit_torch(data, arch, hidden=64, layers_n=1, epochs=40, patience=6, batch=512, lr=2e-3, seed=0, log=print):
    import copy

    import torch
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch_model(arch, hidden, layers_n).to(dev)
    Xtr, ytr = (torch.tensor(a, device=dev) for a in data.part("train"))
    Xva, yva = (torch.tensor(a, device=dev) for a in data.part("val"))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    hist, best, wait = {"loss": [], "val_loss": []}, (np.inf, None), 0
    t0 = time.perf_counter()
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=dev)
        tot = 0.0
        for i in range(0, len(perm), batch):
            b = perm[i:i + batch]
            opt.zero_grad()
            loss = torch.nn.functional.mse_loss(model(Xtr[b]), ytr[b])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = torch.nn.functional.mse_loss(model(Xva), yva).item()
        sched.step(val)
        hist["loss"].append(tot / len(Xtr))
        hist["val_loss"].append(val)
        if val < best[0] - 1e-5:
            best, wait = (val, copy.deepcopy(model.state_dict())), 0
        else:
            wait += 1
            if wait >= patience:
                break
    secs = time.perf_counter() - t0
    model.load_state_dict(best[1])
    log(f"  torch {arch}×{layers_n}: {len(hist['loss'])} epoch, best val {best[0]:.4f}, {secs:.0f}s")
    return model, hist, secs, sum(p.numel() for p in model.parameters())


def predict_torch(model, X):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, device=dev)).cpu().numpy()
