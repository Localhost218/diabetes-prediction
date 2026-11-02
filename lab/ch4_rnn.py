"""
Chương 4 — RNN dự báo đường huyết liên tục (CGM) trên bộ ShanghaiT1DM / ShanghaiT2DM (Zhao et al., Scientific
Data 2023): 12 bệnh nhân ĐTĐ type 1 và 100 bệnh nhân type 2, cảm biến FreeStyle Libre, 15 phút/lần.

Bài toán: từ 3 giờ gần nhất (12 điểm) dự báo glucose sau 15, 30, 45, 60 phút (4 điểm).
Chia theo BỆNH NHÂN (70/15/15, phân tầng T1/T2) để không rò rỉ: mô hình phải tổng quát sang người chưa gặp.
  - Persistence (giữ nguyên giá trị cuối), AR tuyến tính (Ridge trên 12 điểm)
  - RNN đơn giản, LSTM, GRU, LSTM 2 tầng (PyTorch)
Đánh giá: RMSE, MAE, MARD (mg/dL, %) và tỷ lệ vùng A / A+B của lưới sai số Clarke.

    .venv/bin/python lab/ch4_rnn.py
"""

import copy
import glob
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge

from common import ACCENT, DATA, GREY, MODELS, PALETTE, plt, r4, save_fig, save_results, seed_everything

warnings.filterwarnings("ignore")
torch.set_num_threads(6)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # huấn luyện trên GPU nếu có
RAW = DATA / "shanghai" / "extracted" / "raw"
CACHE = DATA / "shanghai" / "cgm_long.csv"
LOOKBACK, HORIZON, STEP_MIN = 12, 4, 15


# ------------------------------------------------------------------ dữ liệu
def build_cache() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW / "Shanghai_T*" / "*.xls*"))):
        if "~$" in f:
            continue
        d = pd.read_excel(f)
        col = next(c for c in d.columns if str(c).startswith("CGM"))
        name = f.split("/")[-1].split(".")[0]
        pid, period = name.split("_")[0], name.split("_")[1]
        rows.append(pd.DataFrame({"patient": pid, "record": name, "type": "T1" if "T1DM" in f else "T2",
                                  "time": pd.to_datetime(d["Date"]), "cgm": pd.to_numeric(d[col], errors="coerce")}))
    df = pd.concat(rows, ignore_index=True).dropna(subset=["cgm"])
    df.to_csv(CACHE, index=False)
    return df


def load() -> pd.DataFrame:
    df = pd.read_csv(CACHE, parse_dates=["time"]) if CACHE.exists() else build_cache()
    return df.sort_values(["record", "time"]).reset_index(drop=True)


def segments(df_rec: pd.DataFrame):
    """Cắt chuỗi tại các khoảng trống > 20 phút (cảm biến mất tín hiệu) để cửa sổ không nối qua chỗ hở."""
    gap = df_rec.time.diff().dt.total_seconds().div(60).fillna(STEP_MIN) > 20
    for _, seg in df_rec.groupby(gap.cumsum()):
        yield seg.cgm.to_numpy(np.float32), seg.time.dt.hour.to_numpy() + seg.time.dt.minute.to_numpy() / 60


def windows(df: pd.DataFrame, records):
    X, Y, H = [], [], []
    for rec in records:
        for v, hour in segments(df[df.record == rec]):
            for i in range(len(v) - LOOKBACK - HORIZON + 1):
                X.append(v[i:i + LOOKBACK])
                Y.append(v[i + LOOKBACK:i + LOOKBACK + HORIZON])
                H.append(hour[i + LOOKBACK - 1])
    return np.array(X), np.array(Y), np.array(H, np.float32)


def split_patients(df, seed=0):
    rng = np.random.default_rng(seed)
    out = {"train": [], "val": [], "test": []}
    for t in ["T1", "T2"]:
        pts = sorted(df[df.type == t].patient.unique())
        rng.shuffle(pts)
        n_val = n_te = max(2, round(0.15 * len(pts)))
        out["test"] += pts[:n_te]
        out["val"] += pts[n_te:n_te + n_val]
        out["train"] += pts[n_te + n_val:]
    return out


# ------------------------------------------------------------------ mô hình
class SeqModel(nn.Module):
    """Đầu vào mỗi bước: [glucose chuẩn hóa, sin(giờ), cos(giờ)] → RNN/LSTM/GRU → trạng thái ẩn cuối → Linear 4."""

    def __init__(self, cell="lstm", hidden=64, layers=1):
        super().__init__()
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(3, hidden, num_layers=layers, batch_first=True, dropout=0.2 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, HORIZON)

    def forward(self, x):
        out, _ = self.rnn(x)
        # dự báo phần chênh so với điểm cuối (residual) — dễ học hơn dự báo giá trị tuyệt đối
        return x[:, -1:, 0] + self.head(out[:, -1])


def to_tensor(X, H, mu, sd):
    g = (X - mu) / sd
    ang = 2 * np.pi * H / 24
    tod = np.stack([np.sin(ang), np.cos(ang)], 1)[:, None, :].repeat(X.shape[1], 1)
    return torch.tensor(np.concatenate([g[..., None], tod], -1), dtype=torch.float32)


def train_seq(cell, layers, data, mu, sd, epochs=40):
    seed_everything(0)
    model = SeqModel(cell, 64, layers).to(DEVICE)
    Xtr, Ytr = to_tensor(*data["train"][::2], mu, sd).to(DEVICE), torch.tensor((data["train"][1] - mu) / sd).to(DEVICE)
    Xva, Yva = to_tensor(*data["val"][::2], mu, sd).to(DEVICE), torch.tensor((data["val"][1] - mu) / sd).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=3)
    best, history, wait = (np.inf, None), [], 0
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), device=DEVICE)
        tot = 0.0
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            opt.zero_grad()
            loss = nn.functional.mse_loss(model(Xtr[b]), Ytr[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # chặn bùng nổ gradient
            opt.step()
            tot += loss.item() * len(b)
        model.eval()
        with torch.no_grad():
            val = nn.functional.mse_loss(model(Xva), Yva).item()
        sched.step(val)
        history.append({"epoch": ep + 1, "train_mse": r4(tot / len(Xtr)), "val_mse": r4(val)})
        if val < best[0] - 1e-5:
            best, wait = (val, copy.deepcopy(model.state_dict())), 0
        else:
            wait += 1
            if wait >= 8:  # dừng sớm
                break
    model.load_state_dict(best[1])
    return model, history, time.perf_counter() - t0


def predict_seq(model, X, H, mu, sd):
    model.eval()
    with torch.no_grad():
        return model(to_tensor(X, H, mu, sd).to(DEVICE)).cpu().numpy() * sd + mu


# ------------------------------------------------------------------ đánh giá
def clarke_zones(ref, pred):
    """Phân vùng lưới sai số Clarke (1987). A: sai ≤ 20% hoặc cả hai < 70; E: sai lầm nguy hiểm ngược chiều."""
    zones = np.empty(len(ref), dtype="<U1")
    for i, (r, p) in enumerate(zip(ref, pred)):
        if (r <= 70 and p <= 70) or (0.8 * r <= p <= 1.2 * r):
            zones[i] = "A"
        elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
            zones[i] = "E"
        elif (70 <= r <= 290 and p >= r + 110) or (130 <= r <= 180 and p <= (7 / 5) * r - 182):
            zones[i] = "C"
        elif (r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180) or \
                (175 / 3 <= r <= 70 and p >= (6 / 5) * r):
            zones[i] = "D"
        else:
            zones[i] = "B"
    return zones


def scores(Y, P):
    out = {}
    for h in range(HORIZON):
        y, p = Y[:, h], P[:, h]
        z = clarke_zones(y, p)
        out[f"{(h + 1) * STEP_MIN}min"] = {
            "rmse": r4(np.sqrt(np.mean((y - p) ** 2))), "mae": r4(np.mean(np.abs(y - p))),
            "mard": r4(np.mean(np.abs(y - p) / y) * 100), "clarke_A": r4((z == "A").mean() * 100),
            "clarke_AB": r4(np.isin(z, ["A", "B"]).mean() * 100)}
    return out


def gradient_through_time(data, mu, sd, models):
    """|∂ŷ/∂x_t| theo vị trí t trong cửa sổ 48 điểm — minh họa biến mất gradient của RNN đơn giản."""
    X, _, H = data["val"]
    out = {}
    for key, model in models.items():
        x = to_tensor(X[:256], H[:256], mu, sd).repeat(1, 4, 1).to(DEVICE)  # nối 4 lần → chuỗi 48 bước
        x.requires_grad_(True)
        model.train(False)
        torch.backends.cudnn.enabled = False
        model(x)[:, 0].sum().backward()
        g = x.grad[..., 0].abs().mean(0).cpu().numpy()
        out[key] = (g / g.max()).round(5).tolist()
    return out


# ------------------------------------------------------------------ hình
def plot_data(df, split, data):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3))
    for t, c in [("T1", PALETTE[1]), ("T2", ACCENT)]:
        axes[0].hist(df[df.type == t].cgm, bins=60, density=True, alpha=0.55, color=c, label=f"ĐTĐ type {t[1]}")
    for v in (70, 180):
        axes[0].axvline(v, ls="--", color=GREY, lw=0.8)
    axes[0].set(title="Phân bố glucose CGM (mg/dL)", xlim=(30, 400))
    axes[0].legend()
    hour = df.time.dt.hour
    prof = df.groupby([hour, "type"]).cgm.median().unstack()
    axes[1].plot(prof.index, prof["T1"], color=PALETTE[1], label="type 1")
    axes[1].plot(prof.index, prof["T2"], color=ACCENT, label="type 2")
    axes[1].set(title="Trung vị glucose theo giờ trong ngày", xlabel="giờ", xticks=range(0, 24, 3))
    axes[1].legend()
    rec = df[df.record == df[df.type == "T1"].record.iloc[0]].iloc[:4 * 96]
    axes[2].plot(rec.time, rec.cgm, color=PALETTE[1], lw=1)
    axes[2].axhspan(70, 180, color="#dcfce7", alpha=0.6, label="vùng mục tiêu 70–180")
    axes[2].set(title=f"Một bản ghi type 1 ({rec.record.iloc[0]}), 4 ngày")
    axes[2].tick_params(axis="x", labelrotation=30, labelsize=7)
    axes[2].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "ch4_data")


def plot_history(histories, names):
    fig, ax = plt.subplots(figsize=(6, 2.8))
    for i, (k, h) in enumerate(histories.items()):
        ax.plot([r["epoch"] for r in h], [r["val_mse"] for r in h], color=PALETTE[i], label=names[k])
    ax.set(title="MSE trên validation (glucose chuẩn hóa)", xlabel="epoch", yscale="log")
    ax.legend(fontsize=8)
    save_fig(fig, "ch4_history")


def plot_forecast(Y, preds, names, data):
    _, _, H = data["test"]
    seg = slice(300, 300 + 96)  # 24 giờ liên tục của tập kiểm tra
    fig, ax = plt.subplots(figsize=(10, 3))
    t = np.arange(96) * STEP_MIN / 60
    ax.plot(t, Y[seg, 1], color="black", lw=1.4, label="thật (sau 30 phút)")
    for i, k in enumerate(["persistence", "ridge", "lstm"]):
        ax.plot(t, preds[k][seg, 1], lw=1, color=[GREY, PALETTE[1], ACCENT][i], label=names[k],
                ls="--" if k == "persistence" else "-")
    ax.axhspan(70, 180, color="#dcfce7", alpha=0.5)
    ax.set(title="Dự báo trước 30 phút trên một bệnh nhân của tập kiểm tra (24 giờ)", xlabel="giờ",
           ylabel="mg/dL")
    ax.legend(fontsize=8, ncol=4)
    save_fig(fig, "ch4_forecast")


def plot_rmse_clarke(res, names, Y, P):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    hs = [15, 30, 45, 60]
    for i, k in enumerate(res):
        axes[0].plot(hs, [res[k][f"{h}min"]["rmse"] for h in hs], "o-", color=PALETTE[i], label=names[k])
    axes[0].set(title="RMSE theo tầm dự báo (tập kiểm tra)", xlabel="phút", ylabel="mg/dL", xticks=hs)
    axes[0].legend(fontsize=7.5)
    ax = axes[1]
    z = clarke_zones(Y[:, 1], P[:, 1])
    colors = {"A": "#16a34a", "B": ACCENT, "C": "#ca8a04", "D": "#dc2626", "E": "#7f1d1d"}
    for zone, c in colors.items():
        m = z == zone
        ax.scatter(Y[m, 1], P[m, 1], s=2, color=c, alpha=0.4, label=f"{zone}: {m.mean():.1%}")
    ax.plot([0, 400], [0, 400], color="black", lw=0.6)
    ax.plot([0, 400], [0, 480], "--", color=GREY, lw=0.6)
    ax.plot([0, 400], [0, 320], "--", color=GREY, lw=0.6)
    ax.set(xlim=(0, 400), ylim=(0, 400), xlabel="tham chiếu (mg/dL)", ylabel="dự báo (mg/dL)",
           title="Lưới Clarke — LSTM, tầm 30 phút")
    ax.legend(markerscale=5, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    save_fig(fig, "ch4_rmse_clarke")


def plot_grad(grads, names):
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    for i, (k, g) in enumerate(grads.items()):
        ax.semilogy(np.arange(len(g)) - len(g) + 1, g, color=PALETTE[i + 2], label=names[k])
    ax.set(title="Độ lớn gradient |∂ŷ/∂x_t| theo khoảng cách tới bước cuối (chuẩn hóa)",
           xlabel="bước thời gian t (0 = điểm gần nhất)")
    ax.legend(fontsize=8)
    save_fig(fig, "ch4_gradient_time")


def main():
    seed_everything()
    df = load()
    df = df[(df.cgm >= 20) & (df.cgm <= 500)]
    split = split_patients(df)
    recs = {s: sorted(df[df.patient.isin(p)].record.unique()) for s, p in split.items()}
    data = {s: windows(df, r) for s, r in recs.items()}
    mu, sd = float(data["train"][0].mean()), float(data["train"][0].std())
    for s in data:
        print(f"  {s}: {len(split[s])} bệnh nhân, {len(recs[s])} bản ghi, {len(data[s][0]):,} cửa sổ")

    names = {"persistence": "Persistence", "ridge": "AR tuyến tính (Ridge)", "rnn": "RNN đơn giản",
             "lstm": "LSTM", "gru": "GRU", "lstm2": "LSTM 2 tầng"}
    Xte, Yte, Hte = data["test"]
    preds, res, histories, meta, trained = {}, {}, {}, {}, {}
    preds["persistence"] = np.repeat(Xte[:, -1:], HORIZON, 1)
    ridge = Ridge(alpha=1.0).fit(data["train"][0], data["train"][1] - data["train"][0][:, -1:])
    preds["ridge"] = ridge.predict(Xte) + Xte[:, -1:]
    for key, (cell, layers) in {"rnn": ("rnn", 1), "lstm": ("lstm", 1), "gru": ("gru", 1),
                                "lstm2": ("lstm", 2)}.items():
        model, hist, secs = train_seq(cell, layers, data, mu, sd)
        trained[key], histories[key] = model, hist
        preds[key] = predict_seq(model, Xte, Hte, mu, sd)
        meta[key] = {"params": sum(p.numel() for p in model.parameters()), "epochs_run": len(hist),
                     "train_s": round(secs, 1)}
        print(f"  {names[key]}: {meta[key]}")
    for k in names:
        res[k] = scores(Yte, preds[k])
        print(f"  {names[k]:24s} 30' RMSE={res[k]['30min']['rmse']:.2f} MARD={res[k]['30min']['mard']:.2f}% "
              f"A={res[k]['30min']['clarke_A']:.1f}%  60' RMSE={res[k]['60min']['rmse']:.2f}")
    # theo loại ĐTĐ trên tập kiểm tra
    by_type = {}
    for t in ["T1", "T2"]:
        r = [x for x in recs["test"] if df[df.record == x].type.iloc[0] == t]
        Xt, Yt, Ht = windows(df, r)
        by_type[t] = {"n_windows": len(Xt),
                      "persistence": scores(Yt, np.repeat(Xt[:, -1:], HORIZON, 1))["30min"],
                      "lstm": scores(Yt, predict_seq(trained["lstm"], Xt, Ht, mu, sd))["30min"]}
    grads = gradient_through_time(data, mu, sd, {k: trained[k] for k in ["rnn", "lstm", "gru"]})

    plot_data(df, split, data)
    plot_history(histories, names)
    plot_forecast(Yte, preds, names, data)
    plot_rmse_clarke(res, names, Yte, preds["lstm"])
    plot_grad(grads, names)

    best = min(["rnn", "lstm", "gru", "lstm2"], key=lambda k: min(h["val_mse"] for h in histories[k]))
    torch.save({"state": {k: v.cpu() for k, v in trained[best].state_dict().items()}, "cell": {"lstm2": "lstm"}.get(best, best),
                "layers": 2 if best == "lstm2" else 1, "mu": mu, "sd": sd, "lookback": LOOKBACK,
                "horizon": HORIZON}, MODELS / "ch4_glucose_rnn.pt")
    per_patient = df.groupby(["type", "patient"]).size()
    tir = {t: {"TIR": r4(((v >= 70) & (v <= 180)).mean() * 100), "TBR": r4((v < 70).mean() * 100),
               "TAR": r4((v > 180).mean() * 100), "mean": r4(v.mean()), "cv": r4(v.std() / v.mean() * 100)}
           for t, v in df.groupby("type").cgm}
    save_results("ch4", {
        "dataset": {"name": "ShanghaiT1DM + ShanghaiT2DM", "n_records": int(df.record.nunique()),
                    "n_points": len(df), "patients": df.groupby("type").patient.nunique().to_dict(),
                    "points_per_patient": {"min": int(per_patient.min()), "median": int(per_patient.median()),
                                           "max": int(per_patient.max())},
                    "glycemic": tir, "split_patients": {s: len(p) for s, p in split.items()},
                    "windows": {s: len(d[0]) for s, d in data.items()}, "mu": r4(mu), "sd": r4(sd),
                    "head": df.head(6).astype(str).to_dict("records")},
        "results": res, "meta": meta, "by_type_30min": by_type, "deploy_key": best, "history": histories,
        "grad_time": {k: [g[0], g[-12], g[-1]] for k, g in grads.items()},
    })


if __name__ == "__main__":
    main()
