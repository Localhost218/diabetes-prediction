"""Huấn luyện BiLSTM (và LSTM, Ridge đối chứng) dự báo lợi suất phiên kế tiếp cho FPT, HPG, VCB, VIC và vàng quy đổi.

Chia theo thời gian: huấn luyện trước 2024, kiểm định năm 2024, kiểm thử từ 2025 tới hết dữ liệu.
Mốc so sánh quan trọng nhất là dự báo ngây thơ "giá không đổi" (lợi suất 0): với chuỗi giá gần như bước
ngẫu nhiên, vượt được mốc này là khó, và báo cáo ghi đúng kết quả thay vì chỉ đưa R² của giá.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2.common import MODELS, device, save_json, seed_all  # noqa: E402
from tl2.data import forecast_features, gold_frame, make_windows, market_frame  # noqa: E402
from tl2.dl_models import SeqForecaster  # noqa: E402

TARGETS = ["FPT", "HPG", "VCB", "VIC", "GOLD_VND"]
WINDOW, HIDDEN, LAYERS, DROPOUT, EPOCHS, BATCH, LR, PATIENCE = 30, 64, 2, 0.2, 80, 128, 1e-3, 10
VAL_START, TEST_START = "2024-01-01", "2025-01-01"


def evaluate(r_true, r_pred, p_prev):
    p_true, p_hat = p_prev * np.exp(r_true), p_prev * np.exp(r_pred)
    rmse = float(np.sqrt(np.mean((r_true - r_pred) ** 2)))
    rmse_naive = float(np.sqrt(np.mean(r_true ** 2)))
    nz = r_true != 0
    return {"rmse_ret": rmse, "rmse_naive": rmse_naive, "rmse_ratio": rmse / rmse_naive,
            "mae_price": float(np.mean(np.abs(p_true - p_hat))), "mae_price_naive": float(np.mean(np.abs(p_true - p_prev))),
            "dir_acc": float(np.mean(np.sign(r_pred[nz]) == np.sign(r_true[nz]))),
            "r2_ret": float(1 - np.sum((r_true - r_pred) ** 2) / np.sum((r_true - r_true.mean()) ** 2))}


def train_rnn(Xtr, ytr, Xva, yva, bidirectional, dev):
    seed_all()
    m = SeqForecaster(Xtr.shape[2], HIDDEN, LAYERS, DROPOUT, bidirectional).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = nn.HuberLoss(delta=1.0)
    T = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
    xtr, ytr_t, xva, yva_t = T(Xtr), T(ytr), T(Xva), T(yva)
    best, state, wait, hist = 1e9, None, 0, []
    for ep in range(1, EPOCHS + 1):
        m.train()
        perm = torch.randperm(len(xtr), device=dev)
        tot = 0.0
        for i in range(0, len(xtr), BATCH):
            b = perm[i:i + BATCH]
            opt.zero_grad()
            loss = loss_fn(m(xtr[b]), ytr_t[b])
            loss.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b)
        m.eval()
        with torch.no_grad():
            v = loss_fn(m(xva), yva_t).item()
        hist.append({"epoch": ep, "train": tot / len(xtr), "val": v})
        if v < best - 1e-6:
            best, wait = v, 0
            state = {k: x.detach().clone() for k, x in m.state_dict().items()}
        else:
            wait += 1
            if wait >= PATIENCE:
                break
    m.load_state_dict(state)
    m.eval()
    return m, hist


def main():
    dev = device()
    mk, gold = market_frame(), gold_frame()
    out = {"config": {"window": WINDOW, "hidden": HIDDEN, "layers": LAYERS, "dropout": DROPOUT, "val_start": VAL_START,
                      "test_start": TEST_START}}
    for target in TARGETS:
        F = forecast_features(gold if target == "GOLD_VND" else mk, target)
        cols = [c for c in F.columns if c not in ("y", "price")]
        tr = F.index < VAL_START
        mean, std = F.loc[tr, cols].mean().to_numpy(), F.loc[tr, cols].std().to_numpy() + 1e-8
        y_std = float(F.loc[tr, "y"].std())
        Fn = F.copy()
        Fn[cols] = (F[cols] - mean) / std
        Fn["y"] = F["y"] / y_std
        X, y, idx = make_windows(Fn, cols, WINDOW)
        price = F["price"].reindex(idx).to_numpy()
        r_true = y * y_std
        s_tr, s_va, s_te = idx < VAL_START, (idx >= VAL_START) & (idx < TEST_START), idx >= TEST_START
        res, preds = {}, {}
        for name, bi in (("bilstm", True), ("lstm", False)):
            t0 = time.perf_counter()
            m, hist = train_rnn(X[s_tr], y[s_tr], X[s_va], y[s_va], bi, dev)
            with torch.no_grad():
                p = m(torch.tensor(X[s_te], device=dev)).cpu().numpy() * y_std
            res[name] = evaluate(r_true[s_te], p, price[s_te]) | {"train_s": time.perf_counter() - t0,
                                                                   "epochs": len(hist)}
            res[name]["history"] = hist
            preds[name] = p
            if bi:
                MODELS.mkdir(exist_ok=True)
                torch.save(m.state_dict(), MODELS / f"forecast_{target}.pt")
                (MODELS / f"forecast_{target}.json").write_text(json.dumps(
                    {"cols": cols, "mean": mean.tolist(), "std": std.tolist(), "y_std": y_std, "window": WINDOW,
                     "hidden": HIDDEN, "layers": LAYERS, "dropout": DROPOUT,
                     "test": {k: v for k, v in res[name].items() if k != "history"}}, indent=1), encoding="utf-8")
        rg = Ridge(alpha=10.0).fit(X[s_tr].reshape(s_tr.sum(), -1), y[s_tr])
        p = rg.predict(X[s_te].reshape(s_te.sum(), -1)) * y_std
        res["ridge"] = evaluate(r_true[s_te], p, price[s_te])
        preds["ridge"] = p
        res["naive"] = evaluate(r_true[s_te], np.zeros(s_te.sum()), price[s_te])
        p_true = price[s_te] * np.exp(r_true[s_te])
        res["test_series"] = {"date": [d.strftime("%Y-%m-%d") for d in idx[s_te]], "true": p_true.tolist(),
                              "bilstm": (price[s_te] * np.exp(preds["bilstm"])).tolist(),
                              "r_true": r_true[s_te].tolist(), "r_bilstm": preds["bilstm"].tolist()}
        res["n"] = {"train": int(s_tr.sum()), "val": int(s_va.sum()), "test": int(s_te.sum()), "features": len(cols)}
        out[target] = res
        print(target, {k: round(res[k]["rmse_ratio"], 4) for k in ("bilstm", "lstm", "ridge")},
              "dir", round(res["bilstm"]["dir_acc"], 3), flush=True)
    save_json("forecast", out)


if __name__ == "__main__":
    main()
