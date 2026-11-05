"""Huấn luyện SimpleRNN / LSTM / GRU / BiLSTM cho ba thực nghiệm của Chương 4 trên một framework.

    python scripts/ch4_train.py --fw torch     # GPU (nhẹ, giới hạn 15 % VRAM)
    python scripts/ch4_train.py --fw keras     # CPU, 6 luồng

Ghi: outputs/metrics/ch4_<exp>_<fw>.json (chỉ số + lịch sử loss), outputs/preds/ch4_<exp>_<fw>.npz (dự báo test,
đơn vị gốc), models/ch4/<exp>_<fw>_<arch>.{pt,keras}.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import OUT, limit_threads, model_dir, save_json, seed_all  # noqa: E402

EXPS = ("ecommerce", "aapl", "gold")


def evaluate(data, pred_std):
    """Đổi dự báo chuẩn hoá về đơn vị gốc cho validation và test rồi tính chỉ số; trả (chỉ số, dự báo test)."""
    from tl3.ch4_metrics import forecast_metrics
    out, test_pred = {}, None
    n_val = len(data.part("val")[0])
    for part, p in (("val", pred_std[:n_val]), ("test", pred_std[n_val:])):
        yhat = data.to_original(p, part)
        out[part] = forecast_metrics(data.truth(part), yhat, data.naive(part))
        if part == "test":
            test_pred = yhat
    return out, test_pred


def run(fw, exps):
    from tl3.ch4_data import Prepared, experiment
    from tl3.ch4_metrics import forecast_metrics
    from tl3.ch4_torch import ARCHS
    preds_dir = OUT / "preds"
    preds_dir.mkdir(exist_ok=True)
    mdir = model_dir("ch4")
    for name in exps:
        seed_all()
        data = Prepared(experiment(name))
        res = {"config": {"window": data.window, "shapes": data.shapes(), "mode": data.mode,
                          "features": len(data.cols), "fw": fw},
               "naive": {p: forecast_metrics(data.truth(p), data.naive(p), data.naive(p)) for p in ("val", "test")},
               "history": {}}
        store = {"y_true": data.truth("test"), "naive": data.naive("test"),
                 "dates": np.array([str(d.date()) for d in data.dates[data.sl["test"]]])}
        Xva, _ = data.part("val")
        Xte, _ = data.part("test")
        Xeval = np.concatenate([Xva, Xte])
        print(f"[{fw}] {name}: {data.shapes()}", flush=True)
        for arch in ARCHS:
            if fw == "torch":
                import torch
                from tl3.ch4_torch import fit, n_params, predict
                from tl3.common import device
                dev = device()
                model, hist, secs = fit(arch, data, dev)
                p = predict(model, Xeval, dev)
                torch.save(model.state_dict(), mdir / f"{name}_torch_{arch}.pt")
                params = n_params(model)
            else:
                from tl3.ch4_keras import fit_keras, predict_keras
                model, hist, secs = fit_keras(arch, data)
                p = predict_keras(model, Xeval)
                model.save(mdir / f"{name}_keras_{arch}.keras")
                params = int(model.count_params())
            m, test_pred = evaluate(data, p)
            m.update({"epochs": len(hist["loss"]), "secs": secs, "sec_per_epoch": secs / len(hist["loss"]),
                      "params": params, "best_val_loss": min(hist["val_loss"])})
            res[arch] = m
            res["history"][arch] = hist
            store[arch] = test_pred
            print(f"    {arch}: test RMSE {m['test']['rmse']:.3f} (naive {res['naive']['test']['rmse']:.3f}) "
                  f"R2 {m['test']['r2']:.3f}", flush=True)
        save_json(f"ch4_{name}_{fw}", res)
        np.savez(preds_dir / f"ch4_{name}_{fw}.npz", **store)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", choices=("torch", "keras"), required=True)
    ap.add_argument("--exp", nargs="*", default=list(EXPS))
    a = ap.parse_args()
    if a.fw == "torch":
        limit_threads(2)
        from tl3.gpu import cap
        cap(0.15)
    else:
        limit_threads(6)
    run(a.fw, a.exp)
