"""Chương 4 — chạy dự báo glucose CGM 30 phút bằng một framework:

    python scripts/run_ch4_cgm.py --fw torch    # GPU: persistence, Ridge, RNN, LSTM, GRU, LSTM 5 lớp
    python scripts/run_ch4_cgm.py --fw keras    # CPU: RNN, LSTM, GRU, LSTM 5 lớp
    python scripts/run_ch4_cgm.py --fw numpy    # CPU: RNN viết tay (BPTT)

Mỗi tiến trình ghi outputs/metrics/ch4_cgm_<fw>.json và mô hình vào models/ch4/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

TLF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TLF / "src"), str(TLF / "src" / "tl03")]
from dm import ch4_cgm as C  # noqa: E402

OUT = TLF / "outputs" / "metrics"
MODELS = TLF / "models" / "ch4"
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
CONFIGS = [("rnn", 1), ("lstm", 1), ("gru", 1), ("lstm", 5)]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def evaluate(data, pred_std):
    _, _, Y = data.raw["test"]
    return C.scores(Y, data.to_mgdl("test", pred_std))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", choices=["torch", "keras", "numpy"], required=True)
    a = ap.parse_args()
    data = C.CGMData()
    sizes = {s: len(data.raw[s][2]) for s in data.raw}
    log(f"bệnh nhân {data.n_patients}, cửa sổ {sizes}, mu {data.mu:.1f} sd {data.sd:.1f}")
    res = {"framework": a.fw, "patients": data.n_patients, "windows": sizes, "mu": data.mu, "sd": data.sd,
           "lookback": C.LOOKBACK, "ahead_min": C.AHEAD * C.STEP_MIN, "models": {}}
    Xte, _ = data.part("test")

    if a.fw == "torch":
        import torch
        from sklearn.linear_model import Ridge
        res["models"]["persistence"] = {**evaluate(data, np.zeros(len(Xte))), "params": 0}
        Xtr, ytr = data.part("train")
        ridge = Ridge(alpha=1.0).fit(Xtr[..., 0], ytr)
        res["models"]["ridge_ar"] = {**evaluate(data, ridge.predict(Xte[..., 0])), "params": C.LOOKBACK + 1}
        for arch, n in CONFIGS:
            m, hist, secs, n_p = C.fit_torch(data, arch, layers_n=n, log=log)
            t0 = time.perf_counter()
            pred = C.predict_torch(m, Xte)
            infer = (time.perf_counter() - t0) / len(Xte) * 1000
            key = f"{arch}{n}"
            res["models"][key] = {**evaluate(data, pred), "params": n_p, "train_s": secs, "epochs": len(hist["loss"]),
                                  "infer_ms": infer, "history": hist}
            torch.save(m.state_dict(), MODELS / f"cgm_torch_{key}.pt")
            log(key, {k: round(v, 3) for k, v in res["models"][key].items() if isinstance(v, float)})
    elif a.fw == "keras":
        for arch, n in CONFIGS:
            m, hist, secs, n_p = C.fit_keras(data, arch, layers_n=n, log=log)
            t0 = time.perf_counter()
            pred = m.predict(Xte, batch_size=2048, verbose=0).reshape(-1)
            infer = (time.perf_counter() - t0) / len(Xte) * 1000
            key = f"{arch}{n}"
            res["models"][key] = {**evaluate(data, pred), "params": n_p, "train_s": secs, "epochs": len(hist["loss"]),
                                  "infer_ms": infer, "history": hist}
            m.save(MODELS / f"cgm_keras_{key}.keras")
            log(key, {k: round(v, 3) for k, v in res["models"][key].items() if isinstance(v, float)})
    else:
        m, hist, secs, n_p, n_used = C.fit_numpy(data, log=log)
        sub = np.random.default_rng(0).permutation(len(Xte))[:5000]      # dự đoán NumPy từng mẫu: tập con 5.000
        t0 = time.perf_counter()
        pred = C.predict_numpy(m, Xte[sub])
        infer = (time.perf_counter() - t0) / len(sub) * 1000
        _, _, Y = data.raw["test"]
        X = data.raw["test"][0][sub]
        res["models"]["rnn1"] = {**C.scores(Y[sub], X[:, -1] + pred * data.sd), "params": int(n_p), "train_s": secs,
                                 "epochs": len(hist["loss"]), "infer_ms": infer, "history": hist,
                                 "train_used": n_used, "test_used": len(sub)}
        np.savez(MODELS / "cgm_numpy_rnn1.npz", **m.p)
        log("rnn1", {k: round(v, 3) for k, v in res["models"]["rnn1"].items() if isinstance(v, float)})
    (OUT / f"ch4_cgm_{a.fw}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    log("xong")


if __name__ == "__main__":
    main()
