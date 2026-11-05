"""Mục 2.4-2.5: huấn luyện MLP cùng kiến trúc trên một nền tảng.

    python scripts/run_ch2_mlp.py --fw scratch   # NumPy, CPU
    python scripts/run_ch2_mlp.py --fw keras     # TensorFlow, CPU
    python scripts/run_ch2_mlp.py --fw torch     # PyTorch, GPU

Ba tiến trình chạy song song, mỗi tiến trình ghi tệp kết quả riêng ch2_mlp_<fw>.json.
Tập kiểm định (10 % của tập huấn luyện) chỉ dùng cho dừng sớm; tập kiểm thử chỉ dùng để đánh giá.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl import data_tabular as dt  # noqa: E402
from tl import metrics as M  # noqa: E402
from tl.common import METRICS, MODELS, SEED, file_kb, log, merge_json, seed_all  # noqa: E402

EPOCHS = {"diabetes": 40, "house": 120, "reviews": 20}
TASK = {"diabetes": "binary", "house": "reg", "reviews": "multi"}


def split_val(X, y, task):
    return train_test_split(X, y, test_size=0.1, random_state=SEED, stratify=None if task == "reg" else y)


def make(fw, input_dim, task):
    out = 3 if task == "multi" else 1
    if fw == "scratch":
        from tl.mlp_scratch import MLPScratch
        return MLPScratch(input_dim, output_dim=out, task=task)
    if fw == "keras":
        from tl.mlp_keras import build_keras_mlp
        return build_keras_mlp(input_dim, task=task, n_classes=out)
    from tl.mlp_torch import TorchMLP
    return TorchMLP(input_dim, output_dim=out)


def fit(fw, model, X, y, Xv, yv, task, epochs, logf):
    if fw == "scratch":
        return model.fit(X, y, Xv, yv, epochs=epochs, log=logf)
    if fw == "keras":
        from tl.mlp_keras import fit as kfit
        dense = (lambda a: a.toarray()) if hasattr(X, "toarray") else (lambda a: a)
        return kfit(model, dense(X), y, dense(Xv), yv, epochs=epochs)
    from tl.mlp_torch import fit as tfit
    return tfit(model, X, y, Xv, yv, task, epochs=epochs, log=logf)


def predict(fw, model, X):
    if fw == "scratch":
        return model.predict_raw(X)
    if fw == "keras":
        return model.predict(X.toarray() if hasattr(X, "toarray") else X, batch_size=4096, verbose=0)
    from tl.mlp_torch import predict_raw
    return predict_raw(model, X)


def save(fw, model, ds):
    d = MODELS / "ch2"
    d.mkdir(parents=True, exist_ok=True)
    if fw == "scratch":
        path = d / f"{ds}_mlp_scratch.npz"
        model.save(path)
    elif fw == "keras":
        path = d / f"{ds}_mlp_keras.keras"
        model.save(path)
    else:
        import torch
        path = d / f"{ds}_mlp_torch.pt"
        torch.save(model.state_dict(), path)
    return path


def run(fw, ds):
    name = f"ch2_mlp_{fw}"
    logf = lambda m: log(f"{ds}: {m}", name)  # noqa: E731
    seed_all()
    task = TASK[ds]
    d = dt.LOADERS[ds]()
    Xtr, ytr = d["X_train"], d["y_train"]
    y_mu, y_sd = (ytr.mean(), ytr.std()) if task == "reg" else (0.0, 1.0)
    ytr_fit = ((ytr - y_mu) / y_sd).astype(np.float32) if task == "reg" else ytr
    Xa, Xv, ya, yv = split_val(Xtr, ytr_fit, task)
    model = make(fw, Xtr.shape[1], task)
    t0 = time.perf_counter()
    hist = fit(fw, model, Xa, ya, Xv, yv, task, EPOCHS[ds], logf)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    raw = predict(fw, model, d["X_test"])
    infer_ms = (time.perf_counter() - t0) / len(d["y_test"]) * 1000
    yte = d["y_test"]
    if task == "reg":
        yp = raw.ravel() * y_sd + y_mu
        res = M.regression(yte, yp)
    elif task == "binary":
        # Keras đã qua sigmoid; scratch cũng vậy; PyTorch trả logit
        prob = raw.ravel() if fw != "torch" else 1 / (1 + np.exp(-raw.ravel()))
        yp = (prob >= 0.5).astype(int)
        res = M.classification(yte, yp, prob)
    else:
        yp = raw.argmax(1)
        res = M.classification(yte, yp, binary=False)
    path = save(fw, model, ds)
    n_params = sum(int(np.prod(w.shape)) for w in (model.W + model.b)) if fw == "scratch" else (
        model.count_params() if fw == "keras" else sum(p.numel() for p in model.parameters()))
    res.update(train_s=train_s, infer_ms=infer_ms, size_kb=file_kb(path), epochs=len(hist["loss"]),
               params=n_params, hist=hist, y_mu=float(y_mu), y_sd=float(y_sd))
    merge_json(name, ds, res)
    np.save(METRICS / f"ch2_pred_{ds}_mlp_{fw}.npy", yp)
    logf(f"xong: {({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", required=True, choices=["scratch", "keras", "torch"])
    ap.add_argument("--ds", nargs="*", default=["diabetes", "house", "reviews"])
    a = ap.parse_args()
    for ds in a.ds:
        run(a.fw, ds)
