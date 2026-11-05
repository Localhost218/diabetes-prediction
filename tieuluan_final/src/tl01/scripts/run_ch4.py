"""Mục 4.4: bốn kiến trúc hồi quy trên hai tập dữ liệu chuỗi, một nền tảng mỗi tiến trình.

    python scripts/run_ch4.py --fw torch    # GPU; cuối cùng kiểm chứng LSTM viết tay bằng NumPy
    python scripts/run_ch4.py --fw keras    # CPU

Chứng khoán FPT: 10 % cuối của phần huấn luyện (theo thời gian) làm tập kiểm định cho dừng sớm.
KKBox: giữ chia 70/15/15 có sẵn; ngưỡng phân loại chọn trên tập kiểm định để F1 lớn nhất.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl import metrics as M  # noqa: E402
from tl.common import METRICS, MODELS, file_kb, log, merge_json, seed_all  # noqa: E402
from tl.data_seq import churn_data, stock_windows, to_price  # noqa: E402

CELLS = ["rnn", "lstm", "gru", "bilstm"]
OUT = MODELS / "ch4"
OUT.mkdir(parents=True, exist_ok=True)
CFG = {"stock": dict(epochs=100, batch=64, patience=15), "churn": dict(epochs=25, batch=512, patience=5)}


class Backend:
    """Bọc khác biệt API giữa hai nền tảng để vòng thí nghiệm viết một lần."""

    def __init__(self, fw):
        self.fw = fw
        if fw == "torch":
            import torch
            torch.set_num_threads(4)
            from tl import rnn_torch
            self.torch, self.rt = torch, rnn_torch
        else:
            from tl import rnn_keras
            self.rk = rnn_keras

    def build(self, cell, shape, task):
        if self.fw == "torch":
            return self.rt.SeqModel(cell, input_dim=shape[1])
        return self.rk.build_keras_seq_model(cell, shape, out="reg" if task == "stock" else "clf")

    def fit(self, m, X, y, Xv, yv, task, logf):
        c = CFG[task]
        if self.fw == "torch":
            return self.rt.fit(m, X, y, Xv, yv, loss="mse" if task == "stock" else "bce", log=logf, **c)
        return self.rk.fit(m, X, y, Xv, yv, **c)

    def predict(self, m, X, task):
        if self.fw == "torch":
            out = self.rt.predict(m, X)
            return out if task == "stock" else 1 / (1 + np.exp(-out))
        return m.predict(X, batch_size=4096, verbose=0).ravel()

    def params(self, m):
        return sum(p.numel() for p in m.parameters()) if self.fw == "torch" else m.count_params()

    def save(self, m, name):
        if self.fw == "torch":
            path = OUT / f"{name}_torch.pt"
            self.torch.save(m.state_dict(), path)
        else:
            path = OUT / f"{name}_keras.keras"
            m.save(path)
        return path

    def latency_cpu(self, m, x, runs=100):
        """Độ trễ một mẫu trên CPU (4 luồng) để so sánh công bằng hai nền tảng trên cùng phần cứng."""
        if self.fw == "torch":
            torch = self.torch
            mc = m.to("cpu").eval()
            xt = torch.as_tensor(x[:1])
            with torch.no_grad():
                for _ in range(10):
                    mc(xt)
                t0 = time.perf_counter()
                for _ in range(runs):
                    mc(xt)
            ms = (time.perf_counter() - t0) / runs * 1000
            m.to(self.rt.DEVICE)
            return ms
        import tensorflow as tf
        xt = tf.constant(x[:1])
        f = tf.function(lambda a: m(a, training=False))
        for _ in range(10):
            f(xt)
        t0 = time.perf_counter()
        for _ in range(runs):
            f(xt)
        return (time.perf_counter() - t0) / runs * 1000


def run_stock(B, name):
    d = stock_windows("FPT")
    n_val = int(len(d["X_train"]) * 0.1)
    Xa, ya = d["X_train"][:-n_val], d["y_train"][:-n_val]
    Xv, yv = d["X_train"][-n_val:], d["y_train"][-n_val:]
    y_true = d["close_next_test"] / 1000                        # nghìn đồng
    base = d["close_now_test"] / 1000                           # dự báo "giữ nguyên giá hôm nay"
    merge_json(name, "stock/naive", M.regression(y_true, base) | {"pred": base, "r2_ret": 0.0})
    preds = {}
    for cell in CELLS:
        seed_all()
        logf = lambda m: log(f"stock/{cell} {m}", name)  # noqa: E731
        m = B.build(cell, d["X_train"].shape[1:], "stock")
        t0 = time.perf_counter()
        hist = B.fit(m, Xa, ya, Xv, yv, "stock", logf)
        train_s = time.perf_counter() - t0
        ret = B.predict(m, d["X_test"], "stock")
        yp = to_price(ret, d["close_now_test"]) / 1000
        preds[cell] = yp
        res = M.regression(y_true, yp)
        res["r2_ret"] = M.regression(d["y_test"], ret)["r2"]     # R^2 trên lợi suất: thước đo khó hơn
        # tỉ lệ đoán đúng chiều tăng/giảm so với giá đóng cửa hôm nay
        res["dir_acc"] = float(np.mean(np.sign(yp - base) == np.sign(y_true - base)))
        path = B.save(m, f"stock_{cell}")
        res.update(train_s=train_s, epochs=len(hist["loss"]), hist=hist, params=B.params(m),
                   size_kb=file_kb(path), lat_cpu_ms=B.latency_cpu(m, d["X_test"]), pred=yp)
        merge_json(name, f"stock/{cell}", res)
        log(f"stock/{cell}: rmse={res['rmse']:.3f} mae={res['mae']:.3f} r2={res['r2']:.4f} "
            f"lat={res['lat_cpu_ms']:.2f}ms", name)
    np.savez(METRICS / f"ch4_stock_pred_{B.fw}.npz", y_true=y_true, naive=base,
             dates=np.array([str(x.date()) for x in d["dates_test"]]), **preds)
    return d


def run_churn(B, name):
    d = churn_data()
    for cell in CELLS:
        seed_all()
        logf = lambda m: log(f"churn/{cell} {m}", name)  # noqa: E731
        m = B.build(cell, d["X_train"].shape[1:], "churn")
        t0 = time.perf_counter()
        hist = B.fit(m, d["X_train"], d["y_train"].astype(np.float32), d["X_val"], d["y_val"].astype(np.float32),
                     "churn", logf)
        train_s = time.perf_counter() - t0
        thr = M.best_f1_threshold(d["y_val"], B.predict(m, d["X_val"], "churn"))
        prob = B.predict(m, d["X_test"], "churn")
        res = M.classification(d["y_test"], (prob >= thr).astype(int), prob)
        path = B.save(m, f"churn_{cell}")
        res.update(threshold=thr, train_s=train_s, epochs=len(hist["loss"]), hist=hist, params=B.params(m),
                   size_kb=file_kb(path), lat_cpu_ms=B.latency_cpu(m, d["X_test"]))
        merge_json(name, f"churn/{cell}", res)
        np.save(METRICS / f"ch4_churn_prob_{B.fw}_{cell}.npy", prob.astype(np.float32))
        log(f"churn/{cell}: acc={res['acc']:.4f} f1={res['f1']:.4f} auc={res['auc']:.4f} thr={thr:.2f}", name)


def scratch_check(B, d):
    """Nạp trọng số LSTM PyTorch đã huấn luyện vào bản NumPy và so sánh dự báo trên tập kiểm thử."""
    import torch

    from tl.rnn_scratch import LSTMRegressorScratch
    from tl.rnn_torch import SeqModel
    m = SeqModel("lstm", input_dim=d["X_test"].shape[2])
    m.load_state_dict(torch.load(OUT / "stock_lstm_torch.pt"))
    m.eval()
    with torch.no_grad():
        ref = m(torch.as_tensor(d["X_test"])).numpy().ravel()
    sc = LSTMRegressorScratch.from_torch(m)
    t0 = time.perf_counter()
    out = sc.predict(d["X_test"])
    ms = (time.perf_counter() - t0) / len(out) * 1000
    y_true = d["close_next_test"] / 1000
    res = {"max_abs_diff": float(np.abs(out - ref).max()), "infer_ms": ms,
           "rmse_scratch": M.regression(y_true, to_price(out, d["close_now_test"]) / 1000)["rmse"],
           "rmse_torch_cpu": M.regression(y_true, to_price(ref, d["close_now_test"]) / 1000)["rmse"]}
    merge_json("ch4_scratch", "lstm_stock", res)
    log(f"scratch LSTM: {res}", "ch4_torch")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", required=True, choices=["torch", "keras"])
    ap.add_argument("--task", nargs="*", default=["stock", "churn"])
    a = ap.parse_args()
    B = Backend(a.fw)
    name = f"ch4_{a.fw}"
    d = run_stock(B, name) if "stock" in a.task else None
    if a.fw == "torch" and d is not None:
        scratch_check(B, d)
    if "churn" in a.task:
        run_churn(B, name)
