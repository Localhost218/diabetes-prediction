"""Huấn luyện mọi mô hình Chương 3 của một framework:  python scripts/ch3_train.py --fw {scratch,torch,tf} [--exp ...]

Ba tiến trình (NumPy CPU 6 luồng, TensorFlow CPU 8 luồng, PyTorch GPU) chạy song song; mỗi lượt ghi
models/ch3/results/<exp>__<model>.json, bỏ qua lượt đã có kết quả.
"""
import argparse
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch3_common import CFG, MODELS, TF_PATIENCE, evaluate, has_result, load_result, resize_nhwc, save_result  # noqa: E402
from tl3.ch3_data import load  # noqa: E402
from tl3.common import LOGS, limit_threads, seed_all  # noqa: E402

EXPS = ["retina", "mnist", "fashion", "rps", "cifar10"]


def logger(fw):
    f = open(LOGS / f"ch3_train_{fw}.log", "a", encoding="utf-8")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        f.write(s + "\n")
        f.flush()
    return log


def data_for(exp, size):
    (xtr, ytr), (xva, yva), (xte, yte) = load(exp)
    return [resize_nhwc(xtr, size), ytr, resize_nhwc(xva, size), yva, resize_nhwc(xte, size), yte]


def finish(exp, mid, hist, best_ep, params, train_time, proba, yte, infer_s, log, extra=None):
    res = {"exp": exp, "model": mid, "params": params, "train_time": train_time, "best_epoch": best_ep,
           "epochs": len(hist["loss"]), "infer_ms": 1000 * infer_s / len(yte), "history": hist,
           **evaluate(yte, proba), **(extra or {})}
    save_result(exp, mid, res, proba)
    log(f"== {exp} {mid}: acc {res['accuracy']:.4f} f1 {res['f1']:.4f} time {train_time:.0f}s params {params}")


def run_scratch(exp, log):
    from tl3.ch3_scratch import ScratchCNN, train_scratch
    cfg = CFG[exp]
    xtr, ytr, xva, yva, xte, yte = data_for(exp, cfg["basic_size"])
    n_full = len(xtr)
    if cfg["scratch_n"] and cfg["scratch_n"] < len(xtr):
        idx = np.random.default_rng(42).permutation(len(xtr))[:cfg["scratch_n"]]
        xtr, ytr = xtr[idx], ytr[idx]
    nchw = lambda a: np.ascontiguousarray(a.transpose(0, 3, 1, 2))
    m = ScratchCNN(nchw(xtr[:1]).shape[1:], int(ytr.max()) + 1)
    t0 = time.perf_counter()
    hist = train_scratch(m, nchw(xtr), ytr, nchw(xva), yva, cfg["scratch_ep"], batch=64, log=log)
    tt = time.perf_counter() - t0
    t0 = time.perf_counter()
    proba = m.predict_proba(nchw(xte))
    np.savez(MODELS / f"{exp}_scratch.npz", **m.p)
    finish(exp, "scratch", hist, len(hist["loss"]), m.n_params(), tt, proba, yte, time.perf_counter() - t0, log,
           {"train_used": int(len(xtr)), "train_full": int(n_full)})


def run_torch(exp, improved, log):
    import json
    import torch
    from tl3 import gpu
    from tl3.ch3_torch import BasicCNN, ImprovedCNN, n_params, predict_proba, train_torch
    cfg = CFG[exp]
    size = cfg["improved_size" if improved else "basic_size"]
    xtr, ytr, xva, yva, xte, yte = data_for(exp, size)
    k, c = int(ytr.max()) + 1, xtr.shape[3]
    mid = "torch_improved" if improved else "torch_basic"
    # Đo thực tế: Basic CNN < 0,3 GB, Improved CNN < 0,7 GB VRAM (kể cả dữ liệu nằm trên GPU) — đều là việc GPU nhẹ theo
    # quy ước dự án (≤ 1,5 GB) nên không giữ khoá nặng, chỉ giới hạn bộ nhớ để không lấn tiến trình khác.
    gpu.cap(0.3 if improved else 0.12)
    seed_all()
    m = ImprovedCNN(c, k, cfg["widths"]) if improved else BasicCNN(c, k, size)
    t0 = time.perf_counter()
    hist, best_ep, norm = train_torch(m, xtr, ytr, xva, yva, cfg,
                                      cfg["ep_improved" if improved else "ep_basic"], improved, log=log)
    tt = time.perf_counter() - t0
    mean, std = (None, None) if norm is None else (norm[0].cuda(), norm[1].cuda())
    predict_proba(m, xte[:64], mean, std)                      # khởi động (warm-up) trước khi đo
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    proba = predict_proba(m, xte, mean, std)
    infer = time.perf_counter() - t0
    torch.save(m.state_dict(), MODELS / f"{exp}_{mid}.pt")
    meta = {"size": size, "in_ch": c, "n_classes": k, "widths": cfg["widths"],
            "mean": None if norm is None else norm[0].flatten().tolist(),
            "std": None if norm is None else norm[1].flatten().tolist()}
    (MODELS / f"{exp}_{mid}.json").write_text(json.dumps(meta), encoding="utf-8")
    peak = torch.cuda.max_memory_allocated() / 2 ** 20
    finish(exp, mid, hist, best_ep, n_params(m), tt, proba, yte, infer, log, {"gpu_peak_mb": peak})


def run_tf(exp, improved, log):
    import keras
    from tl3.ch3_keras import build_basic, build_improved, train_keras
    cfg = CFG[exp]
    size = cfg["improved_size" if improved else "basic_size"]
    xtr, ytr, xva, yva, xte, yte = data_for(exp, size)
    k = int(ytr.max()) + 1
    keras.utils.set_random_seed(42)
    if improved:
        m = build_improved(xtr.shape[1:], k, cfg, xtr.mean(axis=(0, 1, 2)), xtr.std(axis=(0, 1, 2)) + 1e-6)
    else:
        m = build_basic(xtr.shape[1:], k)
    mid = "tf_improved" if improved else "tf_basic"
    t0 = time.perf_counter()
    if improved:    # CPU: số epoch tối đa + dừng sớm theo validation + ngân sách thời gian
        ckpt = MODELS / "ckpt" / f"{exp}_{mid}"
        hist, best_ep, stop, m = train_keras(m, xtr, ytr, xva, yva, cfg, cfg["ep_improved_tf"], True,
                                             patience=TF_PATIENCE, budget_s=60 * cfg["tf_budget_min"], ckpt=ckpt)
    else:
        hist, best_ep, stop, m = train_keras(m, xtr, ytr, xva, yva, cfg, cfg["ep_basic"], False)
    # lượt tiếp tục từ điểm lưu: tổng thời gian các epoch (kể cả các epoch chạy trước sự cố), không phải chỉ tiến trình này
    tt = sum(hist["time"]) if improved else time.perf_counter() - t0
    m.predict(xte[:64], verbose=0)
    t0 = time.perf_counter()
    proba = m.predict(xte, batch_size=512, verbose=0)
    infer = time.perf_counter() - t0
    tmp = MODELS / f"{exp}_{mid}.tmp.keras"     # lưu tạm rồi thay nguyên tử: Web App không nạp phải tệp dở
    m.save(tmp)
    os.replace(tmp, MODELS / f"{exp}_{mid}.keras")
    for ep in range(len(hist["loss"])):
        log(f"epoch {ep + 1} loss {hist['loss'][ep]:.4f} acc {hist['acc'][ep]:.4f} val_acc {hist['val_acc'][ep]:.4f}")
    finish(exp, mid, hist, best_ep, int(m.count_params()), tt, proba, yte, infer, log,
           {"stop": stop, "max_epochs": cfg["ep_improved_tf"] if improved else cfg["ep_basic"]})
    if improved:    # xong lượt thì điểm lưu không còn cần
        shutil.rmtree(MODELS / "ckpt" / f"{exp}_{mid}", ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fw", required=True, choices=["scratch", "torch", "tf"])
    ap.add_argument("--exp", nargs="*", default=EXPS)
    ap.add_argument("--models", nargs="*", help="chỉ chạy các mô hình này (vd. tf_improved)")
    ap.add_argument("--redo", action="store_true", help="huấn luyện lại kết quả cũ chưa có trường 'stop'")
    ap.add_argument("--threads", type=int)
    a = ap.parse_args()
    log = logger(a.fw)
    limit_threads(a.threads or {"scratch": 6, "tf": 12, "torch": 3}[a.fw])
    seed_all()
    if a.fw == "scratch":
        jobs = [(e, "scratch", lambda e=e: run_scratch(e, log)) for e in a.exp]
    elif a.fw == "torch":   # mọi Basic trước để có sớm kết quả đối chứng, rồi mới tới Improved
        jobs = ([(e, "torch_basic", lambda e=e: run_torch(e, False, log)) for e in a.exp]
                + [(e, "torch_improved", lambda e=e: run_torch(e, True, log)) for e in a.exp])
    else:
        jobs = ([(e, "tf_basic", lambda e=e: run_tf(e, False, log)) for e in a.exp]
                + [(e, "tf_improved", lambda e=e: run_tf(e, True, log)) for e in a.exp])
    for exp, mid, job in jobs:
        if a.models and mid not in a.models:
            continue
        # --redo chỉ chạy lại kết quả sinh trước quy trình dừng sớm: chạy lại lệnh sau sự cố không lặp việc đã xong
        if has_result(exp, mid) and not (a.redo and "stop" not in load_result(exp, mid)):
            log(f"-- bỏ qua {exp} {mid} (đã có)")
            continue
        log(f"### {exp} {mid} bắt đầu {time.strftime('%H:%M:%S')}")
        job()


if __name__ == "__main__":
    main()
