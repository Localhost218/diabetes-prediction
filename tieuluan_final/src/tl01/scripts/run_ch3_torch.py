"""Mục 3.4 - tiến trình GPU: huấn luyện từ đầu bốn biến thể CNN trên SVHN và EuroSAT, cộng 1D CNN.

Toàn bộ ảnh uint8 nạp lên GPU một lần; chuẩn hóa và tăng cường dữ liệu (dịch ngẫu nhiên có đệm
4 điểm, lật / xoay 90 độ với EuroSAT) làm ngay trên GPU, nên không cần DataLoader và CPU rảnh cho
các tiến trình Keras / scikit-learn chạy song song.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl import data_vision as dv  # noqa: E402
from tl import metrics as M  # noqa: E402
from tl.cnn_torch import ARCHS, CNN1D, count_flops  # noqa: E402
from tl.common import METRICS, MODELS, file_kb, log, merge_json, seed_all  # noqa: E402

DEV = torch.device("cuda")
EPOCHS = {"svhn": 15, "eurosat": 30}
OUT = MODELS / "ch3"
OUT.mkdir(parents=True, exist_ok=True)
LOGN = "ch3_torch"


def augment(x, name):
    """x: (B,3,H,W) float trên GPU. Dịch ngẫu nhiên +-4 điểm ảnh từng mẫu bằng chỉ số nâng cao."""
    B, C, H, W = x.shape
    xp = F.pad(x, (4, 4, 4, 4), mode="reflect")
    iy = torch.randint(0, 9, (B,), device=x.device)
    ix = torch.randint(0, 9, (B,), device=x.device)
    ar_h = torch.arange(H, device=x.device)
    ar_w = torch.arange(W, device=x.device)
    rows = (iy[:, None] + ar_h)[:, None, :, None]
    cols = (ix[:, None] + ar_w)[:, None, None, :]
    x = xp[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(C, device=x.device)[None, :, None, None], rows, cols]
    if name == "eurosat":                              # ảnh chụp từ trên xuống: không có chiều "trên/dưới"
        m = torch.rand(B, 1, 1, 1, device=x.device)
        x = torch.where(m < 0.5, x.flip(3), x)
        m = torch.rand(B, 1, 1, 1, device=x.device)
        x = torch.where(m < 0.5, x.flip(2), x)
        m = torch.rand(B, 1, 1, 1, device=x.device)
        x = torch.where(m < 0.5, x.transpose(2, 3), x)
    return x


@torch.no_grad()
def evaluate(model, X, Y, mean, std, batch=1024):
    model.eval()
    logits = []
    for s in range(0, len(X), batch):
        xb = (X[s:s + batch].float() / 255 - mean) / std
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits.append(model(xb.contiguous(memory_format=torch.channels_last)).float())
    logits = torch.cat(logits)
    loss = F.cross_entropy(logits, Y).item()
    pred = logits.argmax(1)
    return loss, (pred == Y).float().mean().item(), pred.cpu().numpy(), logits.softmax(1).cpu().numpy()


def latency(model, shape, device, runs):
    model = model.to(device).eval()
    x = torch.randn(1, *shape, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def train_image(name, arch):
    seed_all()
    d = dv.load(name)
    Xtr = torch.as_tensor(d["x_train"], device=DEV)
    Ytr = torch.as_tensor(d["y_train"], device=DEV)
    Xte = torch.as_tensor(d["x_test"], device=DEV)
    Yte = torch.as_tensor(d["y_test"], device=DEV)
    mean = (Xtr.float() / 255).mean((0, 2, 3), keepdim=True)
    std = (Xtr.float() / 255).std((0, 2, 3), keepdim=True)
    kwargs = {"stem_stride": 2} if arch == "mobilenetv2" and name == "eurosat" else {}
    model = ARCHS[arch](n_classes=10, **kwargs).to(DEV).to(memory_format=torch.channels_last)
    epochs, batch = EPOCHS[name], 256
    steps = (len(Xtr) + batch - 1) // batch
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, nesterov=True, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=epochs * steps, pct_start=0.25)
    hist = {k: [] for k in ("loss", "acc", "val_loss", "val_acc")}
    t0 = time.perf_counter()
    for ep in range(epochs):
        model.train()
        order = torch.randperm(len(Xtr), device=DEV)
        tot_loss = tot_ok = 0.0
        for s in range(0, len(Xtr), batch):
            idx = order[s:s + batch]
            xb = augment((Xtr[idx].float() / 255 - mean) / std, name).contiguous(memory_format=torch.channels_last)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
                loss = F.cross_entropy(out, Ytr[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot_loss += loss.item() * len(idx)
            tot_ok += (out.argmax(1) == Ytr[idx]).sum().item()
        vl, va, _, _ = evaluate(model, Xte, Yte, mean, std)
        for k, v in zip(hist, (tot_loss / len(Xtr), tot_ok / len(Xtr), vl, va)):
            hist[k].append(v)
        log(f"{name}/{arch} epoch {ep + 1}/{epochs}: loss={hist['loss'][-1]:.4f} acc={hist['acc'][-1]:.4f} "
            f"test_acc={va:.4f}", LOGN)
    train_s = time.perf_counter() - t0
    _, acc, pred, prob = evaluate(model, Xte, Yte, mean, std)
    res = M.classification(d["y_test"], pred, binary=False)
    path = OUT / f"{name}_{arch}.pt"
    torch.save({"state": model.state_dict(), "mean": mean.cpu(), "std": std.cpu(), "classes": d["classes"],
                "kwargs": kwargs}, path)
    shape = tuple(d["x_train"].shape[1:])
    model.float()
    res.update(train_s=train_s, epochs=epochs, hist=hist, size_kb=file_kb(path),
               params=sum(p.numel() for p in model.parameters()),
               gflops=count_flops(model, shape) / 1e9,
               lat_gpu_ms=latency(model, shape, DEV, 200),
               lat_cpu_ms=latency(model, shape, torch.device("cpu"), 30))
    res["fps_gpu"] = 1000 / res["lat_gpu_ms"]
    model.to(DEV)
    merge_json(f"ch3_{name}", arch, res)
    np.savez_compressed(METRICS / f"ch3_pred_{name}_{arch}.npz", pred=pred, prob=prob.astype(np.float16))
    log(f"{name}/{arch}: acc={res['acc']:.4f} f1={res['f1']:.4f} params={res['params']} "
        f"gflops={res['gflops']:.3f} gpu={res['lat_gpu_ms']:.2f}ms cpu={res['lat_cpu_ms']:.2f}ms", LOGN)


def train_1d():
    """Mục 3.2.3: 1D CNN trên 100.000 hồ sơ tiểu đường, cùng chia 80/20 và tập kiểm định như MLP."""
    from sklearn.model_selection import train_test_split

    from tl.common import SEED
    from tl.data_tabular import load_diabetes
    from tl.mlp_torch import fit, predict_raw
    seed_all()
    d = load_diabetes()
    Xa, Xv, ya, yv = train_test_split(d["X_train"], d["y_train"], test_size=0.1, random_state=SEED, stratify=d["y_train"])
    model = CNN1D(d["X_train"].shape[1])
    t0 = time.perf_counter()
    hist = fit(model, Xa, ya, Xv, yv, "binary", epochs=40, batch=256, log=lambda m: log(f"diabetes1d {m}", LOGN))
    train_s = time.perf_counter() - t0
    prob = 1 / (1 + np.exp(-predict_raw(model, d["X_test"]).ravel()))
    res = M.classification(d["y_test"], (prob >= 0.5).astype(int), prob)
    path = OUT / "diabetes_cnn1d.pt"
    torch.save(model.state_dict(), path)
    shape = (d["X_train"].shape[1],)
    res.update(train_s=train_s, epochs=len(hist["loss"]), hist=hist, size_kb=file_kb(path),
               params=sum(p.numel() for p in model.parameters()), gflops=count_flops(model, shape) / 1e9,
               lat_gpu_ms=latency(model, shape, DEV, 300), lat_cpu_ms=latency(model, shape, torch.device("cpu"), 300))
    res["fps_gpu"] = 1000 / res["lat_gpu_ms"]
    merge_json("ch3_diabetes1d", "cnn1d", res)
    log(f"diabetes1d: acc={res['acc']:.4f} f1={res['f1']:.4f} params={res['params']}", LOGN)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="vd: svhn/resnet18 eurosat/vgg11 diabetes1d")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    torch.set_num_threads(4)
    jobs = a.only or ["diabetes1d"] + [f"{n}/{m}" for n in ("svhn", "eurosat") for m in ARCHS]
    for j in jobs:
        if j == "diabetes1d":
            train_1d()
        else:
            train_image(*j.split("/"))
