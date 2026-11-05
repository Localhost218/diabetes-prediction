"""Đo độ trễ suy luận của mọi mô hình CNN và RNN trong cùng điều kiện, khi máy không chạy việc khác.

Độ trễ ghi lúc huấn luyện bị nhiễu vì các tiến trình CPU/GPU khác chạy song song, nên bảng trong
báo cáo dùng số đo ở đây: lô 1 mẫu, 10 lần khởi động, trung vị của N lần đo; CPU dùng 4 luồng.
Thông lượng (FPS) đo trên GPU với lô 256.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl.cnn_torch import ARCHS  # noqa: E402
from tl.common import MODELS, save_json  # noqa: E402
from tl.rnn_torch import CELLS, SeqModel  # noqa: E402

GPU, CPU = torch.device("cuda"), torch.device("cpu")


def median_ms(fn, runs, sync=False):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(runs):
        if sync:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        if sync:
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    return float(np.median(ts))


@torch.no_grad()
def torch_lat(model, shape, batch_fps=256):
    out = {}
    m = model.to(GPU).eval()
    x = torch.randn(1, *shape, device=GPU)
    out["gpu_ms"] = median_ms(lambda: m(x), 300, sync=True)
    xb = torch.randn(batch_fps, *shape, device=GPU)
    out["fps"] = batch_fps / (median_ms(lambda: m(xb), 30, sync=True) / 1000)
    m = model.to(CPU)
    x = torch.randn(1, *shape)
    out["cpu_ms"] = median_ms(lambda: m(x), 100)
    return out


def keras_lat(path, shape):
    import tensorflow as tf
    import keras
    m = keras.models.load_model(path)
    x = tf.random.normal((1, *shape))
    f = tf.function(lambda a: m(a, training=False))
    return {"cpu_ms": median_ms(lambda: f(x), 100)}


def main():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    res = {}
    for ds, shape in (("svhn", (3, 32, 32)), ("eurosat", (3, 64, 64))):
        for arch in ARCHS:
            ck = torch.load(MODELS / "ch3" / f"{ds}_{arch}.pt", map_location="cpu")
            m = ARCHS[arch](10, **ck.get("kwargs", {}))
            m.load_state_dict(ck["state"])
            res[f"ch3/{ds}/{arch}"] = torch_lat(m, shape)
            print(ds, arch, res[f"ch3/{ds}/{arch}"], flush=True)
        p = MODELS / "ch3" / f"{ds}_custom_keras.keras"
        if p.exists():
            res[f"ch3/{ds}/custom_keras"] = keras_lat(p, (shape[1], shape[2], 3))
    from tl.cnn_torch import CNN1D
    sd = torch.load(MODELS / "ch3" / "diabetes_cnn1d.pt", map_location="cpu")
    n_feat = sd["head.1.weight"].shape[1] // 64          # Linear(64 * F, 64) sau hai tầng Conv1d 64 kênh
    m = CNN1D(n_feat)
    m.load_state_dict(sd)
    res["ch3/diabetes1d/cnn1d"] = torch_lat(m, (n_feat,))
    for task, shape in (("stock", (30, 9)), ("churn", (31, 8))):
        for c in CELLS:
            m = SeqModel(c, input_dim=shape[1])
            m.load_state_dict(torch.load(MODELS / "ch4" / f"{task}_{c}_torch.pt", map_location="cpu"))
            res[f"ch4/{task}/{c}_torch"] = torch_lat(m, shape)
            p = MODELS / "ch4" / f"{task}_{c}_keras.keras"
            if p.exists():
                res[f"ch4/{task}/{c}_keras"] = keras_lat(p, shape)
            print(task, c, res[f"ch4/{task}/{c}_torch"], res.get(f"ch4/{task}/{c}_keras"), flush=True)
    save_json(res, "bench")


if __name__ == "__main__":
    main()
