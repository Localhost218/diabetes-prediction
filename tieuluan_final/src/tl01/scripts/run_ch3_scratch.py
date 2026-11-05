"""Mục 3.3.1: chạy toàn bộ Custom CNN đã huấn luyện (PyTorch, SVHN) bằng các hàm NumPy viết tay.

Kiểm chứng ba điều: (1) bản vòng lặp và bản im2col cho cùng kết quả, (2) đầu ra NumPy khớp PyTorch,
(3) độ chính xác trên 2.000 ảnh kiểm thử giống hệt, cộng thời gian chạy để thấy vì sao cần thư viện.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl import cnn_scratch as S  # noqa: E402
from tl import data_vision as dv  # noqa: E402
from tl.cnn_torch import CustomCNN  # noqa: E402
from tl.common import MODELS, log, save_json  # noqa: E402

N_IMG = 2000


def numpy_forward(sd, x, conv=S.conv2d_im2col):
    """Lặp lại đúng thứ tự tầng của CustomCNN: 3 x [conv-bn-relu, conv-bn-relu, maxpool] -> GAP -> FC."""
    i = 0
    for _block in range(3):
        for _conv in range(2):
            p = f"features.{i}"
            x = conv(x, sd[f"{p}.0.weight"], np.zeros(sd[f"{p}.0.weight"].shape[0]), padding=1)
            x = S.relu(S.batchnorm_eval(x, sd[f"{p}.1.weight"], sd[f"{p}.1.bias"],
                                        sd[f"{p}.1.running_mean"], sd[f"{p}.1.running_var"]))
            i += 1
        x = S.maxpool2d(x)
        i += 2                                           # bỏ qua chỉ số của MaxPool và Dropout
    x = x.mean(axis=(2, 3))                              # Global Average Pooling
    x = np.maximum(x @ sd["head.0.weight"].T + sd["head.0.bias"], 0)
    return x @ sd["head.3.weight"].T + sd["head.3.bias"]


def main():
    ck = torch.load(MODELS / "ch3" / "svhn_custom.pt", map_location="cpu")
    model = CustomCNN()
    model.load_state_dict(ck["state"])
    model.eval()
    sd = {k: v.double().numpy() for k, v in ck["state"].items()}
    d = dv.load("svhn")
    x = ((d["x_test"][:N_IMG] / 255.0 - ck["mean"].double().numpy()) / ck["std"].double().numpy())
    y = d["y_test"][:N_IMG]
    res = {}
    # (1) vòng lặp và im2col trên 8 ảnh (bản vòng lặp chậm nên chỉ đo ít ảnh)
    w = sd["features.0.0.weight"]
    zero = np.zeros(w.shape[0])
    t0 = time.perf_counter()
    a = S.conv2d_loop(x[:8], w, zero, padding=1)
    res["loop_ms_per_img"] = (time.perf_counter() - t0) / 8 * 1000
    t0 = time.perf_counter()
    b = S.conv2d_im2col(x[:8], w, zero, padding=1)
    res["im2col_ms_per_img"] = (time.perf_counter() - t0) / 8 * 1000
    res["loop_vs_im2col_maxdiff"] = float(np.abs(a - b).max())
    # (2)-(3) cả mạng
    t0 = time.perf_counter()
    logits_np = np.concatenate([numpy_forward(sd, x[s:s + 250]) for s in range(0, N_IMG, 250)])
    res["numpy_net_ms_per_img"] = (time.perf_counter() - t0) / N_IMG * 1000
    torch.set_num_threads(4)
    with torch.no_grad():
        t0 = time.perf_counter()
        logits_t = model(torch.as_tensor(x, dtype=torch.float32)).double().numpy()
        res["torch_cpu_ms_per_img"] = (time.perf_counter() - t0) / N_IMG * 1000
    res["net_maxdiff"] = float(np.abs(logits_np - logits_t).max())
    res["acc_numpy"] = float((logits_np.argmax(1) == y).mean())
    res["acc_torch"] = float((logits_t.argmax(1) == y).mean())
    res["agree"] = float((logits_np.argmax(1) == logits_t.argmax(1)).mean())
    res["n_img"] = N_IMG
    save_json(res, "ch3_scratch")
    log(f"scratch CNN: {res}", "ch3_scratch")


if __name__ == "__main__":
    main()
