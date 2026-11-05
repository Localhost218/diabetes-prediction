"""Chương 3 — cấu hình thực nghiệm dùng chung cho ba framework, đánh giá và lưu kết quả từng lượt huấn luyện."""
import json
import os

import numpy as np
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from .common import model_dir

MODELS = model_dir("ch3")
RESULTS = MODELS / "results"
RESULTS.mkdir(exist_ok=True)

# Một cấu hình cho mỗi bộ dữ liệu; mọi framework đọc cùng bảng này nên cỡ lô, kích thước ảnh như nhau.
# Riêng Improved CNN bản TensorFlow chạy trên CPU (TF không hỗ trợ GPU trên Windows): ep_improved_tf là số epoch TỐI ĐA,
# dừng sớm khi val không cải thiện TF_PATIENCE epoch, và tf_budget_min giới hạn thời gian thật của mỗi lượt huấn luyện
# (ước từ thời gian một epoch đo được: hai làn song song 8 luồng, mỗi làn tối đa khoảng 3,5 giờ).
CFG = {
    "mnist":   dict(basic_size=28, improved_size=28, widths=(32, 64, 128), batch=128, ep_basic=8, ep_improved=15, ep_improved_tf=6, tf_budget_min=60,
                    scratch_n=30000, scratch_ep=5, flip=False, rot=10, shift=0.10, zoom=0.10, color=0.0),
    "fashion": dict(basic_size=28, improved_size=28, widths=(32, 64, 128), batch=128, ep_basic=10, ep_improved=25, ep_improved_tf=7, tf_budget_min=85,
                    scratch_n=30000, scratch_ep=6, flip=True, rot=8, shift=0.10, zoom=0.08, color=0.0),
    "rps":     dict(basic_size=64, improved_size=96, widths=(32, 64, 128, 256), batch=32, ep_basic=15,
                    ep_improved=25, ep_improved_tf=6, tf_budget_min=55, scratch_n=None, scratch_ep=12, flip=True, rot=20, shift=0.10, zoom=0.15,
                    color=0.2),
    # Mạch tiểu đường của tiểu luận final: RetinaMNIST 128×128 (5 mức bệnh võng mạc ĐTĐ), 1.200 ảnh train+val, 400 ảnh test
    "retina":  dict(basic_size=48, improved_size=96, widths=(32, 64, 128, 256), batch=32, ep_basic=25, ep_improved=40, ep_improved_tf=15, tf_budget_min=35,
                    scratch_n=None, scratch_ep=12, flip=True, rot=15, shift=0.10, zoom=0.10, color=0.2),
    "cifar10": dict(basic_size=32, improved_size=32, widths=(64, 128, 256), batch=128, ep_basic=15, ep_improved=30, ep_improved_tf=6, tf_budget_min=135,
                    scratch_n=20000, scratch_ep=6, flip=True, rot=0, shift=0.125, zoom=0.0, color=0.1),
}
TF_PATIENCE = 3
MODEL_IDS = ["scratch", "torch_basic", "tf_basic", "torch_improved", "tf_improved"]
MODEL_NAME = {"scratch": "Basic CNN – NumPy", "torch_basic": "Basic CNN – PyTorch", "tf_basic": "Basic CNN – TensorFlow",
              "torch_improved": "Improved CNN – PyTorch", "tf_improved": "Improved CNN – TensorFlow"}


def resize_nhwc(x, size):
    """Đổi kích thước lô ảnh float [0,1] NHWC bằng nội suy diện tích (PIL BOX) — chỉ dùng khi size khác ảnh gốc."""
    if x.shape[1] == size:
        return x
    out = np.empty((len(x), size, size, x.shape[3]), np.float32)
    for i, im in enumerate(x):
        arr = (im * 255).astype(np.uint8)
        pil = Image.fromarray(arr[..., 0] if arr.shape[2] == 1 else arr)
        r = np.asarray(pil.resize((size, size), Image.BOX), np.float32) / 255.0
        out[i] = r[..., None] if r.ndim == 2 else r
    return out


def evaluate(y_true, proba):
    """Accuracy, Precision/Recall/F1 trung bình macro, F1 từng lớp và ma trận nhầm lẫn."""
    y_pred = proba.argmax(1)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    _, _, fc, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)), "precision": float(p), "recall": float(r),
            "f1": float(f), "f1_per_class": fc.tolist(),
            "cm": confusion_matrix(y_true, y_pred, labels=range(proba.shape[1])).tolist()}


def save_result(exp, model_id, result, proba):
    """Ghi models/ch3/results/<exp>__<model>.json và xác suất dự đoán trên tập test (.npy) cho hình ảnh sai."""
    # ghi ra tệp tạm rồi os.replace: tiến trình khác (Web App) không bao giờ đọc phải tệp ghi dở
    npy, js = RESULTS / f"{exp}__{model_id}_proba.npy", RESULTS / f"{exp}__{model_id}.json"
    with open(npy.with_suffix(".tmp"), "wb") as f:
        np.save(f, proba.astype(np.float32))
    os.replace(npy.with_suffix(".tmp"), npy)
    js.with_suffix(".tmp").write_text(
        json.dumps(result, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
        encoding="utf-8")
    os.replace(js.with_suffix(".tmp"), js)


def load_result(exp, model_id):
    p = RESULTS / f"{exp}__{model_id}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def has_result(exp, model_id):
    return (RESULTS / f"{exp}__{model_id}.json").exists()
