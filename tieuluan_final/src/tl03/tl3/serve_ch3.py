"""Chương 3 — hàm suy luận cho Web App: phân loại ảnh MNIST, Fashion-MNIST, Rock/Paper/Scissors, CIFAR-10.

Năm mô hình mỗi bộ (NumPy, PyTorch, TensorFlow; Basic và Improved) nạp lười từ models/ch3/ và chạy trên CPU
để Web App không tranh GPU với các tiến trình huấn luyện. Không phụ thuộc Flask.
"""
import json
import time
from functools import lru_cache

import numpy as np
from PIL import Image, ImageOps

from .ch3_common import CFG, MODEL_IDS, MODEL_NAME, MODELS
from .ch3_data import CLASSES, TITLE, load_raw
from .common import DATA

EXAMPLES = DATA / "ch3" / "examples"
EXAMPLE_INDEX = {"retina": 3, "mnist": 0, "fashion": 0, "rps": 0, "cifar10": 1}   # ảnh test dùng làm ví dụ cho Web App
TOP_K = 5
EXP_TITLE = {"retina": "Phân độ bệnh võng mạc đái tháo đường (ảnh đáy mắt)",
             "mnist": "Nhận dạng chữ số viết tay (MNIST)", "fashion": "Phân loại trang phục (Fashion-MNIST)",
             "rps": "Nhận dạng cử chỉ Oẳn tù tì (Rock/Paper/Scissors)", "cifar10": "Phân loại ảnh vật thể (CIFAR-10)"}
INPUT_LABEL = {"retina": "Ảnh đáy mắt (JPG/PNG)", "mnist": "Ảnh chữ số (vẽ hoặc tải lên)", "fashion": "Ảnh trang phục", "rps": "Ảnh bàn tay",
               "cifar10": "Ảnh vật thể"}


def _model_file(exp, mid):
    ext = {"scratch": "npz", "tf_basic": "keras", "tf_improved": "keras"}.get(mid, "pt")
    return MODELS / f"{exp}_{mid}.{ext}"


def _available(exp):
    return [m for m in ["torch_improved", "tf_improved", "torch_basic", "tf_basic", "scratch"]
            if _model_file(exp, m).exists()]


def _example(exp):
    """Ghi một ảnh test thật ra data/ch3/examples/<exp>.png (phóng to để xem rõ) và trả đường dẫn tuyệt đối."""
    p = EXAMPLES / f"{exp}.png"
    if not p.exists():
        EXAMPLES.mkdir(parents=True, exist_ok=True)
        x = load_raw(exp)[2][EXAMPLE_INDEX[exp]]
        img = Image.fromarray(x[..., 0] if x.shape[2] == 1 else x)
        img.resize((224, 224), Image.NEAREST if x.shape[0] <= 32 else Image.LANCZOS).save(p)
    return str(p.resolve())


SPECS = {
    f"ch3_{exp}": {
        "title": EXP_TITLE[exp], "kind": "image",
        "inputs": [{"name": "image", "label": INPUT_LABEL[exp], "type": "image", "canvas": exp == "mnist"}],
        "models": _available(exp) or MODEL_IDS,
        "model_names": MODEL_NAME,
        "classes": CLASSES[exp],
        "example": {"image": _example(exp)},
    } for exp in ["retina", "mnist", "fashion", "rps", "cifar10"]
}


def _mnist_center(a):
    """Chuẩn hoá nét vẽ giống MNIST: cắt khung chứa nét, co cạnh dài về 20 px, đặt khối tâm vào giữa ô 28×28."""
    ys, xs = np.nonzero(a > 0.1)
    if len(ys) == 0:
        return np.zeros((28, 28), np.float32)
    crop = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    s = 20.0 / max(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    crop = np.asarray(Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.LANCZOS)) / 255.0
    out = np.zeros((28, 28), np.float32)
    cy, cx = np.array(np.nonzero(crop > 0)).mean(axis=1) if crop.any() else (nh / 2, nw / 2)
    top, left = int(round(14 - cy)), int(round(14 - cx))
    top, left = min(max(top, 0), 28 - nh), min(max(left, 0), 28 - nw)
    out[top:top + nh, left:left + nw] = crop
    return out


def preprocess(exp, img, size):
    """PIL.Image -> mảng NHWC float32 [0, 1] cùng phân phối với dữ liệu huấn luyện."""
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (0, 0, 0, 255) if exp in ("mnist", "fashion") else (255, 255, 255, 255))
        img = Image.alpha_composite(bg, img)
    if exp in ("mnist", "fashion"):
        g = np.asarray(img.convert("L"), np.float32) / 255.0
        border = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
        if border.mean() > 0.5:                      # ảnh chụp nền sáng -> đảo thành nét sáng trên nền đen như dữ liệu
            g = 1.0 - g
        if exp == "mnist":
            g = _mnist_center(g)
        else:
            g = np.asarray(ImageOps.pad(Image.fromarray((g * 255).astype(np.uint8)), (size, size), color=0),
                           np.float32) / 255.0
        g = np.asarray(Image.fromarray((g * 255).astype(np.uint8)).resize((size, size), Image.BOX), np.float32) / 255.0
        return g[None, :, :, None]
    rgb = ImageOps.fit(img.convert("RGB"), (size, size), Image.LANCZOS)
    return (np.asarray(rgb, np.float32) / 255.0)[None]


@lru_cache(maxsize=None)
def _load(exp, mid):
    """Nạp mô hình một lần; trả (hàm dự đoán NHWC -> xác suất, kích thước ảnh vào)."""
    f = _model_file(exp, mid)
    if not f.exists():
        raise FileNotFoundError(f"Chưa có mô hình {mid} cho {exp}: {f.name}")
    cfg = CFG[exp]
    if mid == "scratch":
        from .ch3_scratch import ScratchCNN
        z = np.load(f)
        c = 1 if exp in ("mnist", "fashion") else 3
        s = cfg["basic_size"]
        m = ScratchCNN((c, s, s), z["W4"].shape[1])
        m.p = {k: z[k] for k in z.files}
        return (lambda x: m.predict_proba(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))), s
    if mid.startswith("tf_"):
        import keras
        m = keras.models.load_model(f)
        s = cfg["improved_size" if mid == "tf_improved" else "basic_size"]
        return (lambda x: np.asarray(m.predict(x, verbose=0))), s
    import torch
    from .ch3_torch import BasicCNN, ImprovedCNN, predict_proba
    meta = json.loads(f.with_suffix(".json").read_text(encoding="utf-8"))
    m = (ImprovedCNN(meta["in_ch"], meta["n_classes"], tuple(meta["widths"])) if mid == "torch_improved"
         else BasicCNN(meta["in_ch"], meta["n_classes"], meta["size"]))
    m.load_state_dict(torch.load(f, map_location="cpu"))
    mean = std = None
    if meta.get("mean") is not None:
        mean = torch.tensor(meta["mean"]).view(1, -1, 1, 1)
        std = torch.tensor(meta["std"]).view(1, -1, 1, 1)
    return (lambda x: predict_proba(m, x, mean, std, dev="cpu")), meta["size"]


def predict(exp_id, payload, model_id=None):
    exp = exp_id.removeprefix("ch3_")
    if exp_id not in SPECS:
        raise KeyError(f"exp_id không hợp lệ: {exp_id}")
    img = payload.get("image") if isinstance(payload, dict) else payload
    if img is None:
        img = Image.open(SPECS[exp_id]["example"]["image"])
    elif not isinstance(img, Image.Image):
        img = Image.open(img)
    mid = model_id or SPECS[exp_id]["models"][0]
    t0 = time.perf_counter()
    fn, size = _load(exp, mid)
    x = preprocess(exp, img, size)
    p = fn(x)[0]
    order = np.argsort(p)[::-1][:TOP_K]
    names = CLASSES[exp]
    a = x[0, :, :, 0] if x.shape[3] == 1 else x[0]
    shown = Image.fromarray((a * 255).astype(np.uint8))           # ảnh đúng như mô hình nhìn thấy
    return {
        "label": names[int(order[0])],
        "proba": {names[int(i)]: float(p[i]) for i in order},
        "details": {"dataset": TITLE[exp], "model_name": MODEL_NAME[mid], "input_size": f"{size}×{size}",
                    "confidence": float(p[order[0]]), "model_input": shown.resize((168, 168), Image.NEAREST)},
        "model": mid,
        "latency_ms": 1000 * (time.perf_counter() - t0),
    }
