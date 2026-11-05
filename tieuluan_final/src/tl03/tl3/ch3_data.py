"""Chương 3 — nạp bốn bộ dữ liệu ảnh (MNIST, Fashion-MNIST, Rock/Paper/Scissors, CIFAR-10) về một dạng chung.

Mỗi bộ được lưu một lần thành data/ch3/<tên>.npz gồm x_train, y_train, x_test, y_test (uint8, NHWC) bởi
scripts/ch3_download.py; mọi framework (NumPy, PyTorch, Keras) đọc cùng tệp này và cùng phép chia validation,
nên chênh lệch kết quả giữa các framework không đến từ dữ liệu.
"""
import numpy as np

from .common import SEED, data_dir

DATA = data_dir("ch3")

CLASSES = {
    "mnist": [str(i) for i in range(10)],
    "fashion": ["Áo phông", "Quần dài", "Áo len", "Váy liền", "Áo khoác",
                "Xăng-đan", "Sơ mi", "Giày thể thao", "Túi xách", "Bốt cổ ngắn"],
    "rps": ["Búa (rock)", "Bao (paper)", "Kéo (scissors)"],
    "retina": ["Mức 0 — không bệnh", "Mức 1 — nhẹ", "Mức 2 — trung bình", "Mức 3 — nặng", "Mức 4 — tăng sinh"],
    "cifar10": ["Máy bay", "Ô tô", "Chim", "Mèo", "Hươu", "Chó", "Ếch", "Ngựa", "Tàu thủy", "Xe tải"],
}
CLASSES_EN = {
    "mnist": [str(i) for i in range(10)],
    "fashion": ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"],
    "rps": ["rock", "paper", "scissors"],
    "retina": ["No DR", "Mild", "Moderate", "Severe", "Proliferative"],
    "cifar10": ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
}
TITLE = {"retina": "RetinaMNIST (đáy mắt)", "mnist": "MNIST", "fashion": "Fashion-MNIST", "rps": "Rock/Paper/Scissors", "cifar10": "CIFAR-10"}
VAL_FRAC = 0.1


def load_raw(name):
    """Trả (x_train, y_train, x_test, y_test) uint8 NHWC đúng như tệp gốc."""
    z = np.load(DATA / f"{name}.npz")
    return z["x_train"], z["y_train"], z["x_test"], z["y_test"]


def split_val(x, y, frac=VAL_FRAC, seed=SEED):
    """Chia train/validation phân tầng theo lớp, cố định hạt giống — dùng chung cho mọi framework."""
    rng = np.random.default_rng(seed)
    val_idx = []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        val_idx.append(idx[:int(round(frac * len(idx)))])
    val_idx = np.sort(np.concatenate(val_idx))
    mask = np.ones(len(y), bool)
    mask[val_idx] = False
    return x[mask], y[mask], x[val_idx], y[val_idx]


def load(name, frac=VAL_FRAC):
    """Ảnh float32 trong [0, 1], dạng NHWC, cùng phép chia train/val/test."""
    xtr, ytr, xte, yte = load_raw(name)
    xtr, ytr, xva, yva = split_val(xtr, ytr, frac)
    f = lambda a: a.astype(np.float32) / 255.0
    return (f(xtr), ytr.astype(np.int64)), (f(xva), yva.astype(np.int64)), (f(xte), yte.astype(np.int64))


def channel_stats(x):
    """Trung bình và độ lệch chuẩn theo kênh trên tập huấn luyện (dùng để chuẩn hoá ở mô hình cải tiến)."""
    return x.mean(axis=(0, 1, 2)), x.std(axis=(0, 1, 2)) + 1e-6
