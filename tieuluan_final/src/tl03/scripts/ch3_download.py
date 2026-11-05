"""Tải bốn bộ dữ liệu của Chương 3 và chuẩn hoá về data/ch3/<tên>.npz (uint8, NHWC)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch3_data import DATA  # noqa: E402
from tl3.common import save_json  # noqa: E402

RPS_SIZE = 128


def mnist_csv(slug, train_name, test_name):
    """Đọc MNIST/Fashion-MNIST dạng CSV (cột label + 784 pixel) từ Kaggle."""
    import kagglehub
    root = Path(kagglehub.dataset_download(slug))
    out = []
    for fn in (train_name, test_name):
        df = pd.read_csv(next(root.rglob(fn)))
        y = df.iloc[:, 0].to_numpy(np.uint8)
        x = df.iloc[:, 1:].to_numpy(np.uint8).reshape(-1, 28, 28, 1)
        out += [x, y]
    return out, root


def rps():
    """Rock/Paper/Scissors: ảnh PNG 300×300 RGBA -> RGB 128×128; train 2.520, test 372."""
    import kagglehub
    slug = "sanikamal/rock-paper-scissors-dataset"
    root = Path(kagglehub.dataset_download(slug)) / "Rock-Paper-Scissors"
    classes = ["rock", "paper", "scissors"]
    out = []
    for split in ("train", "test"):
        xs, ys = [], []
        for ci, c in enumerate(classes):
            for p in sorted((root / split / c).glob("*.png")):
                im = Image.open(p).convert("RGB").resize((RPS_SIZE, RPS_SIZE), Image.BILINEAR)
                xs.append(np.asarray(im, np.uint8))
                ys.append(ci)
        out += [np.stack(xs), np.array(ys, np.uint8)]
    return out, root


def cifar10():
    """CIFAR-10 qua torchvision (bản gốc của Krizhevsky, Đại học Toronto)."""
    from torchvision.datasets import CIFAR10
    raw = DATA / "raw"
    tr = CIFAR10(raw, train=True, download=True)
    te = CIFAR10(raw, train=False, download=True)
    return [tr.data, np.array(tr.targets, np.uint8), te.data, np.array(te.targets, np.uint8)], raw


def main():
    info = {}
    jobs = {
        "mnist": lambda: mnist_csv("oddrationale/mnist-in-csv", "mnist_train.csv", "mnist_test.csv"),
        "fashion": lambda: mnist_csv("zalando-research/fashionmnist", "fashion-mnist_train.csv",
                                     "fashion-mnist_test.csv"),
        "rps": rps,
        "cifar10": cifar10,
    }
    for name, fn in jobs.items():
        path = DATA / f"{name}.npz"
        if not path.exists():
            (xtr, ytr, xte, yte), src = fn()
            np.savez_compressed(path, x_train=xtr, y_train=ytr, x_test=xte, y_test=yte)
        z = np.load(path)
        info[name] = {"train": int(len(z["y_train"])), "test": int(len(z["y_test"])),
                      "h": int(z["x_train"].shape[1]), "w": int(z["x_train"].shape[2]),
                      "c": int(z["x_train"].shape[3]), "classes": int(len(np.unique(z["y_train"]))),
                      "mb": round(path.stat().st_size / 2 ** 20, 1)}
        print(name, info[name], flush=True)
    save_json("ch3_data", info)


if __name__ == "__main__":
    main()
