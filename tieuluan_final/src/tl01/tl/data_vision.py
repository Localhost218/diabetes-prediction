"""Chương 3: tải SVHN và EuroSAT (RGB) qua torchvision, lưu thành npz uint8 để nạp thẳng lên GPU.

SVHN giữ nguyên cách chia gốc 73.257 / 26.032 ảnh. EuroSAT không có tập kiểm thử chính thức nên
chia phân tầng 80/20 theo lớp với seed cố định.
"""
import numpy as np

from .common import DATA, SEED

VISION = DATA / "vision"
SVHN_NPZ = VISION / "svhn.npz"
EUROSAT_NPZ = VISION / "eurosat.npz"
SVHN_CLASSES = [str(i) for i in range(10)]
EUROSAT_CLASSES_VI = {
    "AnnualCrop": "Cây hằng năm", "Forest": "Rừng", "HerbaceousVegetation": "Thảm cỏ",
    "Highway": "Đường cao tốc", "Industrial": "Khu công nghiệp", "Pasture": "Đồng cỏ chăn thả",
    "PermanentCrop": "Cây lâu năm", "Residential": "Khu dân cư", "River": "Sông", "SeaLake": "Biển/hồ",
}


SHORT = {"Cây hằng năm": "Cây h.năm", "Rừng": "Rừng", "Thảm cỏ": "Thảm cỏ", "Đường cao tốc": "Cao tốc",
         "Khu công nghiệp": "C.nghiệp", "Đồng cỏ chăn thả": "Đồng cỏ", "Cây lâu năm": "Cây l.năm",
         "Khu dân cư": "Dân cư", "Sông": "Sông", "Biển/hồ": "Biển/hồ"}


def short(name):
    """Tên lớp ngắn cho nhãn dưới ảnh nhỏ trong hình lưới."""
    return SHORT.get(name, name)


def download_svhn():
    from torchvision.datasets import SVHN
    raw = VISION / "raw"
    tr = SVHN(raw, split="train", download=True)
    te = SVHN(raw, split="test", download=True)
    # torchvision trả data dạng (N, 3, 32, 32) uint8, nhãn 0..9 (chữ số 0 đã đổi từ nhãn 10 gốc)
    np.savez_compressed(SVHN_NPZ, x_train=tr.data, y_train=tr.labels.astype(np.int64),
                        x_test=te.data, y_test=te.labels.astype(np.int64))


def download_eurosat():
    from PIL import Image
    from torchvision.datasets import EuroSAT
    ds = EuroSAT(VISION / "raw", download=True)
    xs = np.stack([np.asarray(Image.open(p).convert("RGB")) for p, _ in ds.samples])  # (N,64,64,3)
    ys = np.array([y for _, y in ds.samples], dtype=np.int64)
    rng = np.random.default_rng(SEED)
    test_idx = []
    for c in np.unique(ys):
        idx = np.flatnonzero(ys == c)
        test_idx.extend(rng.choice(idx, size=int(round(0.2 * len(idx))), replace=False))
    is_test = np.zeros(len(ys), bool)
    is_test[test_idx] = True
    x = xs.transpose(0, 3, 1, 2)                                                       # (N,3,64,64)
    np.savez_compressed(EUROSAT_NPZ, x_train=x[~is_test], y_train=ys[~is_test],
                        x_test=x[is_test], y_test=ys[is_test], classes=np.array(ds.classes))


def load(name):
    z = np.load(SVHN_NPZ if name == "svhn" else EUROSAT_NPZ)
    out = {k: z[k] for k in ("x_train", "y_train", "x_test", "y_test")}
    out["classes"] = SVHN_CLASSES if name == "svhn" else [EUROSAT_CLASSES_VI[c] for c in z["classes"]]
    return out


if __name__ == "__main__":
    VISION.mkdir(parents=True, exist_ok=True)
    if not SVHN_NPZ.exists():
        download_svhn()
    if not EUROSAT_NPZ.exists():
        download_eurosat()
    for n in ("svhn", "eurosat"):
        d = load(n)
        print(n, d["x_train"].shape, d["x_test"].shape, np.bincount(d["y_train"]))
