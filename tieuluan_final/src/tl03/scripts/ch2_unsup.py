"""Học không giám sát minh hoạ: K-means (elbow + silhouette) và PCA trên dữ liệu Diabetes đã chuẩn hóa.
Ghi outputs/metrics/ch2_unsup.json và outputs/ch2_unsup.npz (tọa độ PCA để vẽ)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import limit_threads, seed_all  # noqa: E402

limit_threads(6)
seed_all()

import numpy as np  # noqa: E402

from tl3 import ch2_data as cd  # noqa: E402
from tl3.ch2_unsup import kmeans_scan, pca_summary  # noqa: E402
from tl3.common import OUT, save_json  # noqa: E402


def main():
    df = cd.clean_diabetes(cd.load_diabetes_raw()).sample(12000, random_state=42).reset_index(drop=True)
    cols = ["age", "bmi", "HbA1c_level", "blood_glucose_level"]
    X = df[cols].to_numpy(dtype=float)
    Z = (X - X.mean(0)) / X.std(0)
    scan, labels, centers = kmeans_scan(Z, range(2, 11), k_final=4)
    pca, P = pca_summary(Z)
    prof = []
    for c in range(4):
        m = labels == c
        prof.append({"cluster": c, "n": int(m.sum()), "diabetes_rate": float(df.diabetes[m].mean()),
                     **{k: float(df[k][m].mean()) for k in cols}})
    res = {"n": len(df), "features": cols, "scan": scan, "k_final": 4, "profiles": prof, "pca": pca}
    np.savez_compressed(OUT / "ch2_unsup.npz", P=P, labels=labels, y=df.diabetes.to_numpy())
    save_json("ch2_unsup", res)
    print(res["scan"]["silhouette"], pca["explained"])


if __name__ == "__main__":
    main()
