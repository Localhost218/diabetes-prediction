"""Chương 2 — K-means và PCA viết gọn (K-means tự cài cho minh hoạ thuật toán, PCA qua phân rã SVD)."""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def kmeans_numpy(Z, k, n_iter=50, seed=0):
    """K-means Lloyd: khởi tạo k tâm ngẫu nhiên, lặp gán cụm gần nhất rồi cập nhật tâm = trung bình cụm."""
    rng = np.random.default_rng(seed)
    C = Z[rng.choice(len(Z), k, replace=False)]
    for _ in range(n_iter):
        d = ((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(1)
        newC = np.array([Z[lab == j].mean(0) if np.any(lab == j) else C[j] for j in range(k)])
        if np.allclose(newC, C):
            break
        C = newC
    inertia = float(((Z - C[lab]) ** 2).sum())
    return lab, C, inertia


def kmeans_scan(Z, ks, k_final=4, seed=42):
    """Quét k: inertia (phương pháp khuỷu tay) và hệ số silhouette; trả nhãn và tâm cụm ở k_final."""
    scan = {"k": [], "inertia": [], "silhouette": []}
    sub = np.random.default_rng(seed).choice(len(Z), min(5000, len(Z)), replace=False)
    for k in ks:
        km = KMeans(k, n_init=10, random_state=seed).fit(Z)
        scan["k"].append(k)
        scan["inertia"].append(float(km.inertia_))
        scan["silhouette"].append(float(silhouette_score(Z[sub], km.labels_[sub])))
    km = KMeans(k_final, n_init=10, random_state=seed).fit(Z)
    return scan, km.labels_, km.cluster_centers_


def pca_svd(Z, n=2):
    """PCA: tâm hóa, SVD Z = U S V^T; thành phần chính là hàng của V^T, phương sai giải thích ~ S²."""
    Zc = Z - Z.mean(0)
    U, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    ratio = S ** 2 / np.sum(S ** 2)
    return Zc @ Vt[:n].T, Vt, ratio


def pca_summary(Z):
    P, Vt, ratio = pca_svd(Z, 2)
    return {"explained": ratio.round(4).tolist(), "cum": np.cumsum(ratio).round(4).tolist(),
            "components": Vt[:2].round(4).tolist()}, P
