"""Hình minh hoạ lý thuyết Chương 2 trên dữ liệu đồ chơi (gradient descent, sigmoid, thiên lệch-phương sai,
biên quyết định, KNN, SVM, độ vẩn đục, K-means, PCA, PLA, Softmax)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import limit_threads, load_json  # noqa: E402

limit_threads(4)

import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from sklearn.datasets import make_blobs, make_moons  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.naive_bayes import GaussianNB  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.tree import DecisionTreeClassifier  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from tl3.ch2_unsup import kmeans_numpy  # noqa: E402
from tl3.plot_style import C, W, plt, save  # noqa: E402

CM2 = ListedColormap(["#DBEAFE", "#FEE2E2"])
PT2 = [C["blue"], C["red"]]


def gd_paths():
    """Gradient descent trên hàm mất mát bậc hai lệch trục với ba tốc độ học."""
    A = np.array([[3.0, 0.0], [0.0, 0.5]])
    f = lambda w: 0.5 * np.einsum("...i,ij,...j->...", w, A, w)
    g = np.linspace(-4, 4, 200)
    G = np.stack(np.meshgrid(g, g), -1)
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.6), sharey=True)
    for ax, lr, title in zip(axes, [0.08, 0.3, 0.66], ["η = 0,08 (chậm)", "η = 0,30 (vừa)", "η = 0,66 (dao động)"]):
        ax.contourf(g, g, f(G), levels=20, cmap="Blues", alpha=0.75)
        w, path = np.array([-3.5, 3.5]), []
        for _ in range(25):
            path.append(w.copy())
            w = w - lr * A @ w
        path = np.array(path)
        ax.plot(path[:, 0], path[:, 1], "o-", color=C["red"], ms=3.5, lw=1.6)
        ax.plot(0, 0, "*", color=C["orange"], ms=14)
        ax.set_title(title, fontsize=11.5)
        ax.set_xlabel("$w_1$")
        ax.grid(False)
    axes[0].set_ylabel("$w_2$")
    save(fig, "ch2_gd_lr")


def sigmoid_logloss():
    z = np.linspace(-8, 8, 300)
    p = np.linspace(0.001, 0.999, 300)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.8))
    a.plot(z, 1 / (1 + np.exp(-z)), color=C["blue"])
    a.axhline(0.5, ls="--", color=C["gray"], lw=1)
    a.axvline(0, ls="--", color=C["gray"], lw=1)
    a.set_title("Hàm sigmoid σ(z)")
    a.set_xlabel("z = wᵀx + b")
    b.plot(p, -np.log(p), color=C["green"], label="y = 1: −log p")
    b.plot(p, -np.log(1 - p), color=C["red"], label="y = 0: −log(1 − p)")
    b.set_ylim(0, 5)
    b.set_title("Binary cross-entropy")
    b.set_xlabel("xác suất dự đoán p")
    b.legend()
    save(fig, "ch2_sigmoid_logloss")


def bias_variance():
    """Khớp đa thức bậc 1/4/15 cho dữ liệu sin nhiễu + sai số Train/Test theo bậc."""
    rng = np.random.default_rng(1)
    x = np.sort(rng.uniform(0, 1, 30))
    f = lambda t: np.sin(2 * np.pi * t)
    y = f(x) + rng.normal(0, 0.25, 30)
    xt = rng.uniform(0, 1, 300)
    yt = f(xt) + rng.normal(0, 0.25, 300)
    grid = np.linspace(0, 1, 300)
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.5), sharey=True)
    for ax, d, t in zip(axes, [1, 4, 15], ["Bậc 1: chưa khớp", "Bậc 4: vừa khớp", "Bậc 15: quá khớp"]):
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], y)
        ax.plot(grid, f(grid), color=C["gray"], ls="--", lw=1.3)
        ax.plot(grid, m.predict(grid[:, None]), color=C["red"])
        ax.scatter(x, y, s=16, color=C["blue"], zorder=3)
        ax.set_ylim(-1.8, 1.8)
        ax.set_title(t, fontsize=11.5)
    save(fig, "ch2_poly_fit")
    degs, tr, te = range(1, 16), [], []
    for d in degs:
        m = make_pipeline(PolynomialFeatures(d), LinearRegression()).fit(x[:, None], y)
        tr.append(np.mean((m.predict(x[:, None]) - y) ** 2))
        te.append(np.mean((m.predict(xt[:, None]) - yt) ** 2))
    fig, ax = plt.subplots(figsize=(W, 3.0))
    ax.plot(list(degs), tr, "o-", label="Sai số Train")
    ax.plot(list(degs), te, "s-", label="Sai số Test")
    ax.set_yscale("log")
    ax.axvspan(0.5, 2.5, color=C["orange"], alpha=0.12)
    ax.axvspan(9.5, 15.5, color=C["red"], alpha=0.10)
    # đặt dưới hai đường (bậc 1–2 có sai số lớn nên phía trên vùng cam đã có đường đi qua)
    ax.text(1.5, min(tr) * 1.15, "thiên lệch\ncao", ha="center", fontsize=11)
    ax.text(12.5, max(te) * 0.6, "phương sai\ncao", ha="center", fontsize=11)
    ax.set_xlabel("Bậc đa thức (độ phức tạp mô hình)")
    ax.set_ylabel("MSE (thang log)")
    ax.legend(loc="upper center")
    save(fig, "ch2_bias_variance")


def boundary(ax, model, X, y, title):
    xx, yy = np.meshgrid(np.linspace(X[:, 0].min() - .5, X[:, 0].max() + .5, 250),
                         np.linspace(X[:, 1].min() - .5, X[:, 1].max() + .5, 250))
    model.fit(X, y)
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=CM2, alpha=0.9)
    ax.scatter(X[:, 0], X[:, 1], c=[PT2[i] for i in y], s=10, edgecolor="white", linewidth=0.3)
    acc = (model.predict(X) == y).mean()
    ax.set_title(f"{title}\n(acc train {acc:.2f})".replace(".", ","), fontsize=10.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)


def decision_boundaries():
    X, y = make_moons(400, noise=0.28, random_state=0)
    models = [(LogisticRegression(), "Logistic Regression"), (KNeighborsClassifier(15), "KNN (k=15)"),
              (GaussianNB(), "Gaussian NB"), (DecisionTreeClassifier(max_depth=6), "Decision Tree"),
              (RandomForestClassifier(200, random_state=0), "Random Forest"),
              (XGBClassifier(n_estimators=200, max_depth=3), "XGBoost"),
              (SVC(kernel="linear"), "SVM tuyến tính"), (SVC(kernel="rbf", C=3), "SVM RBF"),
              (make_pipeline(PolynomialFeatures(3), LogisticRegression(max_iter=2000)), "Logistic + đa thức bậc 3")]
    fig, axes = plt.subplots(3, 3, figsize=(W, 7.6))
    # tiêu đề hai dòng: cần khoảng cách hàng rộng để không đè trục của hàng trên
    fig.subplots_adjust(hspace=0.42, wspace=0.12)
    for ax, (m, t) in zip(axes.ravel(), models):
        boundary(ax, m, X, y, t)
    save(fig, "ch2_boundaries")


def knn_k():
    X, y = make_moons(300, noise=0.35, random_state=2)
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.4))
    for ax, k in zip(axes, [1, 15, 75]):
        boundary(ax, KNeighborsClassifier(k), X, y, f"k = {k}")
    save(fig, "ch2_knn_k")


def svm_margin():
    X, y = make_blobs(80, centers=[(-1.6, -1.2), (1.6, 1.2)], cluster_std=0.85, random_state=5)
    fig, axes = plt.subplots(1, 2, figsize=(W, 3.0))
    for ax, Cc, t in zip(axes, [100, 0.05], ["Lề cứng (C = 100)", "Lề mềm (C = 0,05)"]):
        m = SVC(kernel="linear", C=Cc).fit(X, y)
        xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 200), np.linspace(-4, 4, 200))
        Z = m.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        ax.contourf(xx, yy, Z, levels=[-99, 0, 99], colors=["#DBEAFE", "#FEE2E2"], alpha=0.8)
        ax.contour(xx, yy, Z, levels=[-1, 0, 1], colors=[C["gray"], "k", C["gray"]], linestyles=["--", "-", "--"])
        ax.scatter(X[:, 0], X[:, 1], c=[PT2[i] for i in y], s=18, edgecolor="white")
        sv = m.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=110, facecolor="none", edgecolor=C["orange"], lw=2,
                   label=f"{len(sv)} vector hỗ trợ")
        ax.set_title(t, fontsize=11.5)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, "ch2_svm_margin")


def impurity():
    p = np.linspace(0.001, 0.999, 300)
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    fig, ax = plt.subplots(figsize=(W, 2.8))
    ax.plot(p, ent, label="Entropy H(p)")
    ax.plot(p, 2 * p * (1 - p), label="Gini = 1 − Σ pₖ²")
    ax.plot(p, 1 - np.maximum(p, 1 - p), label="Lỗi phân loại")
    ax.set_xlabel("Tỷ lệ lớp dương p trong nút")
    ax.set_ylabel("Độ vẩn đục")
    ax.legend()
    save(fig, "ch2_impurity")


def kmeans_iters():
    X, _ = make_blobs(300, centers=4, cluster_std=1.0, random_state=11)
    rng = np.random.default_rng(4)
    Cc = X[rng.choice(len(X), 4, replace=False)]
    fig, axes = plt.subplots(1, 4, figsize=(W, 1.95))
    cols = [C["blue"], C["orange"], C["green"], C["purple"]]
    for i, ax in enumerate(axes):
        lab = ((X[:, None] - Cc[None]) ** 2).sum(-1).argmin(1)
        ax.scatter(X[:, 0], X[:, 1], c=[cols[j] for j in lab], s=6)
        ax.scatter(Cc[:, 0], Cc[:, 1], marker="X", s=120, c="k", edgecolor="white")
        ax.set_title(f"Vòng {i}" if i else "Khởi tạo", fontsize=10.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        Cc = np.array([X[lab == j].mean(0) for j in range(4)])
    save(fig, "ch2_kmeans_iter")
    lab, Cc, inertia = kmeans_numpy(X, 4)
    return inertia


def pca_toy():
    rng = np.random.default_rng(0)
    X = rng.multivariate_normal([0, 0], [[3, 2.2], [2.2, 2]], 300)
    Xc = X - X.mean(0)
    _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.9))
    a.scatter(X[:, 0], X[:, 1], s=8, color=C["sky"], alpha=0.7)
    for v, s, c, n in zip(Vt, S, [C["red"], C["green"]], ["PC1", "PC2"]):
        sc = 2.2 * s / np.sqrt(len(X))
        a.annotate("", xy=v * sc, xytext=(0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2.5))
        a.text(*(v * sc * 1.15), n, color=c, fontsize=12, fontweight="bold",
               bbox=dict(fc="white", ec="none", alpha=0.85, pad=1.2))  # PC2 ngắn, nhãn nằm trong đám điểm
    a.set_aspect("equal")
    a.set_title("Dữ liệu gốc và hai trục chính")
    P = Xc @ Vt.T
    b.scatter(P[:, 0], np.zeros(len(P)) + rng.normal(0, 0.05, len(P)), s=8, color=C["purple"], alpha=0.6)
    b.set_yticks([])
    b.set_title(f"Chiếu lên PC1 ({100 * S[0] ** 2 / np.sum(S ** 2):.0f}% phương sai)")
    save(fig, "ch2_pca_toy")


def pla_softmax():
    s = load_json("ch2_scratch")
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 3.0))
    X, y, w = np.array(s["pla"]["X"]), np.array(s["pla"]["y"]), np.array(s["pla"]["w"])
    a.scatter(X[:, 0], X[:, 1], c=[PT2[int(v > 0)] for v in y], s=14)
    xs = np.linspace(X[:, 0].min(), X[:, 0].max(), 10)
    a.plot(xs, -(w[0] + w[1] * xs) / w[2], color="k", lw=2)
    a.set_title(f"PLA hội tụ sau {s['pla']['epochs']} epoch", fontsize=11.5)
    X3, y3, Wm = np.array(s["softmax"]["X"]), np.array(s["softmax"]["y"]), np.array(s["softmax"]["W"])
    xx, yy = np.meshgrid(np.linspace(X3[:, 0].min() - 1, X3[:, 0].max() + 1, 250),
                         np.linspace(X3[:, 1].min() - 1, X3[:, 1].max() + 1, 250))
    G = np.c_[np.ones(xx.size), xx.ravel(), yy.ravel()] @ Wm
    b.contourf(xx, yy, G.argmax(1).reshape(xx.shape), cmap=ListedColormap(["#DBEAFE", "#FEF3C7", "#DCFCE7"]))
    b.scatter(X3[:, 0], X3[:, 1], c=[[C["blue"], C["orange"], C["green"]][i] for i in y3], s=10)
    b.set_title(f"Softmax 3 lớp (acc {s['softmax']['acc']:.2f})".replace(".", ","), fontsize=11.5)
    for ax in (a, b):
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    save(fig, "ch2_pla_softmax")


if __name__ == "__main__":
    gd_paths()
    sigmoid_logloss()
    bias_variance()
    decision_boundaries()
    knn_k()
    svm_margin()
    impurity()
    kmeans_iters()
    pca_toy()
    pla_softmax()
