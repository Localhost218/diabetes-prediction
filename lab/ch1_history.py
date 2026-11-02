"""
Chương 1 — Lịch sử AI qua một bài toán: dự đoán đái tháo đường trên bộ Pima Indians Diabetes.

Mỗi mô hình đại diện cho một giai đoạn lịch sử, đánh giá cùng giao thức kiểm định chéo phân tầng 10 phần:
  1. Hệ chuyên gia (thập niên 1970–80): luật IF–THEN + hệ số chắc chắn kiểu MYCIN, không học từ dữ liệu
  2. Perceptron Rosenblatt (1958): thuật toán học perceptron (PLA), NumPy
  3. ADALINE Widrow–Hoff (1960): quy tắc delta / LMS, NumPy
  4. MLP + lan truyền ngược (1986): 1 lớp ẩn, sigmoid, NumPy
  5. Hồi quy logistic (học máy thống kê), 6. Gradient Boosting (thập niên 2000), scikit-learn

    .venv/bin/python lab/ch1_history.py
"""

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from common import DATA, MODELS, PALETTE, GREY, ACCENT, plt, r4, save_fig, save_results, seed_everything

COLS = ["pregnancies", "glucose", "blood_pressure", "skin_thickness", "insulin", "bmi", "pedigree", "age", "outcome"]
ZERO_IS_MISSING = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


# ------------------------------------------------------------------ 1. hệ chuyên gia
def mycin_combine(cfs):
    """Gộp các hệ số chắc chắn dương theo MYCIN: CF = CF1 + CF2·(1 − CF1)."""
    total = 0.0
    for cf in cfs:
        total = total + cf * (1 - total)
    return total


RULES = [  # (mô tả, điều kiện, CF) — tri thức chuyên gia viết tay theo ngưỡng ADA / WHO
    ("Glucose 2h ≥ 200 mg/dL (ngưỡng chẩn đoán ĐTĐ)", lambda r: r.glucose >= 200, 0.95),
    ("Glucose 2h 140–199 (rối loạn dung nạp)", lambda r: 140 <= r.glucose < 200, 0.55),
    ("BMI ≥ 30 (béo phì)", lambda r: r.bmi >= 30, 0.25),
    ("Tuổi ≥ 45", lambda r: r.age >= 45, 0.20),
    ("Chỉ số phả hệ ≥ 0,5 (tiền sử gia đình mạnh)", lambda r: r.pedigree >= 0.5, 0.20),
    ("Sinh ≥ 4 lần (nguy cơ ĐTĐ thai kỳ cũ)", lambda r: r.pregnancies >= 4, 0.10),
]


class ExpertSystem:
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return np.array([mycin_combine([cf for _, cond, cf in RULES if cond(r)]) for r in X.itertuples()])

    def explain(self, row) -> list:
        return [(desc, cf) for desc, cond, cf in RULES if cond(row)]


# ------------------------------------------------------------------ 2. perceptron
class Perceptron:
    """Rosenblatt (1958): ŷ = sign(w·x + b); sai thì w ← w + η·y·x. Nhãn y ∈ {−1, +1}."""

    def __init__(self, lr=0.01, epochs=100, seed=0):
        self.lr, self.epochs, self.rng = lr, epochs, np.random.default_rng(seed)

    def fit(self, X, y):
        y_pm = np.where(y == 1, 1, -1)
        self.w, self.b, self.errors = np.zeros(X.shape[1]), 0.0, []
        best = (np.inf, self.w.copy(), self.b)
        for _ in range(self.epochs):
            mistakes = 0
            for i in self.rng.permutation(len(X)):
                if y_pm[i] * (X[i] @ self.w + self.b) <= 0:
                    self.w += self.lr * y_pm[i] * X[i]
                    self.b += self.lr * y_pm[i]
                    mistakes += 1
            self.errors.append(mistakes)
            if mistakes < best[0]:  # "pocket": giữ trọng số tốt nhất vì dữ liệu không tách được tuyến tính
                best = (mistakes, self.w.copy(), self.b)
        _, self.w, self.b = best
        return self

    def decision_function(self, X):
        return X @ self.w + self.b


# ------------------------------------------------------------------ 3. ADALINE
class Adaline:
    """Widrow–Hoff (1960): cực tiểu MSE của đầu ra tuyến tính, w ← w + η·(y − w·x)·x."""

    def __init__(self, lr=0.001, epochs=200):
        self.lr, self.epochs = lr, epochs

    def fit(self, X, y):
        y_pm = np.where(y == 1, 1.0, -1.0)
        self.w, self.b, self.cost = np.zeros(X.shape[1]), 0.0, []
        for _ in range(self.epochs):
            err = y_pm - (X @ self.w + self.b)
            self.w += self.lr * X.T @ err
            self.b += self.lr * err.sum()
            self.cost.append(float((err ** 2).mean() / 2))
        return self

    def decision_function(self, X):
        return X @ self.w + self.b


# ------------------------------------------------------------------ 4. MLP lan truyền ngược
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


class MLPBackprop:
    """Rumelhart–Hinton–Williams (1986): x → sigmoid(W1·x + b1) → sigmoid(W2·h + b2), loss entropy chéo."""

    def __init__(self, hidden=16, lr=0.1, epochs=400, l2=1e-3, seed=0):
        self.hidden, self.lr, self.epochs, self.l2 = hidden, lr, epochs, l2
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        n, d = X.shape
        self.W1 = self.rng.normal(0, np.sqrt(1 / d), (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = self.rng.normal(0, np.sqrt(1 / self.hidden), (self.hidden, 1))
        self.b2 = np.zeros(1)
        self.loss = []
        y = y.reshape(-1, 1).astype(float)
        for _ in range(self.epochs):
            h = sigmoid(X @ self.W1 + self.b1)                  # lan truyền xuôi
            p = sigmoid(h @ self.W2 + self.b2)
            self.loss.append(float(-np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))))
            d2 = (p - y) / n                                    # ∂L/∂z2 cho sigmoid + entropy chéo
            d1 = (d2 @ self.W2.T) * h * (1 - h)                 # quy tắc chuỗi qua lớp ẩn
            self.W2 -= self.lr * (h.T @ d2 + self.l2 * self.W2)
            self.b2 -= self.lr * d2.sum(0)
            self.W1 -= self.lr * (X.T @ d1 + self.l2 * self.W1)
            self.b1 -= self.lr * d1.sum(0)
        return self

    def predict_proba(self, X):
        return sigmoid(sigmoid(X @ self.W1 + self.b1) @ self.W2 + self.b2).ravel()


# ------------------------------------------------------------------ dữ liệu
def load_pima() -> pd.DataFrame:
    df = pd.read_csv(DATA / "pima" / "pima.csv", header=None, names=COLS)
    return df


class Prep:
    """Số 0 ở các cột sinh lý = thiếu → điền trung vị của tập train; chuẩn hóa z-score."""

    def fit(self, X: pd.DataFrame):
        X = X.copy()
        X[ZERO_IS_MISSING] = X[ZERO_IS_MISSING].replace(0, np.nan)
        self.median = X.median()
        X = X.fillna(self.median)
        self.mean, self.std = X.mean(), X.std() + 1e-9
        return self

    def raw(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X[ZERO_IS_MISSING] = X[ZERO_IS_MISSING].replace(0, np.nan)
        return X.fillna(self.median)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return ((self.raw(X) - self.mean) / self.std).to_numpy()


def metrics(y, score, threshold):
    pred = (score >= threshold).astype(int)
    return {"accuracy": accuracy_score(y, pred), "roc_auc": roc_auc_score(y, score), "f1": f1_score(y, pred),
            "sensitivity": recall_score(y, pred), "specificity": recall_score(1 - y, 1 - pred)}


MODEL_NAMES = {
    "expert": "Hệ chuyên gia (luật + CF)", "perceptron": "Perceptron (1958)", "adaline": "ADALINE (1960)",
    "mlp": "MLP lan truyền ngược (1986)", "logreg": "Hồi quy logistic", "gbdt": "Gradient Boosting",
}


def cross_validate(df: pd.DataFrame, k=10):
    X, y = df[COLS[:-1]], df["outcome"].to_numpy()
    folds = StratifiedKFold(k, shuffle=True, random_state=42)
    per_model = {m: [] for m in MODEL_NAMES}
    for tr, te in folds.split(X, y):
        prep = Prep().fit(X.iloc[tr])
        Xtr, Xte = prep.transform(X.iloc[tr]), prep.transform(X.iloc[te])
        scores = {
            "expert": (ExpertSystem().predict_proba(prep.raw(X.iloc[te])), 0.5),
            "perceptron": (Perceptron().fit(Xtr, y[tr]).decision_function(Xte), 0.0),
            "adaline": (Adaline().fit(Xtr, y[tr]).decision_function(Xte), 0.0),
            "mlp": (MLPBackprop().fit(Xtr, y[tr]).predict_proba(Xte), 0.5),
            "logreg": (LogisticRegression(max_iter=1000).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1], 0.5),
            "gbdt": (GradientBoostingClassifier(random_state=0).fit(Xtr, y[tr]).predict_proba(Xte)[:, 1], 0.5),
        }
        for m, (s, thr) in scores.items():
            per_model[m].append(metrics(y[te], s, thr))
    summary = {}
    for m, rows in per_model.items():
        summary[m] = {k: {"mean": r4(np.mean([r[k] for r in rows])), "std": r4(np.std([r[k] for r in rows]))}
                      for k in rows[0]}
    return summary


def xor_experiment():
    """Minsky & Papert (1969): perceptron một lớp không học được XOR; MLP một lớp ẩn thì học được."""
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], float)
    y = np.array([0, 1, 1, 0])
    p = Perceptron(lr=0.1, epochs=50).fit(X, y)
    perc_acc = float(((p.decision_function(X) > 0).astype(int) == y).mean())
    m = MLPBackprop(hidden=4, lr=2.0, epochs=5000, l2=0, seed=3).fit(X, y)
    mlp_acc = float(((m.predict_proba(X) > 0.5).astype(int) == y).mean())
    return {"perceptron_acc": perc_acc, "mlp_acc": mlp_acc, "mlp_proba": m.predict_proba(X).round(3).tolist()}, m


# ------------------------------------------------------------------ hình
def plot_dataset(df):
    fig, axes = plt.subplots(2, 4, figsize=(10, 4.6))
    labels = {"pregnancies": "Số lần mang thai", "glucose": "Glucose 2h (mg/dL)", "blood_pressure": "HA tâm trương",
              "skin_thickness": "Nếp gấp da (mm)", "insulin": "Insulin 2h (µU/mL)", "bmi": "BMI",
              "pedigree": "Chỉ số phả hệ", "age": "Tuổi"}
    for ax, col in zip(axes.ravel(), COLS[:-1]):
        v = df[col].replace(0, np.nan) if col in ZERO_IS_MISSING else df[col]
        for cls, c, name in [(0, GREY, "Không ĐTĐ"), (1, ACCENT, "ĐTĐ")]:
            ax.hist(v[df.outcome == cls].dropna(), bins=25, alpha=0.6, color=c, label=name, density=True)
        ax.set_title(labels[col])
        ax.set_yticks([])
    axes[0, 0].legend()
    fig.suptitle("Phân bố 8 đặc trưng theo nhãn (số 0 ở cột sinh lý được coi là thiếu)", y=1.01)
    fig.tight_layout()
    save_fig(fig, "ch1_pima_distribution")


def plot_training(df):
    prep = Prep().fit(df[COLS[:-1]])
    X, y = prep.transform(df[COLS[:-1]]), df.outcome.to_numpy()
    p, a, m = Perceptron().fit(X, y), Adaline().fit(X, y), MLPBackprop().fit(X, y)
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
    axes[0].plot(np.array(p.errors) / len(X), color=PALETTE[0])
    axes[0].set(title="Perceptron: tỷ lệ phân loại sai mỗi epoch", xlabel="epoch")
    axes[1].plot(a.cost, color=PALETTE[1])
    axes[1].set(title="ADALINE: MSE/2", xlabel="epoch")
    axes[2].plot(m.loss, color=PALETTE[2])
    axes[2].set(title="MLP: entropy chéo", xlabel="epoch")
    fig.tight_layout()
    save_fig(fig, "ch1_training_curves")
    return prep, p, m


def plot_xor(mlp):
    xx, yy = np.meshgrid(np.linspace(-0.5, 1.5, 200), np.linspace(-0.5, 1.5, 200))
    zz = mlp.predict_proba(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    fig, ax = plt.subplots(figsize=(3.4, 3))
    ax.contourf(xx, yy, zz, levels=[0, 0.5, 1], colors=["#e5e7eb", "#bfdbfe"])
    ax.scatter([0, 1], [0, 1], s=80, c=GREY, label="XOR = 0")
    ax.scatter([0, 1], [1, 0], s=80, c=ACCENT, marker="s", label="XOR = 1")
    ax.set(title="Biên quyết định của MLP 2-4-1 trên XOR", xticks=[0, 1], yticks=[0, 1])
    ax.legend(loc="center", fontsize=8)
    save_fig(fig, "ch1_xor_mlp")


def plot_comparison(summary):
    keys = list(MODEL_NAMES)
    fig, ax = plt.subplots(figsize=(8, 3))
    x = np.arange(len(keys))
    for j, (met, name) in enumerate([("accuracy", "Accuracy"), ("roc_auc", "ROC-AUC"), ("f1", "F1")]):
        ax.bar(x + (j - 1) * 0.27, [summary[k][met]["mean"] for k in keys], 0.27,
               yerr=[summary[k][met]["std"] for k in keys], label=name, color=PALETTE[j], alpha=0.85, capsize=2)
    ax.set_xticks(x, [MODEL_NAMES[k].replace(" (", "\n(") for k in keys], fontsize=8)
    ax.set_ylim(0.4, 0.9)
    ax.legend(ncol=3)
    ax.set_title("Kiểm định chéo 10 phần trên Pima (trung bình ± độ lệch chuẩn)")
    save_fig(fig, "ch1_comparison")


def main():
    seed_everything()
    df = load_pima()
    zeros = {c: int((df[c] == 0).sum()) for c in ZERO_IS_MISSING}
    stats = df.describe().T[["mean", "std", "min", "max"]].round(2)
    print("Pima:", df.shape, "dương tính", df.outcome.mean().round(3), "số 0 =", zeros)
    summary = cross_validate(df)
    for m, s in summary.items():
        print(f"  {MODEL_NAMES[m]:32s} acc={s['accuracy']['mean']:.3f} auc={s['roc_auc']['mean']:.3f} "
              f"f1={s['f1']['mean']:.3f}")
    xor, mlp_xor = xor_experiment()
    print("XOR:", xor)

    plot_dataset(df)
    prep, perc, mlp = plot_training(df)
    plot_xor(mlp_xor)
    plot_comparison(summary)

    # lưu tham số dạng JSON thuần (không pickle lớp của script) để tầng deploy nạp được
    (MODELS / "ch1_pima.json").write_text(json.dumps({
        "columns": COLS[:-1], "zero_is_missing": ZERO_IS_MISSING,
        "median": prep.median.round(4).to_dict(), "mean": prep.mean.round(6).to_dict(),
        "std": prep.std.round(6).to_dict(),
        "perceptron": {"w": perc.w.round(6).tolist(), "b": round(float(perc.b), 6)},
        "mlp": {"W1": mlp.W1.round(6).tolist(), "b1": mlp.b1.round(6).tolist(),
                "W2": mlp.W2.round(6).tolist(), "b2": mlp.b2.round(6).tolist()},
        "rules": [(d, c) for d, _, c in RULES]}, ensure_ascii=False), encoding="utf-8")
    save_results("ch1", {
        "dataset": {"name": "Pima Indians Diabetes", "n": len(df), "n_features": 8,
                    "positive": int(df.outcome.sum()), "positive_rate": r4(df.outcome.mean()), "zeros": zeros,
                    "stats": stats.to_dict(orient="index"), "head": df.head(5).to_dict(orient="records")},
        "cv10": summary, "xor": xor,
        "perceptron_weights": dict(zip(COLS[:-1], np.round(perc.w, 3).tolist())),
    })


if __name__ == "__main__":
    main()
