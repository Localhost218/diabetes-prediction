"""
Chương 2 — Các kỹ thuật học máy cơ bản trên NHANES 2011–2018 (dữ liệu thật của CDC, 20.646 người ≥ 20 tuổi).

  A. Phân loại ĐTĐ (nhãn theo xét nghiệm + chẩn đoán): LR, KNN, Naive Bayes, cây quyết định, rừng ngẫu nhiên,
     Gradient Boosting, SVM (RBF), MLP — cùng 13 đặc trưng tự khai được mà chatbot dùng.
  B. Hồi quy HbA1c (%): hồi quy tuyến tính, Ridge, SVR, rừng ngẫu nhiên.
  C. Không giám sát: K-means + PCA.

Chia 70/10/20 phân tầng, random_state=42 — trùng với scripts/train_model.py để số liệu khớp mô hình đang chạy.

    .venv/bin/python lab/ch2_ml.py
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, f1_score,
                             mean_absolute_error, mean_squared_error, precision_recall_curve, r2_score,
                             roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, plot_tree

from common import ACCENT, GREY, MODELS, PALETTE, ROOT, plt, r4, save_fig, save_results, seed_everything
from src.models.datasets import NHANES_FEATURES, NHANES_VI

LABEL = "diabetes"


def imputed(model):
    """Điền trung vị + cột chỉ báo thiếu, chuẩn hóa z-score, rồi tới mô hình."""
    return make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler(), model)


def classifiers():
    return {
        "logreg": ("Hồi quy logistic", imputed(LogisticRegression(max_iter=3000, C=1.0))),
        "knn": ("KNN (k=51)", imputed(KNeighborsClassifier(n_neighbors=51, weights="distance"))),
        "nb": ("Naive Bayes Gauss", imputed(GaussianNB())),
        "tree": ("Cây quyết định (sâu 6)", imputed(DecisionTreeClassifier(max_depth=6, min_samples_leaf=40,
                                                                          random_state=0))),
        "rf": ("Rừng ngẫu nhiên", imputed(RandomForestClassifier(400, min_samples_leaf=10, n_jobs=-1,
                                                                 random_state=0))),
        "gbdt": ("Gradient Boosting", HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                                                      random_state=0)),
        # C, gamma, class_weight chọn trên tập validation (svm_grid); balanced bù lệch lớp 80/20
        "svm": ("SVM (RBF)", imputed(SVC(C=1.0, gamma=0.01, class_weight="balanced", probability=True,
                                         random_state=0))),
        "mlp": ("MLP (64-32)", imputed(MLPClassifier((64, 32), alpha=1e-3, early_stopping=True, max_iter=300,
                                                     random_state=0))),
    }


def score(model, X):
    """Điểm xếp hạng: SVM dùng decision_function (Platt scaling của predict_proba được fit bằng CV nội bộ
    nên có thể đảo thứ tự), đưa qua sigmoid để về [0, 1]; mô hình khác dùng predict_proba."""
    if hasattr(model, "steps") and isinstance(model[-1], SVC):
        return 1 / (1 + np.exp(-model.decision_function(X)))
    return model.predict_proba(X)[:, 1]


def best_threshold(y, p):
    """Ngưỡng Youden J = argmax(TPR − FPR), chọn trên tập validation."""
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def run_classification(df):
    X, y = df[NHANES_FEATURES].astype(float), df[LABEL].astype(int)
    X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_val, y_tr, y_val = train_test_split(X_tmp, y_tmp, test_size=0.125, stratify=y_tmp, random_state=42)
    results, probas, fitted = {}, {}, {}
    for key, (name, model) in classifiers().items():
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_s = time.perf_counter() - t0
        thr = best_threshold(y_val, score(model, X_val))
        t0 = time.perf_counter()
        p = score(model, X_te)
        infer_ms = (time.perf_counter() - t0) / len(X_te) * 1000
        pred = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
        results[key] = {"name": name, "roc_auc": r4(roc_auc_score(y_te, p)),
                        "pr_auc": r4(average_precision_score(y_te, p)), "brier": r4(brier_score_loss(y_te, p)),
                        "accuracy": r4(accuracy_score(y_te, pred)), "f1": r4(f1_score(y_te, pred)),
                        "sensitivity": r4(tp / (tp + fn)), "specificity": r4(tn / (tn + fp)), "threshold": r4(thr),
                        "fit_s": round(fit_s, 2), "infer_ms_per_sample": round(infer_ms, 4),
                        "confusion": [[int(tn), int(fp)], [int(fn), int(tp)]]}
        probas[key], fitted[key] = p, model
        print(f"  {name:24s} AUC={results[key]['roc_auc']:.3f} PR={results[key]['pr_auc']:.3f} "
              f"F1={results[key]['f1']:.3f} fit={fit_s:.1f}s")

    # kiểm định chéo 5 phần trên tập train cho 3 mô hình đại diện — độ ổn định
    cv = {}
    for key in ["logreg", "tree", "gbdt"]:
        s = cross_val_score(classifiers()[key][1], X_tr, y_tr, cv=StratifiedKFold(5, shuffle=True, random_state=0),
                            scoring="roc_auc", n_jobs=5)
        cv[key] = {"mean": r4(s.mean()), "std": r4(s.std())}

    # KNN: ảnh hưởng của k (đường cong thiếu khớp / quá khớp)
    k_curve = []
    for k in [1, 3, 5, 11, 21, 51, 101, 201]:
        m = imputed(KNeighborsClassifier(n_neighbors=k)).fit(X_tr, y_tr)
        k_curve.append({"k": k, "train_auc": r4(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
                        "val_auc": r4(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))})
    # cây quyết định: độ sâu
    depth_curve = []
    for dpt in [1, 2, 3, 4, 6, 8, 12, 16, None]:
        m = imputed(DecisionTreeClassifier(max_depth=dpt, random_state=0)).fit(X_tr, y_tr)
        depth_curve.append({"depth": dpt or 0, "train_auc": r4(roc_auc_score(y_tr, m.predict_proba(X_tr)[:, 1])),
                            "val_auc": r4(roc_auc_score(y_val, m.predict_proba(X_val)[:, 1]))})

    svm_grid = []
    for C, cw, g in [(1, None, "scale"), (1, "balanced", "scale"), (0.1, "balanced", "scale"),
                     (1, "balanced", 0.01), (10, "balanced", 0.01)]:
        m = imputed(SVC(C=C, gamma=g, class_weight=cw)).fit(X_tr, y_tr)
        svm_grid.append({"C": C, "class_weight": cw or "none", "gamma": g, "n_sv": int(m[-1].n_support_.sum()),
                         "val_auc": r4(roc_auc_score(y_val, m.decision_function(X_val)))})

    imp = permutation_importance(fitted["gbdt"], X_te, y_te, scoring="roc_auc", n_repeats=5, random_state=0,
                                 n_jobs=-1)
    importance = sorted(zip(NHANES_FEATURES, imp.importances_mean), key=lambda t: -t[1])
    lr = fitted["logreg"][-1]
    n = len(NHANES_FEATURES)
    odds = {f: r4(np.exp(c)) for f, c in zip(NHANES_FEATURES, lr.coef_[0][:n])}  # OR trên 1 độ lệch chuẩn

    split = {"train": len(X_tr), "val": len(X_val), "test": len(X_te)}
    return results, probas, fitted, y_te, X_te, cv, k_curve, depth_curve, importance, odds, split, svm_grid


def run_regression(df):
    d = df.dropna(subset=["hba1c"])
    X, y = d[NHANES_FEATURES].astype(float), d["hba1c"].astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    models = {"linreg": ("Hồi quy tuyến tính", imputed(LinearRegression())),
              "ridge": ("Ridge (α=10)", imputed(Ridge(alpha=10))),
              "svr": ("SVR (RBF, ε=0,1)", imputed(SVR(C=1.0, epsilon=0.1))),
              "rf": ("Rừng ngẫu nhiên", imputed(RandomForestRegressor(300, min_samples_leaf=10, n_jobs=-1,
                                                                      random_state=0)))}
    out, preds = {}, {}
    sub = np.random.default_rng(0).permutation(len(X_tr))[:8000]  # SVR O(n²): huấn luyện trên 8.000 mẫu
    for key, (name, m) in models.items():
        Xf, yf = (X_tr.iloc[sub], y_tr.iloc[sub]) if key == "svr" else (X_tr, y_tr)
        m.fit(Xf, yf)
        p = m.predict(X_te)
        preds[key] = p
        out[key] = {"name": name, "rmse": r4(np.sqrt(mean_squared_error(y_te, p))),
                    "mae": r4(mean_absolute_error(y_te, p)), "r2": r4(r2_score(y_te, p))}
        print(f"  {name:24s} RMSE={out[key]['rmse']:.3f} MAE={out[key]['mae']:.3f} R²={out[key]['r2']:.3f}")
    baseline = np.full(len(y_te), y_tr.mean())
    out["mean"] = {"name": "Dự đoán bằng trung bình", "rmse": r4(np.sqrt(mean_squared_error(y_te, baseline))),
                   "mae": r4(mean_absolute_error(y_te, baseline)), "r2": 0.0}
    coefs = dict(zip(NHANES_FEATURES, np.round(models["linreg"][1][-1].coef_[:len(NHANES_FEATURES)], 4).tolist()))
    return out, preds, y_te, coefs, len(d)


def run_clustering(df):
    cols = ["age_years", "bmi", "waist_cm", "systolic_bp", "diastolic_bp"]
    d = df.dropna(subset=cols)
    Z = StandardScaler().fit_transform(d[cols])
    rows = []
    for k in range(2, 9):
        km = KMeans(k, n_init=10, random_state=0).fit(Z)
        sample = np.random.default_rng(0).permutation(len(Z))[:5000]
        rows.append({"k": k, "inertia": r4(km.inertia_), "silhouette": r4(silhouette_score(Z[sample],
                                                                                          km.labels_[sample]))})
    km = KMeans(4, n_init=10, random_state=0).fit(Z)
    d = d.assign(cluster=km.labels_)
    profile = d.groupby("cluster").agg(n=(LABEL, "size"), diabetes_rate=(LABEL, "mean"),
                                       **{c: (c, "mean") for c in cols}).round(3)
    pca = PCA(2).fit(Z)
    return rows, profile, pca.transform(Z), km.labels_, d[LABEL].to_numpy(), pca.explained_variance_ratio_


# ------------------------------------------------------------------ hình
def plot_data(df):
    fig, axes = plt.subplots(1, 3, figsize=(10, 2.9))
    bins = [20, 30, 40, 50, 60, 70, 81]
    g = df.groupby(pd.cut(df.age_years, bins, right=False), observed=True)[LABEL].mean() * 100
    axes[0].bar(range(len(g)), g.values, color=ACCENT)
    axes[0].set_xticks(range(len(g)), ["20–29", "30–39", "40–49", "50–59", "60–69", "≥70"], fontsize=8)
    axes[0].set(title="Tỷ lệ ĐTĐ theo nhóm tuổi (%)")
    for cls, c, name in [(0, GREY, "Không ĐTĐ"), (1, ACCENT, "ĐTĐ")]:
        axes[1].hist(df.bmi[df[LABEL] == cls].dropna(), bins=40, density=True, alpha=0.6, color=c, label=name)
    axes[1].set(title="Phân bố BMI theo nhãn", xlim=(14, 60))
    axes[1].legend()
    miss = df[NHANES_FEATURES].isna().mean().sort_values(ascending=False) * 100
    axes[2].barh([NHANES_VI[c] for c in miss.index[:7]][::-1], miss.values[:7][::-1], color=GREY)
    axes[2].set(title="Tỷ lệ thiếu dữ liệu (%)")
    axes[2].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_fig(fig, "ch2_nhanes_eda")

    corr = df[NHANES_FEATURES + [LABEL]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    names = [c.replace("_", " ") for c in corr.columns]
    ax.set_xticks(range(len(names)), names, rotation=70, fontsize=7.5)
    ax.set_yticks(range(len(names)), names, fontsize=7.5)
    ax.grid(False)
    fig.colorbar(im, shrink=0.8)
    ax.set_title("Ma trận tương quan Pearson")
    save_fig(fig, "ch2_nhanes_corr")


def plot_roc_pr(results, probas, y_te):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for i, (key, p) in enumerate(probas.items()):
        fpr, tpr, _ = roc_curve(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        lab = f"{results[key]['name']} ({results[key]['roc_auc']:.3f})"
        axes[0].plot(fpr, tpr, color=PALETTE[i], lw=1.2, label=lab)
        axes[1].plot(rc, pr, color=PALETTE[i], lw=1.2, label=f"{results[key]['name']} ({results[key]['pr_auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color=GREY, lw=0.8)
    axes[0].set(title="Đường ROC (AUC)", xlabel="FPR", ylabel="TPR")
    axes[1].axhline(y_te.mean(), ls="--", color=GREY, lw=0.8)
    axes[1].set(title="Đường Precision–Recall (AP)", xlabel="Recall", ylabel="Precision")
    axes[0].legend(fontsize=7)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    save_fig(fig, "ch2_roc_pr")


def plot_curves(k_curve, depth_curve):
    fig, axes = plt.subplots(1, 2, figsize=(9, 2.9))
    ks = [r["k"] for r in k_curve]
    axes[0].semilogx(ks, [r["train_auc"] for r in k_curve], "o-", color=GREY, label="train")
    axes[0].semilogx(ks, [r["val_auc"] for r in k_curve], "o-", color=ACCENT, label="validation")
    axes[0].set(title="KNN: AUC theo k", xlabel="k (thang log)")
    axes[0].legend()
    ds = [str(r["depth"] or "∞") for r in depth_curve]
    axes[1].plot(ds, [r["train_auc"] for r in depth_curve], "o-", color=GREY, label="train")
    axes[1].plot(ds, [r["val_auc"] for r in depth_curve], "o-", color=ACCENT, label="validation")
    axes[1].set(title="Cây quyết định: AUC theo độ sâu", xlabel="max_depth")
    axes[1].legend()
    fig.tight_layout()
    save_fig(fig, "ch2_bias_variance")


def plot_tree_fig(model):
    fig, ax = plt.subplots(figsize=(11, 4.2))
    names = [n.replace("missingindicator_", "thiếu_") for n in model[:-1].get_feature_names_out(NHANES_FEATURES)]
    plot_tree(model[-1], max_depth=2, feature_names=names, class_names=["không", "ĐTĐ"],
              filled=True, impurity=True, proportion=True, fontsize=8, ax=ax, precision=2)
    save_fig(fig, "ch2_tree")


def plot_importance_cm(importance, results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [1.6, 1]})
    names = [NHANES_VI[f] for f, _ in importance][::-1]
    axes[0].barh(names, [v for _, v in importance][::-1], color=ACCENT)
    axes[0].set(title="Độ quan trọng hoán vị (giảm ROC-AUC) — Gradient Boosting")
    axes[0].tick_params(axis="y", labelsize=8)
    cm = np.array(results["gbdt"]["confusion"])
    axes[1].imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        axes[1].text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    axes[1].set_xticks([0, 1], ["Dự đoán 0", "Dự đoán 1"])
    axes[1].set_yticks([0, 1], ["Thật 0", "Thật 1"])
    axes[1].grid(False)
    axes[1].set_title(f"Ma trận nhầm lẫn GBDT (ngưỡng {results['gbdt']['threshold']:.2f})")
    fig.tight_layout()
    save_fig(fig, "ch2_importance_cm")


def plot_regression(preds, y_te, reg):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for ax, key in zip(axes, ["linreg", "rf"]):
        ax.scatter(y_te, preds[key], s=3, alpha=0.25, color=ACCENT)
        ax.plot([4, 14], [4, 14], "--", color=GREY)
        ax.set(xlim=(4, 12), ylim=(4, 12), xlabel="HbA1c thật (%)", ylabel="HbA1c dự đoán (%)",
               title=f"{reg[key]['name']}: R² = {reg[key]['r2']:.3f}")
    fig.tight_layout()
    save_fig(fig, "ch2_regression")


def plot_kmeans(rows, proj, labels, y, evr):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3))
    ks = [r["k"] for r in rows]
    axes[0].plot(ks, [r["inertia"] for r in rows], "o-", color=ACCENT)
    axes[0].set(title="Phương pháp khuỷu tay (inertia)", xlabel="k")
    axes[1].plot(ks, [r["silhouette"] for r in rows], "o-", color=PALETTE[1])
    axes[1].set(title="Hệ số silhouette", xlabel="k")
    idx = np.random.default_rng(0).permutation(len(proj))[:4000]
    for c in range(4):
        m = idx[labels[idx] == c]
        axes[2].scatter(proj[m, 0], proj[m, 1], s=3, alpha=0.5, color=PALETTE[c], label=f"cụm {c}")
    axes[2].set(title=f"K-means k=4 trên PCA ({evr.sum() * 100:.0f}% phương sai)", xlabel="PC1", ylabel="PC2")
    axes[2].legend(markerscale=4, fontsize=7)
    fig.tight_layout()
    save_fig(fig, "ch2_kmeans")


def main():
    seed_everything()
    df = pd.read_csv(ROOT / "data" / "nhanes" / "nhanes_diabetes.csv")
    print(f"NHANES: {len(df):,} người, ĐTĐ {df[LABEL].mean():.1%}")
    print("A. Phân loại")
    (results, probas, fitted, y_te, X_te, cv, k_curve, depth_curve, importance, odds,
     split, svm_grid) = run_classification(df)
    print("B. Hồi quy HbA1c")
    reg, preds, y_reg, coefs, n_reg = run_regression(df)
    print("C. Gom cụm")
    km_rows, km_profile, proj, labels, y_km, evr = run_clustering(df)
    print(km_profile)

    plot_data(df)
    plot_roc_pr(results, probas, y_te)
    plot_curves(k_curve, depth_curve)
    plot_tree_fig(fitted["tree"])
    plot_importance_cm(importance, results)
    plot_regression(preds, y_reg, reg)
    plot_kmeans(km_rows, proj, labels, y_km, evr)

    joblib.dump({k: fitted[k] for k in ["logreg", "knn", "nb", "tree", "rf", "gbdt", "svm"]},
                MODELS / "ch2_classic.joblib", compress=3)
    desc = df[NHANES_FEATURES + ["hba1c", "fasting_glucose"]].describe().T.round(2)
    save_results("ch2", {
        "dataset": {"name": "NHANES 2011–2018", "n": len(df), "positive_rate": r4(df[LABEL].mean()),
                    "n_asian": int(df.asian.sum()), "undiagnosed": int(df.undiagnosed.sum()),
                    "diagnosed": int(df.diagnosed.sum()), "prediabetes_rate": r4(df.prediabetes.mean()),
                    "cycles": df.cycle.value_counts().sort_index().to_dict(), "split": split,
                    "missing": {c: r4(df[c].isna().mean()) for c in NHANES_FEATURES},
                    "describe": desc[["count", "mean", "std", "min", "50%", "max"]].to_dict(orient="index"),
                    "head": df[["seqn", "age_years", "sex", "bmi", "waist_cm", "systolic_bp", "high_bp",
                                "family_history", "hba1c", "fasting_glucose", LABEL]].head(5).to_dict("records")},
        "classification": results, "cv5_train": cv, "knn_k_curve": k_curve, "tree_depth_curve": depth_curve,
        "svm_grid": svm_grid,
        "permutation_importance": [(f, r4(v)) for f, v in importance], "logreg_odds_ratio_per_sd": odds,
        "regression": reg, "regression_n": n_reg, "linreg_coef_per_sd": coefs,
        "kmeans": km_rows, "kmeans_profile": km_profile.reset_index().to_dict("records"),
        "pca_explained": [r4(v) for v in evr],
    })


if __name__ == "__main__":
    main()
