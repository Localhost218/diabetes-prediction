"""TN1 — Dự đoán bệnh tiểu đường: khảo sát, làm sạch, chia tập, huấn luyện 9 mô hình + MLP, chọn ngưỡng, đánh giá.
Ghi outputs/metrics/ch2_diabetes.json, outputs/ch2_diabetes_preds.npz, models/ch2/diabetes_*.joblib|pt."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import limit_threads, seed_all  # noqa: E402

limit_threads(8)
seed_all()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_validate, learning_curve  # noqa: E402

from tl3 import ch2_data as cd  # noqa: E402
from tl3.ch2_metrics import best_threshold, clf_metrics  # noqa: E402
from tl3.ch2_mlp import fit_mlp, proba as mlp_proba  # noqa: E402
from tl3.ch2_models import CLF_NAMES, classifiers, clf_pipeline  # noqa: E402
from tl3.common import OUT, model_dir, save_json  # noqa: E402
from tl3.gpu import cap  # noqa: E402

M = model_dir("ch2")
SVM_SUBSET = 20000          # SVC RBF có độ phức tạp ~O(n²): huấn luyện trên tập con phân tầng


def survey(raw):
    """Thống kê mô tả trước làm sạch: kiểu dữ liệu, phân bố nhãn, hạng mục của biến phân loại."""
    return {"shape": list(raw.shape), "dtypes": {c: str(t) for c, t in raw.dtypes.items()},
            "label_counts": raw[cd.DIAB_TARGET].value_counts().sort_index().to_dict(),
            "pos_rate": float(raw[cd.DIAB_TARGET].mean()),
            "cats": {c: raw[c].value_counts().to_dict() for c in cd.DIAB_CAT},
            "describe": raw[cd.DIAB_NUM].describe().T.round(3).to_dict(orient="index")}


def fit_classic(name, model, pre, X_tr, y_tr):
    """Huấn luyện một Pipeline (tiền xử lý + mô hình); SVM dùng tập con phân tầng."""
    pipe = clf_pipeline(pre, model)
    if name == "svm":
        rng = np.random.default_rng(42)
        idx = np.concatenate([rng.choice(np.where(y_tr == c)[0], int(SVM_SUBSET * np.mean(y_tr == c)),
                                         replace=False) for c in (0, 1)])
        X_tr, y_tr = X_tr.iloc[idx], y_tr[idx]
    t0 = time.perf_counter()
    pipe.fit(X_tr, y_tr)
    return pipe, time.perf_counter() - t0


def main():
    raw = cd.load_diabetes_raw()
    res = {"survey": survey(raw), "audit_raw": cd.audit(raw, ["age", "bmi", "HbA1c_level", "blood_glucose_level"])}
    df, (X_tr, X_va, X_te, y_tr, y_va, y_te) = cd.diabetes_splits()
    res["clean"] = {"rows": len(df), "n_rows": len(df), "removed": len(raw) - len(df), "pos": int(df.diabetes.sum()),
                    "neg": int((df.diabetes == 0).sum()), "pos_rate": float(df.diabetes.mean()),
                    "rate_by_smoking": df.groupby("smoking_history").diabetes.mean().round(4).to_dict(),
                    "rate_by_gender": df.groupby("gender").diabetes.mean().round(4).to_dict(),
                    "rate_hyper": df.groupby("hypertension").diabetes.mean().round(4).to_dict(),
                    "rate_heart": df.groupby("heart_disease").diabetes.mean().round(4).to_dict(),
                    "corr_target": df[cd.DIAB_NUM + ["diabetes"]].corr()["diabetes"].drop("diabetes").round(4).to_dict()}
    pre = cd.diabetes_preprocessor().fit(X_tr)
    feats = cd.feature_names(pre)
    res["split"] = {"train": len(y_tr), "val": len(y_va), "test": len(y_te), "n_features": len(feats),
                    "features": feats, "pos_train": float(y_tr.mean()), "pos_val": float(y_va.mean()),
                    "pos_test": float(y_te.mean()), "example_vector": pre.transform(X_tr.iloc[[0]])[0].round(3).tolist()}

    pos_w = float((y_tr == 0).sum() / (y_tr == 1).sum())
    models, preds, res["models"] = {}, {}, {}
    for name, model in classifiers(pos_w).items():
        pipe, secs = fit_classic(name, model, cd.diabetes_preprocessor(), X_tr, y_tr)
        p_va, p_te = pipe.predict_proba(X_va)[:, 1], pipe.predict_proba(X_te)[:, 1]
        models[name], preds[name] = pipe, (p_va, p_te)
        res["models"][name] = {"train_s": secs, "n_train": len(y_tr) if name != "svm" else SVM_SUBSET}
        print(f"{name:9s} {secs:7.1f}s  AUC val={clf_metrics(y_va, p_va)['auc']:.4f}", flush=True)

    # MLP PyTorch: cùng bộ tiền xử lý, BCE có pos_weight để bù mất cân bằng
    cap(0.15)
    Xtr_np, Xva_np, Xte_np = (pre.transform(x).astype(np.float32) for x in (X_tr, X_va, X_te))
    mlp, info = fit_mlp(Xtr_np, y_tr.astype(np.float32), Xva_np, y_va.astype(np.float32), task="clf",
                        pos_weight=None)
    preds["mlp"] = (mlp_proba(mlp, Xva_np), mlp_proba(mlp, Xte_np))
    res["models"]["mlp"] = {"train_s": info["train_s"], "n_train": len(y_tr), "epochs": info["epochs_run"],
                            "n_params": info["n_params"], "history": info["history"]}
    torch.save(mlp.state_dict(), M / "diabetes_mlp.pt")
    joblib.dump(pre, M / "diabetes_pre.joblib")

    # chọn ngưỡng trên Validation, rồi báo cáo Validation (0,5 và ngưỡng tối ưu) và Test (ngưỡng tối ưu)
    res["val05"], res["val"], res["test"], res["thr_curve"] = {}, {}, {}, {}
    for name, (p_va, p_te) in preds.items():
        thr, curve = best_threshold(y_va, p_va) if name != "baseline" else (0.5, None)
        res["val05"][name] = clf_metrics(y_va, p_va, 0.5)
        res["val"][name] = clf_metrics(y_va, p_va, thr)
        res["test"][name] = clf_metrics(y_te, p_te, thr)
        if curve:
            res["thr_curve"][name] = curve
    ranked = sorted((n for n in preds if n != "baseline"), key=lambda n: -res["val"][n]["f1"])
    best = ranked[0]
    res["ranking"] = ranked
    res["best"] = {"name": best, "label": CLF_NAMES[best], **{k: res["test"][best][k] for k in
                   ("acc", "prec", "rec", "f1", "auc", "ap", "thr")}}

    # kiểm định chéo 5-fold trên Train cho 4 mô hình đại diện
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    res["cv"] = {}
    for name in ("logreg", "tree", "rf", "xgb"):
        s = cross_validate(clf_pipeline(cd.diabetes_preprocessor(), classifiers(pos_w)[name]), X_tr, y_tr, cv=cv,
                           scoring=["roc_auc", "f1", "accuracy"], n_jobs=1)
        res["cv"][name] = {k: {"mean": float(np.mean(v)), "std": float(np.std(v))}
                           for k, v in s.items() if k.startswith("test_")}

    # đường cong kiểm định: độ sâu cây và k của KNN (minh hoạ thiên lệch - phương sai)
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier
    Ztr, Zva = pre.transform(X_tr), pre.transform(X_va)
    depth = {"depth": [], "train_auc": [], "val_auc": []}
    for d in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25]:
        t = DecisionTreeClassifier(max_depth=d, random_state=42).fit(Ztr, y_tr)
        depth["depth"].append(d)
        depth["train_auc"].append(roc_auc_score(y_tr, t.predict_proba(Ztr)[:, 1]))
        depth["val_auc"].append(roc_auc_score(y_va, t.predict_proba(Zva)[:, 1]))
    res["curve_depth"] = depth
    knn = {"k": [], "val_f1": [], "val_auc": []}
    sub = np.random.default_rng(0).choice(len(y_tr), 30000, replace=False)
    for k in [1, 3, 5, 9, 15, 25, 41, 61, 101]:
        m = KNeighborsClassifier(k, weights="distance", n_jobs=8).fit(Ztr[sub], y_tr[sub])
        p = m.predict_proba(Zva)[:, 1]
        knn["k"].append(k)
        knn["val_f1"].append(f1_score(y_va, (p >= 0.5).astype(int)))
        knn["val_auc"].append(roc_auc_score(y_va, p))
    res["curve_knn"] = knn

    # đường học của mô hình tốt nhất (nếu là MLP thì dùng XGBoost đại diện cho nhóm cây)
    lc_name = best if best not in ("mlp", "svm", "knn") else "xgb"
    sizes, tr_s, va_s = learning_curve(clf_pipeline(cd.diabetes_preprocessor(), classifiers(pos_w)[lc_name]),
                                       X_tr, y_tr, train_sizes=np.linspace(0.05, 1.0, 8), cv=cv, scoring="f1")
    res["learning_curve"] = {"model": lc_name, "sizes": sizes.tolist(), "train": tr_s.mean(1).tolist(),
                             "val": va_s.mean(1).tolist(), "val_std": va_s.std(1).tolist()}

    # tầm quan trọng đặc trưng: hệ số LogReg, impurity RF, gain XGB, hoán vị trên Validation cho mô hình tốt nhất
    imp_name = best if best != "mlp" else "xgb"
    pi = permutation_importance(models[imp_name], X_va, y_va, scoring="roc_auc", n_repeats=5, random_state=42,
                                n_jobs=1)
    res["importance"] = {
        "logreg_coef": dict(zip(feats, models["logreg"]["model"].coef_[0].round(4).tolist())),
        "rf": dict(zip(feats, models["rf"]["model"].feature_importances_.round(4).tolist())),
        "xgb": dict(zip(feats, models["xgb"]["model"].feature_importances_.round(4).tolist())),
        "perm_model": imp_name,
        "perm": {c: {"mean": float(m), "std": float(s)} for c, m, s in
                 zip(X_va.columns, pi.importances_mean, pi.importances_std)},
    }
    for name in ("logreg", "rf", "xgb", "lgbm", "tree", "nb", "knn"):
        joblib.dump(models[name], M / f"diabetes_{name}.joblib", compress=3)
    res["thresholds"] = {n: res["val"][n]["thr"] for n in preds}
    res["pos_weight"] = pos_w
    joblib.dump({"thresholds": res["thresholds"], "features": feats}, M / "diabetes_meta.joblib")
    np.savez_compressed(OUT / "ch2_diabetes_preds.npz", y_va=y_va, y_te=y_te,
                        **{f"{n}_va": p[0] for n, p in preds.items()}, **{f"{n}_te": p[1] for n, p in preds.items()})
    save_json("ch2_diabetes", res)
    print("best", best, res["best"])


if __name__ == "__main__":
    main()
