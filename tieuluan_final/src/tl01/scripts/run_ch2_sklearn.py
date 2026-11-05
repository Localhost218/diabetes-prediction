"""Mục 2.3 - tiến trình CPU: huấn luyện các mô hình học máy cơ bản trên ba tập dữ liệu.

Kết quả: outputs/metrics/ch2_sklearn.json, mô hình ở models/ch2/*.joblib, dự đoán trên tập kiểm thử
ở outputs/metrics/ch2_pred_*.npz (để vẽ ma trận nhầm lẫn và đồ thị thực tế - dự báo).
"""
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl import data_tabular as dt  # noqa: E402
from tl import metrics as M  # noqa: E402
from tl.common import METRICS, MODELS, SEED, file_kb, log, merge_json, seed_all  # noqa: E402
from tl.ml_classic import (build_linear_regression_pipeline, build_logistic_regression_pipeline,  # noqa: E402
                           build_rf_pipeline, build_svm_pipeline)

OUT = MODELS / "ch2"
OUT.mkdir(parents=True, exist_ok=True)
LOGN = "ch2_sklearn"


def run(ds, name, pipe, Xtr, ytr, Xte, yte, kind, preds):
    log(f"{ds}/{name}: huấn luyện trên {len(ytr)} mẫu", LOGN)
    t0 = time.perf_counter()
    pipe.fit(Xtr, ytr)
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    yp = pipe.predict(Xte)
    infer_ms = (time.perf_counter() - t0) / len(yte) * 1000
    if kind == "reg":
        res = M.regression(yte, yp)
    else:
        score = None
        if kind == "bin" and hasattr(pipe, "predict_proba"):
            score = pipe.predict_proba(Xte)[:, 1]
        elif kind == "bin":
            score = pipe.decision_function(Xte)
        res = M.classification(yte, yp, score, binary=kind == "bin")
    path = OUT / f"{ds}_{name}.joblib"
    joblib.dump(pipe, path, compress=3)
    res.update(train_s=train_s, infer_ms=infer_ms, size_kb=file_kb(path))
    preds[name] = yp
    merge_json(LOGN, f"{ds}/{name}", res)
    log(f"{ds}/{name}: {({k: round(v, 4) for k, v in res.items() if isinstance(v, float)})}", LOGN)


def main():
    seed_all()
    # --- Tiểu đường: 80/20 phân tầng, cùng seed với các mô hình MLP
    df = dt.diabetes_frame()
    X, y = df.drop(columns="diabetes"), df.diabetes.to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    p = {}
    run("diabetes", "logreg", build_logistic_regression_pipeline(dt.diabetes_preprocessor()), Xtr, ytr, Xte, yte, "bin", p)
    run("diabetes", "rf", build_rf_pipeline(dt.diabetes_preprocessor()), Xtr, ytr, Xte, yte, "bin", p)
    run("diabetes", "svm", build_svm_pipeline(dt.diabetes_preprocessor()), Xtr, ytr, Xte, yte, "bin", p)
    np.savez(METRICS / "ch2_pred_diabetes_sklearn.npz", y_test=yte, **p)

    # --- Giá nhà
    df = dt.house_frame()
    X, y = df[dt.HOUSE_NUM + dt.HOUSE_CAT], df.Price.to_numpy(np.float64)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    p = {}
    run("house", "ols", build_linear_regression_pipeline(dt.house_preprocessor()), Xtr, ytr, Xte, yte, "reg", p)
    run("house", "ridge", build_linear_regression_pipeline(dt.house_preprocessor(), alpha=1.0), Xtr, ytr, Xte, yte, "reg", p)
    run("house", "rf", build_rf_pipeline(dt.house_preprocessor(), task="regression"), Xtr, ytr, Xte, yte, "reg", p)
    run("house", "svr", build_svm_pipeline(dt.house_preprocessor(), task="regression", C=3.0), Xtr, ytr, Xte, yte, "reg", p)
    np.savez(METRICS / "ch2_pred_house_sklearn.npz", y_test=yte, **p)

    # --- Nhận xét khách hàng
    df = dt.reviews_frame()
    Xtr, Xte, ytr, yte = train_test_split(df.text, df.sentiment.to_numpy(), test_size=0.2,
                                          stratify=df.sentiment, random_state=SEED)
    p = {}
    run("reviews", "logreg", build_logistic_regression_pipeline(dt.tfidf_vectorizer()), Xtr, ytr, Xte, yte, "multi", p)
    run("reviews", "linsvm", build_svm_pipeline(dt.tfidf_vectorizer(), kernel="linear", C=0.5), Xtr, ytr, Xte, yte, "multi", p)
    run("reviews", "rf", build_rf_pipeline(dt.tfidf_vectorizer(), n_estimators=300), Xtr, ytr, Xte, yte, "multi", p)
    np.savez(METRICS / "ch2_pred_reviews_sklearn.npz", y_test=yte, **p)
    log("xong", LOGN)


if __name__ == "__main__":
    main()
