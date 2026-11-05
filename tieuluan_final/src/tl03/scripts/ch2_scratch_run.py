"""Đối chiếu cài đặt NumPy từ đầu với scikit-learn: hồi quy tuyến tính (House), hồi quy logistic (Diabetes),
PLA trên dữ liệu tách được tuyến tính. Ghi outputs/metrics/ch2_scratch.json."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import limit_threads, seed_all  # noqa: E402

limit_threads(4)
seed_all()

import numpy as np  # noqa: E402
from sklearn.datasets import make_blobs  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, r2_score, roc_auc_score  # noqa: E402

from tl3 import ch2_data as cd  # noqa: E402
from tl3 import ch2_scratch as sc  # noqa: E402
from tl3.common import save_json  # noqa: E402


def main():
    res = {}
    # hồi quy tuyến tính: phương trình chuẩn và GD so với sklearn trên House Price (đặc trưng đã chuẩn hóa)
    _, (X_tr, X_va, X_te, y_tr, y_va, y_te) = cd.house_splits()
    pre = cd.house_preprocessor().fit(X_tr)
    Ztr, Zte = pre.transform(X_tr), pre.transform(X_te)
    ym, ys = y_tr.mean(), y_tr.std()
    t0 = time.perf_counter()
    w_ne = sc.linreg_normal_equation(Ztr, y_tr)
    t_ne = time.perf_counter() - t0
    w_gd, hist = sc.linreg_gd(Ztr, (y_tr - ym) / ys, lr=0.05, epochs=600)
    pred_ne = np.hstack([np.ones((len(Zte), 1)), Zte]) @ w_ne
    pred_gd = (np.hstack([np.ones((len(Zte), 1)), Zte]) @ w_gd) * ys + ym
    sk = LinearRegression().fit(Ztr, y_tr)
    res["linreg"] = {"r2_normal_eq": r2_score(y_te, pred_ne), "r2_gd": r2_score(y_te, pred_gd),
                     "r2_sklearn": r2_score(y_te, sk.predict(Zte)), "time_ne_ms": 1000 * t_ne,
                     "max_coef_diff": float(np.max(np.abs(w_ne[1:] - sk.coef_))),
                     "gd_mse_history": hist[::10], "gd_final_mse": hist[-1]}

    # hồi quy logistic: mini-batch GD tự viết so với sklearn trên Diabetes
    _, (X_tr, X_va, X_te, y_tr, y_va, y_te) = cd.diabetes_splits()
    pre = cd.diabetes_preprocessor().fit(X_tr)
    Ztr, Zte = pre.transform(X_tr), pre.transform(X_te)
    t0 = time.perf_counter()
    w, lhist = sc.logreg_gd(Ztr, y_tr.astype(float), lr=0.5, epochs=150)
    t_np = time.perf_counter() - t0
    p_np = sc.logreg_predict(w, Zte)
    t0 = time.perf_counter()
    skl = LogisticRegression(max_iter=2000).fit(Ztr, y_tr)
    t_sk = time.perf_counter() - t0
    p_sk = skl.predict_proba(Zte)[:, 1]
    res["logreg"] = {"auc_numpy": roc_auc_score(y_te, p_np), "auc_sklearn": roc_auc_score(y_te, p_sk),
                     "f1_numpy": f1_score(y_te, p_np >= 0.5), "f1_sklearn": f1_score(y_te, p_sk >= 0.5),
                     "time_numpy_s": t_np, "time_sklearn_s": t_sk, "loss_history": lhist,
                     "corr_coef": float(np.corrcoef(w[1:], skl.coef_[0])[0, 1])}

    # PLA trên hai cụm tách được tuyến tính
    X, y = make_blobs(200, centers=[(-2, -1.5), (2, 1.5)], cluster_std=1.0, random_state=3)
    yy = np.where(y == 1, 1, -1)
    wp, errs = sc.pla(X, yy)
    res["pla"] = {"epochs": len(errs), "errors": errs, "w": wp.tolist(), "X": X.tolist(), "y": yy.tolist()}

    # Softmax 3 lớp trên dữ liệu 2D
    X3, y3 = make_blobs(450, centers=[(-2.5, 0), (2.5, 0.5), (0, 3)], cluster_std=1.1, random_state=7)
    W = sc.softmax_gd(X3, y3, 3, lr=0.3, epochs=500)
    res["softmax"] = {"acc": float((sc.softmax_predict(W, X3) == y3).mean()), "W": W.tolist(),
                      "X": X3.tolist(), "y": y3.tolist()}
    save_json("ch2_scratch", res)
    print({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in res.items()})


if __name__ == "__main__":
    main()
