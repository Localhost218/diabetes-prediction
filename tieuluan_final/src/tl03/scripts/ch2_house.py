"""TN2 — Dự đoán giá nhà: khảo sát, biểu diễn, 10 mô hình hồi quy + MLP, kiểm định chéo, phân tích phần dư.
Ghi outputs/metrics/ch2_house.json, outputs/ch2_house_preds.npz, models/ch2/house_*.joblib|pt."""
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
from sklearn.linear_model import Lasso, Ridge  # noqa: E402
from sklearn.model_selection import KFold, cross_validate, learning_curve  # noqa: E402

from tl3 import ch2_data as cd  # noqa: E402
from tl3.ch2_metrics import reg_metrics  # noqa: E402
from tl3.ch2_mlp import fit_mlp, predict_raw  # noqa: E402
from tl3.ch2_models import REG_NAMES, reg_pipeline, regressors  # noqa: E402
from tl3.common import OUT, device, model_dir, save_json  # noqa: E402
from tl3.gpu import cap  # noqa: E402

M = model_dir("ch2")


def regularization_path(Z, y, feats):
    """Hệ số chuẩn hóa của Ridge và Lasso theo alpha (đường chính quy hóa, minh hoạ L2 co đều / L1 về 0)."""
    alphas = np.logspace(-1, 5, 30)
    ridge = [Ridge(alpha=a).fit(Z, y).coef_.tolist() for a in alphas]
    lasso_alphas = np.logspace(1, 5.3, 30)
    lasso = [Lasso(alpha=a, max_iter=50000).fit(Z, y).coef_.tolist() for a in lasso_alphas]
    return {"features": feats, "ridge_alpha": alphas.tolist(), "ridge": ridge,
            "lasso_alpha": lasso_alphas.tolist(), "lasso": lasso}


def main():
    raw = cd.load_house_raw()
    res = {"survey": {"shape": list(raw.shape), "describe": raw.describe().T.round(2).to_dict(orient="index"),
                      "cats": {c: raw[c].value_counts().to_dict() for c in cd.HOUSE_CAT},
                      "corr_target": raw.corr(numeric_only=True)["Price"].drop("Price").round(4).to_dict(),
                      "price_by_city": raw.groupby("City").Price.median().round(0).to_dict(),
                      "price_by_furnishing": raw.groupby("Furnishing").Price.median().round(0).to_dict(),
                      "price_by_bedrooms": raw.groupby("Bedrooms").Price.median().round(0).to_dict(),
                      "skew_price": float(raw.Price.skew())},
           "audit": cd.audit(raw, cd.HOUSE_NUM + [cd.HOUSE_TARGET])}
    df, (X_tr, X_va, X_te, y_tr, y_va, y_te) = cd.house_splits()
    pre = cd.house_preprocessor().fit(X_tr)
    feats = cd.feature_names(pre)
    res["split"] = {"train": len(y_tr), "val": len(y_va), "test": len(y_te), "n_features": len(feats),
                    "features": feats}

    models, preds, res["models"] = {}, {}, {}
    for name, model in regressors().items():
        pipe = reg_pipeline(cd.house_preprocessor(), model)
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        models[name] = pipe
        preds[name] = (pipe.predict(X_va), pipe.predict(X_te))
        res["models"][name] = {"train_s": time.perf_counter() - t0}
        print(f"{name:8s} R2 val={reg_metrics(y_va, preds[name][0])['r2']:.4f}", flush=True)

    # MLP PyTorch trên giá chuẩn hóa (z-score), đổi ngược khi dự đoán
    cap(0.15)
    mu, sd = y_tr.mean(), y_tr.std()
    Ztr, Zva, Zte = (pre.transform(x).astype(np.float32) for x in (X_tr, X_va, X_te))
    mlp, info = fit_mlp(Ztr, ((y_tr - mu) / sd).astype(np.float32), Zva, ((y_va - mu) / sd).astype(np.float32),
                        task="reg", hidden=(64, 32), dropout=0.1, lr=2e-3, epochs=300, bs=64, patience=25)
    preds["mlp"] = tuple(predict_raw(mlp, z, device()) * sd + mu for z in (Zva, Zte))
    res["models"]["mlp"] = {"train_s": info["train_s"], "epochs": info["epochs_run"], "n_params": info["n_params"],
                            "history": info["history"]}
    torch.save({"state": mlp.state_dict(), "mu": float(mu), "sd": float(sd), "n_in": Ztr.shape[1]},
               M / "house_mlp.pt")
    joblib.dump(pre, M / "house_pre.joblib")

    res["val"] = {n: reg_metrics(y_va, p[0]) for n, p in preds.items()}
    res["test"] = {n: reg_metrics(y_te, p[1]) for n, p in preds.items()}
    ranked = sorted((n for n in preds if n != "baseline"), key=lambda n: res["val"][n]["rmse"])
    best = ranked[0]
    res["ranking"] = ranked
    res["best"] = {"name": best, "label": REG_NAMES[best], **res["test"][best]}

    cv = KFold(5, shuffle=True, random_state=42)
    res["cv"] = {}
    for name in ("linear", "ridge", "lasso", "poly", "knn", "tree", "rf", "xgb", "svr"):
        s = cross_validate(reg_pipeline(cd.house_preprocessor(), regressors()[name]), X_tr, y_tr, cv=cv,
                           scoring=["r2", "neg_root_mean_squared_error"], n_jobs=1)
        res["cv"][name] = {"r2_mean": float(s["test_r2"].mean()), "r2_std": float(s["test_r2"].std()),
                           "rmse_mean": float(-s["test_neg_root_mean_squared_error"].mean())}

    lin = models["linear"]["model"]
    res["linear_coef"] = dict(zip(feats, lin.coef_.round(1).tolist()))
    res["linear_intercept"] = float(lin.intercept_)
    res["rf_importance"] = dict(zip(feats, models["rf"]["model"].feature_importances_.round(4).tolist()))
    Z = pre.transform(X_tr)
    res["reg_path"] = regularization_path(Z, y_tr, feats)
    sizes, tr_s, va_s = learning_curve(reg_pipeline(cd.house_preprocessor(), regressors()[best if best != "mlp" else "ridge"]),
                                       X_tr, y_tr, train_sizes=np.linspace(0.1, 1.0, 8), cv=cv, scoring="r2")
    res["learning_curve"] = {"model": best if best != "mlp" else "ridge", "sizes": sizes.tolist(),
                             "train": tr_s.mean(1).tolist(), "val": va_s.mean(1).tolist()}
    # sai số theo phân vị giá của mô hình tốt nhất trên Test
    p = preds[best][1]
    q = np.quantile(y_te, [0, 0.25, 0.5, 0.75, 1.0])
    res["err_by_quartile"] = [{"lo": float(q[i]), "hi": float(q[i + 1]),
                               "mae": float(np.mean(np.abs(p - y_te)[(y_te >= q[i]) & (y_te <= q[i + 1])]))}
                              for i in range(4)]
    for name in ("linear", "ridge", "lasso", "poly", "rf", "xgb", "knn", "tree", "svr"):
        joblib.dump(models[name], M / f"house_{name}.joblib", compress=3)
    np.savez_compressed(OUT / "ch2_house_preds.npz", y_va=y_va, y_te=y_te,
                        **{f"{n}_va": p[0] for n, p in preds.items()}, **{f"{n}_te": p[1] for n, p in preds.items()})
    save_json("ch2_house", res)
    print("best", best, res["best"])


if __name__ == "__main__":
    main()
