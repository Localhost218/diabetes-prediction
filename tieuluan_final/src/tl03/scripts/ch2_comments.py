"""TN3 — Phân loại nhận xét khách hàng (Recommended IND) bằng TF-IDF + mô hình tuyến tính/xác suất, và Softmax
Regression 5 lớp cho Rating. Ghi outputs/metrics/ch2_comments.json, models/ch2/comments_*.joblib."""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import limit_threads, seed_all  # noqa: E402

limit_threads(6)
seed_all()

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from tl3 import ch2_data as cd  # noqa: E402
from tl3.ch2_metrics import macro_metrics  # noqa: E402
from tl3.ch2_text import TEXT_NAMES, text_models, top_terms  # noqa: E402
from tl3.common import OUT, model_dir, save_json  # noqa: E402

M = model_dir("ch2")


def main():
    df, (X_tr, X_va, X_te, y_tr, y_va, y_te) = cd.reviews_splits()
    raw = cd.pd.read_csv(cd.D / "womens_reviews.csv", index_col=0)
    res = {"survey": {"raw_rows": len(raw), "missing_text": int(raw["Review Text"].isna().sum()),
                      "missing_title": int(raw["Title"].isna().sum()), "rows": len(df), "n_rows": len(df),
                      "pos_rate": float(df["Recommended IND"].mean()),
                      "rating_counts": df.Rating.value_counts().sort_index().to_dict(),
                      "rec_by_rating": df.groupby("Rating")["Recommended IND"].mean().round(4).to_dict(),
                      "words_median": float(df.n_words.median()), "words_mean": float(df.n_words.mean()),
                      "words_p95": float(df.n_words.quantile(0.95)),
                      "words_by_class": df.groupby("Recommended IND").n_words.median().to_dict(),
                      "dept_counts": df["Department Name"].value_counts().to_dict()},
           "split": {"train": len(y_tr), "val": len(y_va), "test": len(y_te)}}
    models, res["val"], res["test"], res["models"], preds = {}, {}, {}, {}, {}
    for name, pipe in text_models().items():
        t0 = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        secs = time.perf_counter() - t0
        sc = (lambda X: pipe.predict_proba(X)[:, 1]) if hasattr(pipe, "predict_proba") else \
             (lambda X: pipe.decision_function(X))
        res["val"][name] = macro_metrics(y_va, pipe.predict(X_va), sc(X_va))
        res["test"][name] = macro_metrics(y_te, pipe.predict(X_te), sc(X_te))
        res["models"][name] = {"train_s": secs, "vocab": len(pipe["tfidf"].vocabulary_)}
        models[name] = pipe
        preds[name] = pipe.predict(X_te)
        print(f"{name:8s} macroF1 val={res['val'][name]['macro_f1']:.4f}", flush=True)
    ranked = sorted((n for n in models if n != "baseline"), key=lambda n: -res["val"][n]["macro_f1"])
    best = ranked[0]
    res["ranking"] = ranked
    res["best"] = {"name": best, "label": TEXT_NAMES[best], **res["test"][best]}
    res["confusion"] = {n: confusion_matrix(y_te, p).tolist() for n, p in preds.items()}
    res["top_terms"] = top_terms(models["logreg"], 15)

    # Softmax Regression nhiều lớp: dự đoán Rating 1..5 từ cùng văn bản
    Xa, Xb, ra, rb = train_test_split(df.text.to_numpy(), df.Rating.to_numpy(), test_size=0.2,
                                      stratify=df.Rating, random_state=42)
    soft = text_models()["softmax"]
    soft.fit(Xa, ra)
    pr = soft.predict(Xb)
    res["rating"] = {"acc": float((pr == rb).mean()), "macro_f1": float(f1_score(rb, pr, average="macro")),
                     "within1": float((np.abs(pr - rb) <= 1).mean()),
                     "confusion": confusion_matrix(rb, pr, labels=[1, 2, 3, 4, 5]).tolist(),
                     "f1_per_class": f1_score(rb, pr, average=None, labels=[1, 2, 3, 4, 5]).round(4).tolist()}
    for name in ("logreg", "nb", "svm", "cnb"):
        joblib.dump(models[name], M / f"comments_{name}.joblib", compress=3)
    np.savez_compressed(OUT / "ch2_comments_preds.npz", y_te=y_te,
                        **{f"{n}_te": p for n, p in preds.items()})
    save_json("ch2_comments", res)
    print("best", best, res["best"])


if __name__ == "__main__":
    main()
