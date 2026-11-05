"""Chương 2 — chỉ số đánh giá phân loại/hồi quy và chọn ngưỡng quyết định theo F1 trên tập Validation."""
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix, f1_score,
                             mean_absolute_error, mean_absolute_percentage_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)


def clf_metrics(y, proba, thr=0.5):
    """Accuracy, Precision, Recall, F1 của lớp dương ở ngưỡng thr; ROC-AUC và PR-AUC không phụ thuộc ngưỡng."""
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"acc": accuracy_score(y, pred), "prec": precision_score(y, pred, zero_division=0),
            "rec": recall_score(y, pred, zero_division=0), "f1": f1_score(y, pred, zero_division=0),
            "auc": roc_auc_score(y, proba) if len(set(proba)) > 1 else 0.5,
            "ap": average_precision_score(y, proba), "tn": int(tn), "fp": int(fp), "fn": int(fn),
            "tp": int(tp), "thr": float(thr)}


def best_threshold(y, proba, grid=None):
    """Quét ngưỡng 0,05–0,95 trên Validation, trả ngưỡng cho F1 lớn nhất (không bao giờ chọn trên Test)."""
    grid = np.round(np.arange(0.05, 0.951, 0.01), 2) if grid is None else grid
    f1s = [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]
    i = int(np.argmax(f1s))
    return float(grid[i]), {"grid": grid.tolist(), "f1": [float(v) for v in f1s]}


def reg_metrics(y, pred):
    """MAE, RMSE, R² và MAPE cho bài toán hồi quy."""
    return {"mae": mean_absolute_error(y, pred), "rmse": float(np.sqrt(mean_squared_error(y, pred))),
            "r2": r2_score(y, pred), "mape": mean_absolute_percentage_error(y, pred)}


def macro_metrics(y, pred, proba=None):
    """Chỉ số cho phân loại văn bản mất cân bằng: Accuracy, macro-F1, F1 từng lớp, ROC-AUC."""
    out = {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
           "f1_neg": f1_score(y, pred, pos_label=0), "f1_pos": f1_score(y, pred, pos_label=1),
           "prec_neg": precision_score(y, pred, pos_label=0, zero_division=0),
           "rec_neg": recall_score(y, pred, pos_label=0, zero_division=0)}
    if proba is not None:
        out["auc"] = roc_auc_score(y, proba)
    return out
