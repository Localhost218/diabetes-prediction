"""Chỉ số đánh giá dùng thống nhất cho mọi mô hình ở cả ba chương."""
import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)


def classification(y_true, y_pred, y_score=None, binary=True):
    """Nhị phân: precision/recall/F1 của lớp dương (lớp hiếm cần phát hiện). Đa lớp: trung bình macro."""
    avg = "binary" if binary else "macro"
    out = {"acc": accuracy_score(y_true, y_pred),
           "precision": precision_score(y_true, y_pred, average=avg, zero_division=0),
           "recall": recall_score(y_true, y_pred, average=avg, zero_division=0),
           "f1": f1_score(y_true, y_pred, average=avg, zero_division=0),
           "confusion": confusion_matrix(y_true, y_pred).tolist()}
    if binary and y_score is not None:
        out["auc"] = roc_auc_score(y_true, y_score)
    return out


def regression(y_true, y_pred):
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": r2_score(y_true, y_pred)}


def best_f1_threshold(y_true, score):
    """Ngưỡng cho F1 lớp dương cao nhất, chọn trên tập kiểm định (không bao giờ trên tập kiểm thử)."""
    grid = np.linspace(0.05, 0.95, 91)
    f1s = [f1_score(y_true, score >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1s))])
