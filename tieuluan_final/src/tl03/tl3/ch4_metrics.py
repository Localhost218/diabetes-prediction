"""Chương 4 — chỉ số đánh giá dự báo chuỗi thời gian (tính trên đơn vị gốc của biến mục tiêu)."""
import numpy as np


def forecast_metrics(y, yhat, last=None):
    """MAE, RMSE, MAPE (%), R² và độ chính xác hướng (dấu của thay đổi so với bước trước có khớp không)."""
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    err = y - yhat
    out = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mape": float(100 * np.mean(np.abs(err) / np.maximum(np.abs(y), 1e-9))),
        "r2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    }
    if last is not None:
        last = np.asarray(last, float)
        moved = np.sign(y - last) != 0
        out["dir_acc"] = float(np.mean(np.sign(yhat - last)[moved] == np.sign(y - last)[moved]))
    return out


def skill(model_rmse, naive_rmse):
    """Theil-U kiểu tỉ số: RMSE mô hình / RMSE dự báo ngây thơ; < 1 nghĩa là thắng naive."""
    return float(model_rmse / naive_rmse)
