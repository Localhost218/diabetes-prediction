"""Chương 4 — biến một cửa sổ dữ liệu thô thành đầu vào mô hình và dự báo nhiều bước kiểu đệ quy (recursive).

Dùng chung cho đánh giá theo tầm dự báo (horizon) và cho Web App (tl3/serve_ch4.py).
"""
import numpy as np


def window_to_input(prep, rows):
    """rows: mảng thô (window × F) theo thứ tự cột của prep.cols → tensor (1, window, F) đúng phép chuẩn hoá lúc huấn luyện."""
    rows = np.asarray(rows, dtype=np.float64)
    if prep.mode == "level":
        z = (rows - prep.mu) / prep.sd
    elif prep.mode == "minmax":
        z = (rows - prep.lo) / (prep.hi - prep.lo + 1e-9)
    else:
        level = [prep.cols.index(c) for c in prep.exp.level_cols]
        price = [i for i in range(len(prep.cols)) if i not in level]
        logp = np.log(rows[:, price]) - np.log(rows[-1, price])
        lv = (np.log1p(rows[:, level]) - prep.lv_mu) / prep.lv_sd
        z = np.concatenate([logp, lv], axis=1) * 10.0
    return z[None].astype(np.float32)


def output_to_value(prep, out, rows):
    """Đầu ra chuẩn hoá của mô hình → giá trị mục tiêu bước kế tiếp (đơn vị gốc)."""
    out = float(np.asarray(out).reshape(-1)[0])
    t = prep.t_idx
    if prep.mode == "level":
        return out * prep.sd[t] + prep.mu[t]
    if prep.mode == "minmax":
        return out * (prep.hi[t] - prep.lo[t]) + prep.lo[t]
    return float(rows[-1, t] * np.exp(out * prep.ret_sd))


def next_row(prep, rows, value):
    """Dựng hàng dữ liệu của ngày kế tiếp khi chỉ biết dự báo của biến mục tiêu.

    Giá (relative): các cột giá cùng tài sản (open/high/low/close…) nhân cùng tỉ lệ thay đổi của biến mục tiêu,
    chỉ báo ngoại sinh và khối lượng giữ như bước cuối. E-commerce (level): các đặc trưng khác lấy trung bình 7 ngày
    cuối. Đây là giả định đơn giản hoá của dự báo đệ quy và là nguồn sai số tích luỹ theo tầm dự báo.
    """
    t = prep.t_idx
    if prep.mode == "level":
        new = rows[-7:].mean(0).copy()
    else:
        new = rows[-1].copy()
        ratio = value / rows[-1, t]
        stem = prep.cols[t].split("_")[0] if "_" in prep.cols[t] else None
        for j, c in enumerate(prep.cols):
            same_asset = (stem is not None and c.startswith(stem + "_") and c not in prep.exp.level_cols) or \
                         (stem is None and c not in prep.exp.level_cols)
            if same_asset:
                new[j] = rows[-1, j] * ratio
    new[t] = value
    return new


def recursive_forecast(prep, predict_fn, rows, steps=10):
    """Dự báo `steps` bước: mỗi bước dự báo một giá trị, nối hàng mới vào cuối cửa sổ, bỏ hàng đầu, lặp lại."""
    rows = np.asarray(rows, dtype=np.float64).copy()
    out = []
    for _ in range(steps):
        v = output_to_value(prep, predict_fn(window_to_input(prep, rows)), rows)
        out.append(v)
        rows = np.vstack([rows[1:], next_row(prep, rows, v)])
    return np.array(out)
