"""Chương 4 — hàm suy luận cho Web App: dự báo chuỗi thời gian (e-commerce, AAPL, vàng) bằng RNN đã huấn luyện.

predict("ch4_aapl", {"horizon": 10}, "torch_lstm") → giá trị bước kế tiếp + dự báo đệ quy nhiều bước + lịch sử
để vẽ biểu đồ. Không phụ thuộc Flask; mô hình nạp lười từ models/ch4/.
"""
import json
import time

import numpy as np
import pandas as pd

from .common import METRICS, model_dir

ARCH_LABEL = {"rnn": "SimpleRNN", "lstm": "LSTM", "gru": "GRU", "bilstm": "BiLSTM"}
MODELS = [f"{fw}_{a}" for fw in ("torch", "keras") for a in ("rnn", "lstm", "gru", "bilstm")]
_EXP = {"ch4_ecommerce": ("ecommerce", "Doanh thu ngày kế tiếp (E-commerce)", "TRY", "D"),
        "ch4_aapl": ("aapl", "Giá đóng cửa AAPL phiên kế tiếp", "USD", "B"),
        "ch4_gold": ("gold", "Giá đóng cửa quỹ vàng GLD phiên kế tiếp", "USD", "B")}


def _inputs(exp_id):
    return [{"name": "horizon", "label": "Số bước dự báo (đệ quy)", "type": "number", "min": 1, "max": 30,
             "default": 10},
            {"name": "history", "label": "Chuỗi giá trị gần nhất (≥ 30, phẩy hoặc xuống dòng; để trống = dữ liệu cuối)",
             "type": "text", "default": ""}]


SPECS = {eid: {"title": title, "kind": "series", "inputs": _inputs(eid), "models": MODELS, "unit": unit,
               "default_model": "torch_lstm", "example": {"horizon": 10}}
         for eid, (_, title, unit, _) in _EXP.items()}

_cache = {}


def _prepared(name):
    if ("prep", name) not in _cache:
        from .ch4_data import Prepared, experiment
        _cache[("prep", name)] = Prepared(experiment(name))
    return _cache[("prep", name)]


def _model(name, model_id, n_features):
    """Trả hàm f(X) -> đầu ra chuẩn hoá cho mô hình <fw>_<arch> của thực nghiệm name."""
    key = ("model", name, model_id)
    if key not in _cache:
        fw, arch = model_id.split("_", 1)
        path = model_dir("ch4") / f"{name}_{fw}_{arch}.{'pt' if fw == 'torch' else 'keras'}"
        if fw == "torch":
            import torch
            from .ch4_torch import SeqRegressor
            m = SeqRegressor(n_features, arch)
            m.load_state_dict(torch.load(path, map_location="cpu"))
            m.eval()

            def fn(x, m=m):
                with torch.no_grad():
                    return m(torch.as_tensor(x)).numpy()
        else:
            import keras
            m = keras.models.load_model(path)

            def fn(x, m=m):
                return np.asarray(m(x, training=False)).reshape(-1)
        _cache[key] = fn
    return _cache[key]


def _parse_history(h):
    if h is None or h == "":
        return None
    if isinstance(h, str):
        h = [v for v in h.replace(";", ",").replace("\n", ",").split(",") if v.strip()]
    return np.array([float(str(v).strip()) for v in h], dtype=np.float64)


def _test_metrics(name, model_id):
    fw, arch = model_id.split("_", 1)
    try:
        m = json.loads((METRICS / f"ch4_{name}_{fw}.json").read_text(encoding="utf-8"))
        return {"test": m[arch]["test"], "naive_test": m["naive"]["test"]}
    except (OSError, KeyError):
        return {}


def predict(exp_id, payload, model_id=None):
    """Dự báo bước kế tiếp và `horizon` bước đệ quy; payload: {"horizon": int, "history": list|str (tuỳ chọn)}."""
    from .ch4_forecast import recursive_forecast
    t0 = time.perf_counter()
    name, title, unit, freq = _EXP[exp_id]
    model_id = model_id or "torch_lstm"
    prep = _prepared(name)
    frame = prep.exp.frame
    horizon = int(max(1, min(30, int(payload.get("horizon", 10) or 10))))
    rows = frame.values[-prep.window:].astype(np.float64).copy()
    hist_vals = frame[prep.exp.target].values[-90:].astype(float)
    hist_dates = [d.date().isoformat() for d in frame.index[-90:]]
    user = _parse_history(payload.get("history"))
    if user is not None:
        if len(user) < prep.window:
            raise ValueError(f"Cần ít nhất {prep.window} giá trị lịch sử, nhận {len(user)}")
        user = user[-prep.window:]
        t = prep.t_idx
        if prep.mode == "relative":                 # các cột giá cùng tài sản đi theo tỉ lệ của chuỗi nhập
            ratio = user / rows[:, t]
            target = prep.exp.target
            asset = target.split("_")[0] + "_"
            for j, c in enumerate(prep.cols):
                same_asset = c == target or "_" not in target or c.startswith(asset)
                if c not in prep.exp.level_cols and same_asset:
                    rows[:, j] = rows[:, j] * ratio
        rows[:, t] = user
        hist_vals, hist_dates = user.tolist(), None
    fn = _model(name, model_id, prep.X.shape[-1])
    forecast = recursive_forecast(prep, fn, rows, horizon)
    last_date = frame.index[-1]
    if hist_dates is None:
        fdates = None
    else:
        fdates = [d.date().isoformat() for d in pd.date_range(last_date, periods=horizon + 1, freq=freq)[1:]]
    return {"value": float(forecast[0]), "label": f"{forecast[0]:,.2f} {unit}",
            "details": {"history": [float(v) for v in hist_vals], "dates": hist_dates,
                        "forecast": [float(v) for v in forecast], "forecast_dates": fdates,
                        "last_value": float(rows[-1, prep.t_idx]), "naive": float(rows[-1, prep.t_idx]),
                        # so với trung bình 7 phiên cuối: chuỗi doanh thu ngày dao động mạnh nên so với một ngày lẻ vô nghĩa
                        "mean_7": float(rows[-7:, prep.t_idx].mean()),
                        "change_pct": float(100 * (forecast[0] / rows[-7:, prep.t_idx].mean() - 1)),
                        "target": prep.exp.target, "unit": unit, "window": prep.window,
                        "architecture": ARCH_LABEL[model_id.split("_", 1)[1]],
                        "framework": "PyTorch" if model_id.startswith("torch") else "Keras",
                        **_test_metrics(name, model_id)},
            "model": model_id, "latency_ms": (time.perf_counter() - t0) * 1000}
