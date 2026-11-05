"""Web App — Chương 4 (mạch tiểu đường): dự báo glucose 30 phút tới từ 3 giờ đo CGM gần nhất.

Đầu vào: chọn một đoạn thật của bệnh nhân thuộc tập KIỂM TRA (chưa từng dùng để huấn luyện), hoặc tự nhập 12 giá trị
mg/dL (mỗi 15 phút) và giờ của điểm cuối. Mô hình: các bản PyTorch và Keras đã huấn luyện bởi scripts/run_ch4_cgm.py.
"""
import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import ch4_cgm as C

TLF = Path(__file__).resolve().parents[2]
MODELS_DIR = TLF / "models" / "ch4"
METRICS = TLF / "outputs" / "metrics"
MODEL_IDS = ["torch_lstm1", "torch_lstm5", "torch_gru1", "torch_rnn1", "keras_lstm1", "keras_lstm5", "keras_gru1", "keras_rnn1"]
MODEL_NAMES = {"torch_lstm1": "LSTM – PyTorch", "torch_lstm5": "LSTM 5 lớp – PyTorch", "torch_gru1": "GRU – PyTorch",
               "torch_rnn1": "SimpleRNN – PyTorch", "keras_lstm1": "LSTM – Keras", "keras_lstm5": "LSTM 5 lớp – Keras",
               "keras_gru1": "GRU – Keras", "keras_rnn1": "SimpleRNN – Keras"}
N_SAMPLES = 400

SPECS = {"ch4_dm_cgm": {
    "title": "Dự báo đường huyết CGM 30 phút tới",
    "kind": "series",
    "description": "Từ 12 lần đo gần nhất (3 giờ, cảm biến 15 phút/lần) dự báo glucose sau 30 phút. Đoạn mẫu lấy từ "
                   "17 bệnh nhân thuộc tập kiểm tra của ShanghaiT1DM/T2DM.",
    "inputs": [{"name": "sample", "label": f"Đoạn mẫu của tập kiểm tra (0–{N_SAMPLES - 1})", "type": "number",
                "min": 0, "max": N_SAMPLES - 1, "default": 7},
               {"name": "history", "label": "Hoặc tự nhập 12 giá trị mg/dL (cách nhau bằng dấu phẩy)", "type": "text",
                "default": ""},
               {"name": "hour", "label": "Giờ của lần đo cuối (khi tự nhập)", "type": "number", "min": 0, "max": 23.75,
                "step": 0.25, "default": 8}],
    "models": MODEL_IDS, "model_names": MODEL_NAMES, "unit": "mg/dL",
    "example": {"sample": 7, "history": "", "hour": 8},
}}


@lru_cache(maxsize=1)
def _data():
    d = C.CGMData()
    X, H, Y = d.raw["test"]
    idx = np.random.default_rng(1).permutation(len(Y))[:N_SAMPLES]
    return d, X[idx], H[idx], Y[idx]


@lru_cache(maxsize=None)
def _model(model_id):
    fw, key = model_id.split("_", 1)
    arch, n = key[:-1], int(key[-1])
    if fw == "torch":
        import torch
        m = C.torch_model(arch, layers_n=n)
        m.load_state_dict(torch.load(MODELS_DIR / f"cgm_torch_{key}.pt", map_location="cpu"))
        m.eval()
        return lambda x: C.predict_torch(m, x)
    import keras
    m = keras.models.load_model(MODELS_DIR / f"cgm_keras_{key}.keras")
    return lambda x: np.asarray(m(x, training=False)).reshape(-1)


def _test_metrics(model_id):
    fw, key = model_id.split("_", 1)
    try:
        m = json.loads((METRICS / f"ch4_cgm_{fw}.json").read_text(encoding="utf-8"))["models"][key]
        return {"test_rmse": round(m["rmse"], 2), "test_clarke_a": round(m["clarke_a"], 3)}
    except (OSError, KeyError):
        return {}


def _zone(v):
    return "hạ đường huyết (< 70)" if v < 70 else "tăng đường huyết (> 180)" if v > 180 else "trong ngưỡng 70–180"


def predict(exp_id, payload, model_id=None):
    t0 = time.perf_counter()
    model_id = model_id or MODEL_IDS[0]
    d, Xs, Hs, Ys = _data()
    raw = str(payload.get("history") or "").replace(";", ",").replace("\n", ",")
    vals = [float(v) for v in raw.split(",") if v.strip()]
    actual = None
    if vals:
        if len(vals) < C.LOOKBACK:
            raise ValueError(f"Cần {C.LOOKBACK} giá trị, nhận {len(vals)}")
        x = np.array(vals[-C.LOOKBACK:], np.float32)
        hour = float(payload.get("hour", 8) or 8)
    else:
        i = int(payload.get("sample", 0) or 0) % len(Ys)
        x, hour, actual = Xs[i], float(Hs[i]), float(Ys[i])
    feats = d.features(x[None], np.array([hour], np.float32))
    pred = float(x[-1] + _model(model_id)(feats)[0] * d.sd)
    end_min = int(round(hour * 60))
    times = [f"{((end_min - 15 * (C.LOOKBACK - 1 - k)) % 1440) // 60:02d}:{((end_min - 15 * (C.LOOKBACK - 1 - k)) % 1440) % 60:02d}"
             for k in range(C.LOOKBACK)]
    details = {"history": [float(v) for v in x], "dates": times, "forecast": [pred], "forecast_dates": ["+30 phút"],
               "last_value": float(x[-1]), "naive": float(x[-1]), "zone": _zone(pred), "unit": "mg/dL",
               "framework": "PyTorch" if model_id.startswith("torch") else "Keras", **_test_metrics(model_id)}
    if actual is not None:
        details["actual_30min"] = actual
        details["abs_error"] = round(abs(pred - actual), 2)
    return {"value": pred, "label": f"{pred:.0f} mg/dL", "details": details, "model": model_id,
            "latency_ms": (time.perf_counter() - t0) * 1000}
