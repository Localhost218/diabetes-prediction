"""
Chương 4: Mạng nơ-ron hồi quy cho chuỗi đường huyết liên tục (CGM).

- Chỉ số CGM theo đồng thuận quốc tế về Time in Range (2019): TIR 70–180 mg/dL > 70%, TBR < 70 mg/dL < 4%,
  TAR > 180 mg/dL < 25%, CV ≤ 36%.
- Dự báo glucose sau 15/30/45/60 phút bằng mô hình hồi quy (LSTM/GRU) huấn luyện trên ShanghaiT1DM/T2DM
  (lab/ch4_rnn.py → data/models/ch4_glucose_rnn.pt). Đầu vào: 12 điểm gần nhất cách nhau 15 phút.
"""

from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.config import BASE_DIR

MODEL_PATH = BASE_DIR / "data" / "models" / "ch4_glucose_rnn.pt"


class SeqModel(nn.Module):
    """Mỗi bước: [glucose chuẩn hóa, sin(giờ), cos(giờ)] → RNN/LSTM/GRU → trạng thái ẩn cuối → Linear(horizon).
    Dự báo phần chênh so với điểm cuối (residual)."""

    def __init__(self, cell: str = "lstm", hidden: int = 64, layers: int = 1, horizon: int = 4):
        super().__init__()
        rnn = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = rnn(3, hidden, num_layers=layers, batch_first=True, dropout=0.2 if layers > 1 else 0.0)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        out, _ = self.rnn(x)
        return x[:, -1:, 0] + self.head(out[:, -1])


class GlucoseForecaster:
    def __init__(self, path=MODEL_PATH):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.cell, self.mu, self.sd = ckpt["cell"], ckpt["mu"], ckpt["sd"]
        self.lookback, self.horizon = ckpt["lookback"], ckpt["horizon"]
        self.model = SeqModel(ckpt["cell"], 64, ckpt["layers"], ckpt["horizon"])
        self.model.load_state_dict(ckpt["state"])
        self.model.eval()

    def forecast(self, series: List[float], last_hour: float) -> List[float]:
        """series: ≥ lookback điểm (mg/dL) cách nhau 15 phút; last_hour: giờ của điểm cuối (0–24)."""
        x = np.asarray(series[-self.lookback:], dtype=np.float32)
        g = (x - self.mu) / self.sd
        # như lúc huấn luyện: giờ của điểm cuối được lặp lại ở mọi bước của cửa sổ
        ang = np.full(self.lookback, 2 * np.pi * (last_hour % 24) / 24)
        feats = np.stack([g, np.sin(ang), np.cos(ang)], -1)[None].astype(np.float32)
        with torch.no_grad():
            out = self.model(torch.from_numpy(feats))[0].numpy() * self.sd + self.mu
        return [round(float(v), 1) for v in out]


@lru_cache(maxsize=1)
def get_forecaster() -> Optional[GlucoseForecaster]:
    return GlucoseForecaster() if MODEL_PATH.exists() else None


def cgm_metrics(series: List[float]) -> Dict[str, Any]:
    v = np.asarray(series, dtype=float)
    mean, cv = float(v.mean()), float(v.std() / max(v.mean(), 1e-6) * 100)
    tir = float(((v >= 70) & (v <= 180)).mean() * 100)
    tar, tbr = float((v > 180).mean() * 100), float((v < 70).mean() * 100)
    if tbr > 4:
        status = "CẢNH BÁO: thời gian dưới 70 mg/dL > 4% — nguy cơ hạ đường huyết"
    elif tar > 25 or mean > 160:
        status = "CẢNH BÁO: thời gian trên 180 mg/dL > 25% — kiểm soát kém"
    elif cv > 36:
        status = "BẤT ỔN ĐỊNH: hệ số biến thiên > 36%"
    else:
        status = "ỔN ĐỊNH"
    return {"n_points": int(len(v)), "mean_glucose_mg_dl": round(mean, 1),
            "estimated_hba1c": round((mean + 46.7) / 28.7, 2),  # công thức eAG (ADAG)
            "tir_percentage": round(tir, 1), "tar_percentage": round(tar, 1), "tbr_percentage": round(tbr, 1),
            "cv_variability_percentage": round(cv, 1), "trajectory_status": status}


class RNNTemporalEngine:
    """Giao diện dùng bởi PredictionAgent: chỉ số CGM (tính thật) + dự báo bằng mô hình hồi quy đã huấn luyện."""

    def analyze_glucose_trajectory(self, glucose_series: List[float],
                                   last_time: Optional[datetime] = None) -> Dict[str, Any]:
        result = cgm_metrics(glucose_series)
        result["glucose_history_sample"] = [round(float(x), 1) for x in glucose_series[-10:]]
        fc = get_forecaster()
        if fc is None or len(glucose_series) < fc.lookback:
            result["forecast"] = None
            result["forecast_note"] = (f"Cần ≥ {fc.lookback if fc else 12} điểm đo cách nhau 15 phút để dự báo"
                                       if fc else "Chưa có mô hình dự báo (chạy lab/ch4_rnn.py)")
            return result
        hour = (last_time or datetime.now()).hour + (last_time or datetime.now()).minute / 60
        pred = fc.forecast(glucose_series, hour)
        result["forecast"] = [{"minutes": 15 * (i + 1), "glucose_mg_dl": p} for i, p in enumerate(pred)]
        result["forecast_next_glucose"] = pred[0]
        result["forecast_model"] = fc.cell.upper()
        result["forecast_note"] = ("Dự báo giả định các lần đo cách nhau 15 phút như dữ liệu huấn luyện "
                                   "(RMSE ≈ 14 mg/dL ở 30 phút trên bệnh nhân chưa gặp)")
        return result
