"""Hai mô hình học sâu làm công cụ cho tác tử: MLP chẩn đoán nguy cơ đái tháo đường và BiLSTM dự báo chuỗi giá."""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn

from .common import MODELS
from .data import diabetes_features, forecast_features, gold_frame, market_frame

class DiabetesMLP(nn.Module):
    """13 -> 128 -> 64 -> 1, BatchNorm + ReLU + Dropout; đầu ra là logit của P(đái tháo đường)."""

    def __init__(self, d_in, hidden=(128, 64), dropout=0.2):
        super().__init__()
        layers, d = [], d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


class DiabetesRiskModel:
    """Bọc mô hình đã huấn luyện thành công cụ: nhận hồ sơ (dict) trả xác suất và mức nguy cơ."""

    def __init__(self, dev="cpu"):
        meta = json.loads((MODELS / "diabetes_mlp.json").read_text(encoding="utf-8"))
        self.mean, self.std = np.array(meta["mean"]), np.array(meta["std"])
        self.cols, self.threshold = meta["columns"], meta["threshold"]
        self.model = DiabetesMLP(len(self.cols), tuple(meta["hidden"]), meta["dropout"])
        self.model.load_state_dict(torch.load(MODELS / "diabetes_mlp.pt", map_location=dev))
        self.model.to(dev).eval()
        self.dev = dev

    def predict(self, profile):
        p = {"gender": "Male", "age": 50, "hypertension": 0, "heart_disease": 0, "smoking_history": "never",
             "bmi": 23.0, "HbA1c_level": 5.5, "blood_glucose_level": 100}
        p.update({k: v for k, v in profile.items() if v is not None})
        X = diabetes_features(pd.DataFrame([p]))[self.cols].to_numpy()
        x = torch.tensor((X - self.mean) / self.std, dtype=torch.float32, device=self.dev)
        with torch.no_grad():
            prob = float(torch.sigmoid(self.model(x))[0])
        # huấn luyện với pos_weight đẩy xác suất lên cao, nên mốc "cao" là ngưỡng tối ưu F1 chứ không phải 0,5
        level = "cao" if prob >= self.threshold else "trung bình" if prob >= 0.5 else "thấp"
        return {"model": "DiabetesMLP (PyTorch)", "probability": prob, "threshold": self.threshold,
                "positive": prob >= self.threshold, "risk_level": level, "input": p}


class SeqForecaster(nn.Module):
    """LSTM một chiều hoặc hai chiều; đầu ra là lợi suất log phiên t+1."""

    def __init__(self, d_in, hidden=64, layers=2, dropout=0.2, bidirectional=True):
        super().__init__()
        self.rnn = nn.LSTM(d_in, hidden, num_layers=layers, batch_first=True, dropout=dropout,
                           bidirectional=bidirectional)
        k = 2 if bidirectional else 1
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(k * hidden, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1]).squeeze(-1)


class Forecaster:
    """Công cụ dự báo của tác tử: lấy cửa sổ cuối cùng và dự báo giá phiên tới, kèm khoảng dự báo 90%.

    MC Dropout chỉ đo bất định của mô hình (rất hẹp với chuỗi lợi suất); mỗi mẫu được cộng thêm nhiễu phần dư
    N(0, RMSE kiểm thử) để khoảng dự báo và xác suất tăng phản ánh cả độ nhiễu của chính chuỗi giá.
    """

    def __init__(self, target, dev="cpu"):
        meta = json.loads((MODELS / f"forecast_{target}.json").read_text(encoding="utf-8"))
        self.meta, self.target, self.dev = meta, target, dev
        self.model = SeqForecaster(len(meta["cols"]), meta["hidden"], meta["layers"], meta["dropout"], True)
        self.model.load_state_dict(torch.load(MODELS / f"forecast_{target}.pt", map_location=dev))
        self.model.to(dev)

    def predict(self, n_samples=200):
        df = gold_frame() if self.target == "GOLD_VND" else market_frame()
        F = forecast_features(df, self.target, keep_last=True)
        cols, w = self.meta["cols"], self.meta["window"]
        x = (F[cols].to_numpy(np.float32)[-w:] - self.meta["mean"]) / self.meta["std"]
        xt = torch.tensor(x[None], dtype=torch.float32, device=self.dev).repeat(n_samples, 1, 1)
        self.model.train()                                      # bật dropout: mỗi lượt là một mẫu bất định mô hình
        torch.manual_seed(0)                                    # cùng đầu vào cho cùng khoảng dự báo ở app, tác tử, báo cáo
        with torch.no_grad():
            r_mc = self.model(xt).cpu().numpy() * self.meta["y_std"]
        self.model.eval()
        r = r_mc + np.random.default_rng(0).normal(0.0, self.meta["test"]["rmse_ret"], n_samples)
        p0 = float(F["price"].iloc[-1])
        prices = p0 * np.exp(r)
        return {"model": "BiLSTM (PyTorch)", "target": self.target, "last_date": F.index[-1].strftime("%Y-%m-%d"),
                "last_price": p0, "pred_price": float(p0 * np.exp(np.mean(r_mc))),
                "pred_return_pct": float(100 * np.mean(r_mc)), "mc_std_pct": float(100 * np.std(r_mc)),
                "interval90": [float(np.percentile(prices, 5)), float(np.percentile(prices, 95))],
                "prob_up": float(np.mean(r > 0)), "test_dir_acc": self.meta["test"]["dir_acc"],
                "test_rmse_vs_naive": self.meta["test"]["rmse_ratio"]}
