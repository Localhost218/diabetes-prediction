"""
Học sâu (Deep Learning): mạng nơ-ron truyền thẳng nhiều lớp (MLP) bằng PyTorch cho dữ liệu bảng.

Đầu vào = đặc trưng đã chuẩn hóa (giá trị thiếu thay bằng 0 = trung bình) + cờ "thiếu" cho từng đặc trưng,
để mạng phân biệt được "không biết" với "giá trị trung bình". Huấn luyện với early stopping theo ROC-AUC
trên tập validation. Chạy trên CPU.
"""

from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from src.config import settings
from src.models.datasets import DatasetSpec


class MLPNet(nn.Module):
    def __init__(self, n_inputs: int, hidden=(128, 64, 32), dropout: float = 0.2):
        super().__init__()
        layers, prev = [], n_inputs
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MLPRisk:
    kind = "DL"
    algorithm = "MLP 3 lớp ẩn (PyTorch)"

    def __init__(self, spec: DatasetSpec, net: MLPNet, mean: np.ndarray, std: np.ndarray, base_rate: float,
                 metrics: Dict[str, Any] = None):
        self.spec, self.net, self.mean, self.std, self.base_rate = spec, net, mean, std, base_rate
        self.metrics = metrics or {}
        self.net.eval()

    @staticmethod
    def path(spec: DatasetSpec):
        return settings.model_dir / f"{spec.name}_mlp.pt"

    def _encode(self, X: pd.DataFrame) -> torch.Tensor:
        values = X[self.spec.features].to_numpy(dtype=np.float32)
        missing = np.isnan(values).astype(np.float32)
        z = np.nan_to_num((values - self.mean) / self.std, nan=0.0).astype(np.float32)
        return torch.from_numpy(np.concatenate([z, missing], axis=1))

    @torch.no_grad()
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return torch.sigmoid(self.net(self._encode(X))).numpy()

    @classmethod
    def train(cls, spec: DatasetSpec, X: pd.DataFrame, y: pd.Series, X_val: pd.DataFrame, y_val: pd.Series,
              epochs: int = 80, patience: int = 8, batch_size: int = 512, seed: int = 42) -> "MLPRisk":
        torch.manual_seed(seed)
        values = X[spec.features].to_numpy(dtype=np.float32)
        mean, std = np.nanmean(values, axis=0), np.nanstd(values, axis=0) + 1e-6
        model = cls(spec, MLPNet(2 * len(spec.features)), mean, std, float(y.mean()))

        x_train, y_train = model._encode(X), torch.tensor(y.to_numpy(), dtype=torch.float32)
        x_val = model._encode(X_val)
        optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        loss_fn = nn.BCEWithLogitsLoss()

        best_auc, best_state, stale, history = -1.0, None, 0, []
        for epoch in range(epochs):
            model.net.train()
            perm = torch.randperm(len(x_train))
            for i in range(0, len(perm), batch_size):
                idx = perm[i:i + batch_size]
                if len(idx) < 2:
                    continue
                optimizer.zero_grad()
                loss = loss_fn(model.net(x_train[idx]), y_train[idx])
                loss.backward()
                optimizer.step()
            model.net.eval()
            with torch.no_grad():
                val_auc = roc_auc_score(y_val, torch.sigmoid(model.net(x_val)).numpy())
            history.append(round(float(val_auc), 4))
            scheduler.step(val_auc)
            if val_auc > best_auc + 1e-4:
                best_auc, stale = val_auc, 0
                best_state = {k: v.clone() for k, v in model.net.state_dict().items()}
            else:
                stale += 1
                if stale >= patience:
                    break
        model.net.load_state_dict(best_state)
        model.net.eval()
        model.metrics = {"epochs_trained": len(history), "best_val_auc": round(float(best_auc), 4)}
        return model

    def save(self) -> None:
        settings.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": self.net.state_dict(), "n_inputs": 2 * len(self.spec.features),
                    "mean": self.mean, "std": self.std, "base_rate": self.base_rate, "metrics": self.metrics,
                    "saved_at": datetime.now().isoformat(timespec="seconds")}, self.path(self.spec))

    @classmethod
    def load(cls, spec: DatasetSpec) -> "MLPRisk":
        bundle = torch.load(cls.path(spec), weights_only=False)
        net = MLPNet(bundle["n_inputs"])
        net.load_state_dict(bundle["state_dict"])
        return cls(spec, net, bundle["mean"], bundle["std"], bundle["base_rate"], bundle["metrics"])
