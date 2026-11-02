"""
Chương 3: CNN phân độ bệnh võng mạc đái tháo đường (DR) từ ảnh đáy mắt, thang ICDR 0–4.

Mô hình ResNet-18 học chuyển giao từ ImageNet, tinh chỉnh trên RetinaMNIST 128×128 (lab/ch3_cnn.py →
data/models/ch3_resnet18.pt). Kèm Grad-CAM để chỉ ra vùng ảnh quyết định dự đoán.
"""

import base64
import io
from functools import lru_cache
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.config import BASE_DIR

MODEL_PATH = BASE_DIR / "data" / "models" / "ch3_resnet18.pt"
MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])

DR_CLASSES = [
    {"stage": 0, "code": "NO_DR", "name_vi": "Không có bệnh võng mạc ĐTĐ",
     "advice": "Khám lại mắt sau 1–2 năm nếu đường huyết kiểm soát tốt (ADA 2024)."},
    {"stage": 1, "code": "MILD_NPDR", "name_vi": "Bệnh võng mạc không tăng sinh nhẹ",
     "advice": "Khám mắt ít nhất mỗi năm, kiểm soát đường huyết và huyết áp."},
    {"stage": 2, "code": "MODERATE_NPDR", "name_vi": "Bệnh võng mạc không tăng sinh trung bình",
     "advice": "Khám chuyên khoa mắt trong vòng 3–6 tháng."},
    {"stage": 3, "code": "SEVERE_NPDR", "name_vi": "Bệnh võng mạc không tăng sinh nặng",
     "advice": "Chuyển bác sĩ nhãn khoa sớm."},
    {"stage": 4, "code": "PDR", "name_vi": "Bệnh võng mạc tăng sinh",
     "advice": "Chuyển bác sĩ nhãn khoa ngay — nguy cơ mất thị lực."},
]


def build_resnet18() -> nn.Module:
    from torchvision import models
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 5))
    return m


def preprocess(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB").resize((128, 128), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(((arr - MEAN) / STD).transpose(2, 0, 1).astype(np.float32))[None]


class RetinaGrader:
    def __init__(self, path=MODEL_PATH):
        self.model = build_resnet18()
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()

    def grad_cam(self, x: torch.Tensor, cls: int) -> np.ndarray:
        store = {}

        def keep(module, inputs, output):
            store["a"] = output
            output.register_hook(lambda g: store.setdefault("g", g))

        handle = self.model.layer4.register_forward_hook(keep)
        out = self.model(x)
        self.model.zero_grad()
        out[0, cls].backward()
        handle.remove()
        w = store["g"].mean((2, 3), keepdim=True)
        cam = F.relu((w * store["a"]).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return (cam / (cam.max() + 1e-9)).detach().numpy()

    def predict(self, image_bytes: bytes, with_cam: bool = True) -> Dict[str, Any]:
        img = Image.open(io.BytesIO(image_bytes))
        x = preprocess(img)
        with torch.no_grad():
            prob = F.softmax(self.model(x), 1)[0].numpy()
        grade = int(prob.argmax())
        out = {"grade": grade, "label": f"{grade} – {DR_CLASSES[grade]['name_vi']}",
               "probabilities": [round(float(p), 4) for p in prob], "advice": DR_CLASSES[grade]["advice"],
               "model": "ResNet-18 (ImageNet → RetinaMNIST)"}
        if with_cam:
            out["gradcam_png"] = overlay_png(img, self.grad_cam(x.clone().requires_grad_(True), grade))
        return out


def overlay_png(img: Image.Image, cam: np.ndarray) -> str:
    """Phủ bản đồ nhiệt Grad-CAM (thang màu jet tự tính, không cần matplotlib) lên ảnh 128×128."""
    base = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
    r = np.clip(1.5 - np.abs(4 * cam - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * cam - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * cam - 1), 0, 1)
    heat = np.stack([r, g, b], -1)
    mix = (0.55 * base + 0.45 * heat) * 255
    buf = io.BytesIO()
    Image.fromarray(mix.astype(np.uint8)).resize((256, 256), Image.BILINEAR).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


@lru_cache(maxsize=1)
def get_grader() -> Optional[RetinaGrader]:
    return RetinaGrader() if MODEL_PATH.exists() else None


class CNNRetinopathyEngine:
    """Giao diện dùng bởi PredictionAgent: chạy CNN khi hồ sơ có ảnh đáy mắt (base64)."""

    def analyze_fundus(self, fundus: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        grader = get_grader()
        image_b64 = (fundus or {}).get("image_b64")
        if grader is None or not image_b64:
            return {"skipped": True, "reason": "Cần ảnh đáy mắt (image_b64) và mô hình ch3_resnet18.pt"}
        res = grader.predict(base64.b64decode(image_b64), with_cam=False)
        return {**res, "note": "Kết quả tham khảo từ CNN, không thay thế khám nhãn khoa"}
