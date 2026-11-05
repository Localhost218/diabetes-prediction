"""Chương 2 — hàm suy luận cho Web App: dự đoán tiểu đường, giá nhà, nhận xét khách hàng.
Mô hình nạp lười (lazy) một lần từ models/ch2/; không phụ thuộc Flask."""
import time
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from .common import MODELS, load_json

M = MODELS / "ch2"

DIAB_MODELS = ["xgb", "lgbm", "rf", "logreg", "tree", "knn", "nb", "mlp"]
HOUSE_MODELS = ["lasso", "linear", "ridge", "xgb", "poly", "rf", "svr", "knn", "tree"]
TEXT_MODELS = ["logreg", "svm", "nb", "cnb"]

SPECS = {
    "ch2_diabetes": {
        "title": "Dự đoán nguy cơ tiểu đường", "kind": "tabular",
        "inputs": [
            {"name": "gender", "label": "Giới tính", "type": "select", "options": ["Female", "Male"], "default": "Female"},
            {"name": "age", "label": "Tuổi", "type": "number", "min": 1, "max": 100, "default": 54},
            {"name": "hypertension", "label": "Tăng huyết áp (0/1)", "type": "select", "options": [0, 1], "default": 0},
            {"name": "heart_disease", "label": "Bệnh tim (0/1)", "type": "select", "options": [0, 1], "default": 0},
            {"name": "smoking_history", "label": "Tiền sử hút thuốc", "type": "select",
             "options": ["never", "No Info", "former", "current", "not current", "ever"], "default": "never"},
            {"name": "bmi", "label": "BMI", "type": "number", "min": 10, "max": 80, "default": 29.5},
            {"name": "HbA1c_level", "label": "HbA1c (%)", "type": "number", "min": 3.5, "max": 9, "default": 6.6},
            {"name": "blood_glucose_level", "label": "Đường huyết (mg/dL)", "type": "number", "min": 80,
             "max": 300, "default": 160}],
        "models": DIAB_MODELS,
        "example": {"gender": "Male", "age": 62, "hypertension": 1, "heart_disease": 0,
                    "smoking_history": "former", "bmi": 31.2, "HbA1c_level": 7.0, "blood_glucose_level": 200}},
    "ch2_house": {
        "title": "Dự đoán giá nhà", "kind": "tabular",
        "inputs": [
            {"name": "Area", "label": "Diện tích (ft²)", "type": "number", "min": 500, "max": 6000, "default": 3200},
            {"name": "Bedrooms", "label": "Phòng ngủ", "type": "number", "min": 1, "max": 5, "default": 3},
            {"name": "Bathrooms", "label": "Phòng tắm", "type": "number", "min": 1, "max": 3, "default": 2},
            {"name": "Stories", "label": "Số tầng", "type": "number", "min": 1, "max": 3, "default": 2},
            {"name": "Parking", "label": "Chỗ đỗ xe", "type": "number", "min": 0, "max": 2, "default": 1},
            {"name": "Age", "label": "Tuổi nhà (năm)", "type": "number", "min": 0, "max": 49, "default": 10},
            {"name": "Locality Rating", "label": "Điểm khu vực (1–10)", "type": "number", "min": 1, "max": 10, "default": 7},
            {"name": "City", "label": "Thành phố", "type": "select",
             "options": ["Mumbai", "Hyderabad", "Kolkata", "Pune", "Delhi", "Bangalore", "Chennai"], "default": "Pune"},
            {"name": "Furnishing", "label": "Nội thất", "type": "select",
             "options": ["Furnished", "Semi-Furnished", "Unfurnished"], "default": "Semi-Furnished"},
            {"name": "Main Road", "label": "Mặt đường chính", "type": "select", "options": ["Yes", "No"], "default": "Yes"},
            {"name": "Guest Room", "label": "Phòng khách riêng", "type": "select", "options": ["Yes", "No"], "default": "No"},
            {"name": "Basement", "label": "Tầng hầm", "type": "select", "options": ["Yes", "No"], "default": "No"},
            {"name": "Water Supply", "label": "Nguồn nước", "type": "select",
             "options": ["Both", "Corporation", "Borewell"], "default": "Both"},
            {"name": "Air Conditioning", "label": "Điều hòa", "type": "select", "options": ["Yes", "No"], "default": "Yes"},
            {"name": "Preferred Tenant", "label": "Người thuê ưu tiên", "type": "select",
             "options": ["Family", "Bachelor", "Company"], "default": "Family"}],
        "models": HOUSE_MODELS,
        "example": {"Area": 3200, "Bedrooms": 3, "Bathrooms": 2, "Stories": 2, "Parking": 1, "Age": 10,
                    "Locality Rating": 7, "City": "Pune", "Furnishing": "Semi-Furnished", "Main Road": "Yes",
                    "Guest Room": "No", "Basement": "No", "Water Supply": "Both", "Air Conditioning": "Yes",
                    "Preferred Tenant": "Family"}},
    "ch2_comments": {
        "title": "Phân loại nhận xét khách hàng", "kind": "text",
        "inputs": [{"name": "text", "label": "Nhận xét (tiếng Anh)", "type": "text",
                    "default": "The fabric feels cheap and the fit is too small, I am returning it."}],
        "models": TEXT_MODELS,
        "example": {"text": "Love this dress! The fabric is soft, it fits perfectly and the color is gorgeous."}},
}


@lru_cache(maxsize=None)
def _load(name):
    return joblib.load(M / f"{name}.joblib")


@lru_cache(maxsize=None)
def _diab_mlp():
    import torch

    from .ch2_mlp import TabularMLP
    pre = _load("diabetes_pre")
    model = TabularMLP(len(pre.get_feature_names_out()))
    model.load_state_dict(torch.load(M / "diabetes_mlp.pt", map_location="cpu"))
    return pre, model.eval()


@lru_cache(maxsize=None)
def _house_mlp():
    import torch

    from .ch2_mlp import TabularMLP
    ck = torch.load(M / "house_mlp.pt", map_location="cpu")
    model = TabularMLP(ck["n_in"], hidden=(64, 32), dropout=0.1)
    model.load_state_dict(ck["state"])
    return _load("house_pre"), model.eval(), ck["mu"], ck["sd"]


def _default(exp_id):
    name = load_json(exp_id)["best"]["name"]
    return name if name in SPECS[exp_id]["models"] else SPECS[exp_id]["models"][0]


def predict_diabetes(payload, model_id):
    """Trả nhãn, xác suất và ngưỡng quyết định (chọn theo F1 trên Validation) của mô hình được chọn."""
    row = pd.DataFrame([{k: payload[k] for k in SPECS["ch2_diabetes"]["example"]}])
    if model_id == "mlp":
        import torch  # chỉ MLP cần torch; nạp torch mất hàng chục giây nên không nạp cho mô hình cổ điển
        pre, model = _diab_mlp()
        with torch.no_grad():
            p = float(torch.sigmoid(model(torch.tensor(pre.transform(row), dtype=torch.float32)))[0])
    else:
        p = float(_load(f"diabetes_{model_id}").predict_proba(row)[0, 1])
    thr = float(_load("diabetes_meta")["thresholds"][model_id])
    label = "Nguy cơ tiểu đường" if p >= thr else "Không tiểu đường"
    return {"label": label, "proba": {"Không tiểu đường": 1 - p, "Nguy cơ tiểu đường": p},
            "details": {"threshold": thr, "probability": p}}


def predict_house(payload, model_id):
    """Trả giá dự đoán (đơn vị tiền tệ gốc của dữ liệu)."""
    row = pd.DataFrame([{k: payload[k] for k in SPECS["ch2_house"]["example"]}])
    if model_id == "mlp":
        import torch  # chỉ MLP cần torch
        pre, model, mu, sd = _house_mlp()
        with torch.no_grad():
            v = float(model(torch.tensor(pre.transform(row), dtype=torch.float32))[0]) * sd + mu
    else:
        v = float(_load(f"house_{model_id}").predict(row)[0])
    return {"value": round(v, 0), "details": {"unit": "đơn vị tiền (INR)"}}


def predict_comments(payload, model_id):
    """Trả nhãn Giới thiệu / Không giới thiệu và xác suất."""
    pipe = _load(f"comments_{model_id}")
    p = float(pipe.predict_proba([payload["text"]])[0, 1])
    label = "Giới thiệu sản phẩm" if p >= 0.5 else "Không giới thiệu"
    return {"label": label, "proba": {"Không giới thiệu": 1 - p, "Giới thiệu sản phẩm": p},
            "details": {"n_words": len(payload["text"].split())}}


def predict(exp_id, payload, model_id=None):
    """Điểm vào chung cho Web App: chọn hàm theo exp_id, mặc định dùng mô hình tốt nhất trên Validation."""
    fn = {"ch2_diabetes": predict_diabetes, "ch2_house": predict_house, "ch2_comments": predict_comments}[exp_id]
    model_id = model_id or _default(exp_id)
    t0 = time.perf_counter()
    out = fn(payload, model_id)
    out.update(model=model_id, latency_ms=1000 * (time.perf_counter() - t0))
    return out
