"""Serve giả lập (TL3_MOCK=1) để thử giao diện mọi `kind` khi các chương chưa có serve thật. Không dùng cho báo cáo."""
import math
import random
from types import SimpleNamespace

SPECS = {
    "ch2_diabetes": {"title": "Dự đoán tiểu đường", "kind": "tabular", "models": ["xgboost", "random_forest", "logreg"],
                     "inputs": [{"name": "gender", "label": "Giới tính", "type": "select", "options": ["Female", "Male"],
                                 "default": "Female"},
                                {"name": "age", "label": "Tuổi", "type": "number", "min": 1, "max": 90, "default": 54},
                                {"name": "bmi", "label": "BMI", "type": "number", "min": 10, "max": 60, "default": 27.3, "step": 0.1},
                                {"name": "HbA1c_level", "label": "HbA1c (%)", "type": "number", "default": 6.6, "step": 0.1}],
                     "example": {"gender": "Male", "age": 61, "bmi": 31.2, "HbA1c_level": 7.1}},
    "ch2_house": {"title": "Dự đoán giá nhà", "kind": "tabular", "models": ["ridge", "rf"],
                  "inputs": [{"name": "area", "label": "Diện tích", "type": "number", "default": 1800}],
                  "example": {"area": 2400}},
    "ch2_comments": {"title": "Phân loại nhận xét", "kind": "text", "models": ["tfidf_logreg"],
                     "inputs": [{"name": "text", "label": "Nhận xét", "type": "text", "default": "Love this dress!"}],
                     "example": {"text": "The fabric is cheap and it ripped after one wash."}},
    "ch3_mnist": {"title": "Nhận dạng chữ số MNIST", "kind": "image", "models": ["torch_improved", "scratch"],
                  "inputs": [{"name": "image", "label": "Ảnh chữ số", "type": "image", "canvas": True}], "example": {}},
    "ch4_aapl": {"title": "Dự báo giá AAPL", "kind": "series", "models": ["torch_lstm", "keras_gru"],
                 "inputs": [{"name": "horizon", "label": "Số phiên dự báo", "type": "number", "default": 7}],
                 "example": {"horizon": 10}},
    "ch5_diabetes": {"title": "Chatbot tiểu đường", "kind": "chat", "models": ["auto"],
                     "inputs": [{"name": "question", "label": "Câu hỏi", "type": "text"},
                                {"name": "mode", "label": "Chế độ", "type": "select", "options": ["auto", "graph", "rag"]}],
                     "example": {"question": "Tỷ lệ mắc tiểu đường ở nhóm trên 60 tuổi là bao nhiêu?", "mode": "auto"}},
    "ch6_cora": {"title": "Phân loại nút Cora", "kind": "graph", "models": ["gcn", "sage", "gat"],
                 "inputs": [{"name": "node_id", "label": "Mã nút", "type": "number", "default": 12}],
                 "example": {"node_id": 1701}},
}


def predict(exp_id, payload, model_id=None):
    kind = SPECS[exp_id]["kind"]
    r = random.Random(str(payload))
    if kind in ("tabular", "text", "image", "graph"):
        labels = {"image": [str(i) for i in range(10)], "graph": ["Neural_Networks", "Theory", "Genetic_Algorithms",
                                                                  "Reinforcement", "Probabilistic", "Case_Based", "Rule_Learning"],
                  "text": ["Không giới thiệu", "Giới thiệu"]}.get(kind, ["Không mắc", "Mắc"])
        if exp_id == "ch2_house":
            return {"value": 412345.6, "unit": "USD", "details": {"khoảng 90%": "[350k; 470k]"}}
        w = [r.random() ** 3 for _ in labels]
        s = sum(w)
        proba = {l: x / s for l, x in zip(labels, w)}
        out = {"label": max(proba, key=proba.get), "proba": proba, "details": {"ngưỡng": 0.42}}
        if kind == "graph":
            out["details"] = {"neighbors": [{"id": 100 + i, "label": labels[i % 7], "same": i % 2 == 0} for i in range(6)],
                              "degree": 6, "true_label": labels[0]}
        return out
    if kind == "series":
        h = int(payload.get("horizon", 7))
        hist = [150 + 10 * math.sin(i / 9) + i * 0.3 for i in range(90)]
        fc = [hist[-1] + 0.5 * (i + 1) for i in range(h)]
        return {"value": fc[0], "details": {"history": hist, "forecast": fc, "dates": None, "unit": "USD"}}
    return {"answer": "Trong dữ liệu, nhóm trên 60 tuổi có tỷ lệ mắc tiểu đường 19,8 % [1], cao hơn nhóm 40–60 [2].",
            "sources": [{"id": 1, "text": "AgeGroup_60+ has_diabetes_rate 0.198", "score": 0.91},
                        {"id": 2, "text": "AgeGroup_40-60 has_diabetes_rate 0.081", "score": 0.77}],
            "tools": ["graph_stats"], "trace": [{"step": "Thought", "text": "Câu hỏi về thống kê nhóm tuổi → công cụ đồ thị."},
                                                {"step": "Action", "text": "graph_stats(age_group='60+')"},
                                                {"step": "Observation", "text": "rate=0.198, n=12 034"}]}


def for_chapter(ch):
    specs = {k: v for k, v in SPECS.items() if k.startswith(ch + "_")}
    if not specs:
        raise ImportError(f"mock không có {ch}")
    return SimpleNamespace(SPECS=specs, predict=predict)
