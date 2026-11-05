"""
API thử nghiệm cho từng chương của báo cáo (trang /lab). Mô hình được nạp lười ở lần gọi đầu tiên.

  POST /api/lab/ch1/pima        hệ chuyên gia + perceptron + MLP (Pima)
  POST /api/lab/ch2/nhanes      7 bộ phân loại cổ điển (NHANES)
  POST /api/lab/ch3/retina      ResNet-18 phân độ võng mạc + Grad-CAM;  GET /api/lab/ch3/sample/{i}
  POST /api/lab/ch4/forecast    chỉ số CGM + dự báo LSTM/GRU;          GET /api/lab/ch4/sample/{i}
  POST /api/lab/ch5/search      truy xuất vector / full-text / lai / GraphRAG trên KB
  POST /api/lab/ch5/ask         hỏi đáp RAG qua hệ đa tác tử;          GET /api/lab/ch5/graph
  GET  /api/lab/ch6/paper       GAT phân loại bài báo Pubmed-Diabetes + láng giềng theo trọng số chú ý
  GET  /api/lab/summary         số liệu đánh giá chính của từng chương (lab/results/*.json)
"""

import asyncio
import base64
import io
import json
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from src.config import BASE_DIR
from src.knowledge.entity_linking import link_entities
from src.models.datasets import NHANES_FEATURES, NHANES_VI

router = APIRouter(prefix="/api/lab", tags=["lab"])
MODELS = BASE_DIR / "data" / "models"
RESULTS = BASE_DIR / "lab" / "results"


def _sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


# ------------------------------------------------------------------ Chương 1
class PimaIn(BaseModel):
    pregnancies: float = Field(0, ge=0, le=20)
    glucose: float = Field(..., ge=0, le=300, description="Glucose 2 giờ sau OGTT (mg/dL)")
    blood_pressure: float = Field(0, ge=0, le=200)
    skin_thickness: float = Field(0, ge=0, le=100)
    insulin: float = Field(0, ge=0, le=1000)
    bmi: float = Field(..., ge=0, le=80)
    pedigree: float = Field(0.3, ge=0, le=3)
    age: float = Field(..., ge=18, le=100)


@lru_cache(maxsize=1)
def _pima():
    return json.loads((MODELS / "ch1_pima.json").read_text(encoding="utf-8"))


RULE_CONDITIONS = [  # cùng thứ tự với RULES trong lab/ch1_history.py
    lambda r: r["glucose"] >= 200, lambda r: 140 <= r["glucose"] < 200, lambda r: r["bmi"] >= 30,
    lambda r: r["age"] >= 45, lambda r: r["pedigree"] >= 0.5, lambda r: r["pregnancies"] >= 4,
]


@router.post("/ch1/pima")
def ch1_pima(x: PimaIn) -> Dict[str, Any]:
    m = _pima()
    raw = x.model_dump()
    filled = {c: (m["median"][c] if c in m["zero_is_missing"] and raw[c] == 0 else raw[c]) for c in m["columns"]}
    fired, cf, trace = [], 0.0, []
    for (desc, c), cond in zip(m["rules"], RULE_CONDITIONS):
        if cond(filled):
            cf = cf + c * (1 - cf)  # gộp hệ số chắc chắn kiểu MYCIN
            fired.append([desc, c])
            trace.append(round(cf, 4))
    z = np.array([(filled[c] - m["mean"][c]) / m["std"][c] for c in m["columns"]])
    score = float(z @ np.array(m["perceptron"]["w"]) + m["perceptron"]["b"])
    h = _sigmoid(z @ np.array(m["mlp"]["W1"]) + np.array(m["mlp"]["b1"]))
    p = float(_sigmoid(h @ np.array(m["mlp"]["W2"]) + np.array(m["mlp"]["b2"]))[0])
    return {"expert": {"cf": round(cf, 4), "positive": cf >= 0.5, "fired": fired, "cf_trace": trace},
            "perceptron": {"score": round(score, 4), "positive": score > 0},
            "mlp": {"probability": round(p, 4), "positive": p >= 0.5},
            "imputed": [c for c in m["zero_is_missing"] if raw[c] == 0]}


# ------------------------------------------------------------------ Chương 2
class NhanesIn(BaseModel):
    age_years: float = Field(..., ge=18, le=90)
    sex: Optional[int] = Field(None, description="1 = nam, 0 = nữ")
    bmi: Optional[float] = None
    waist_cm: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    high_bp: Optional[int] = None
    high_chol: Optional[int] = None
    family_history: Optional[int] = None
    smoker: Optional[int] = None
    phys_activity: Optional[int] = None
    heart_disease: Optional[int] = None
    stroke: Optional[int] = None


CH2_NAMES = {"logreg": "Hồi quy logistic", "knn": "KNN (k=51)", "nb": "Naive Bayes", "tree": "Cây quyết định",
             "rf": "Rừng ngẫu nhiên", "gbdt": "Gradient Boosting", "svm": "SVM (RBF)"}


@lru_cache(maxsize=1)
def _ch2():
    models = joblib.load(MODELS / "ch2_classic.joblib")
    res = json.loads((RESULTS / "ch2.json").read_text(encoding="utf-8")) if (RESULTS / "ch2.json").exists() else {}
    return models, res.get("classification", {})


@router.post("/ch2/nhanes")
def ch2_nhanes(x: NhanesIn) -> Dict[str, Any]:
    models, metrics = _ch2()
    row = pd.DataFrame([{f: (np.nan if getattr(x, f) is None else float(getattr(x, f))) for f in NHANES_FEATURES}])
    out = {}
    for key, model in models.items():
        if key == "svm":
            p = float(_sigmoid(model.decision_function(row))[0])
        else:
            p = float(model.predict_proba(row)[0, 1])
        thr = metrics.get(key, {}).get("threshold", 0.5)
        out[key] = {"name": CH2_NAMES[key], "probability": round(p, 4), "threshold": thr, "positive": p >= thr,
                    "test_roc_auc": metrics.get(key, {}).get("roc_auc")}
    lr = models["logreg"]
    z = lr[:-1].transform(row)[0]
    contrib = sorted(((NHANES_VI[f], round(float(c * v), 4)) for f, c, v in
                      zip(NHANES_FEATURES, lr[-1].coef_[0][:len(NHANES_FEATURES)], z[:len(NHANES_FEATURES)])),
                     key=lambda t: -abs(t[1]))
    return {"models": out, "logreg_contributions": contrib[:6],
            "missing": [f for f in NHANES_FEATURES if getattr(x, f) is None]}


# ------------------------------------------------------------------ Chương 3
@router.post("/ch3/retina")
async def ch3_retina(file: UploadFile = File(...)) -> Dict[str, Any]:
    from src.models.cnn_vision import get_grader
    grader = get_grader()
    if grader is None:
        raise HTTPException(503, "Chưa có mô hình ch3_resnet18.pt")
    data = await file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "Ảnh tối đa 10 MB")
    t0 = time.perf_counter()
    try:
        out = await asyncio.to_thread(grader.predict, data)
    except Exception as exc:  # noqa: BLE001 — ảnh hỏng / không đọc được
        raise HTTPException(400, f"Không đọc được ảnh: {exc}")
    return {**out, "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@lru_cache(maxsize=1)
def _ch3_samples():
    return np.load(MODELS / "ch3_samples.npz", allow_pickle=False)


@router.get("/ch3/sample/{i}")
def ch3_sample(i: int) -> Dict[str, Any]:
    from PIL import Image
    s = _ch3_samples()
    if not 0 <= i < len(s["labels"]):
        raise HTTPException(404, f"Chỉ có {len(s['labels'])} ảnh mẫu")
    buf = io.BytesIO()
    Image.fromarray(s["images"][i]).save(buf, format="PNG")
    return {"index": i, "true_grade": int(s["labels"][i]), "image_png": base64.b64encode(buf.getvalue()).decode(),
            "source": "RetinaMNIST — tập kiểm tra"}


# ------------------------------------------------------------------ Chương 4
class ForecastIn(BaseModel):
    glucose: List[float] = Field(..., min_length=5, max_length=2000, description="mg/dL, cách nhau 15 phút")
    last_hour: Optional[float] = Field(None, ge=0, lt=24, description="giờ của điểm đo cuối (0–24)")


@router.post("/ch4/forecast")
def ch4_forecast(x: ForecastIn) -> Dict[str, Any]:
    from src.models.rnn_temporal import RNNTemporalEngine
    from datetime import datetime
    last = None
    if x.last_hour is not None:
        last = datetime.now().replace(hour=int(x.last_hour), minute=int((x.last_hour % 1) * 60))
    return RNNTemporalEngine().analyze_glucose_trajectory(x.glucose, last)


@router.get("/ch4/sample/{i}")
def ch4_sample(i: int) -> Dict[str, Any]:
    samples = json.loads((MODELS / "ch4_samples.json").read_text(encoding="utf-8"))
    if not 0 <= i < len(samples):
        raise HTTPException(404, f"Chỉ có {len(samples)} đoạn mẫu")
    return samples[i]


# ------------------------------------------------------------------ Chương 5
class SearchIn(BaseModel):
    query: str = Field(..., min_length=2, max_length=500)
    mode: str = Field("hybrid", pattern="^(vector|fulltext|hybrid|graphrag)$")
    k: int = Field(5, ge=1, le=10)


@lru_cache(maxsize=1)
def _kg_nodes() -> List[str]:
    triples = json.loads((BASE_DIR / "data" / "kb_sources" / "curated_triples.json").read_text(encoding="utf-8"))
    return sorted({t["head"] for t in triples} | {t["tail"] for t in triples})


def _search(q: SearchIn) -> Dict[str, Any]:
    from src.knowledge.embedder import get_embedder
    from src.storage.db import session_scope
    from src.storage.graph_store import get_graph_store
    from src.storage.vector_store import get_vector_store
    t0 = time.perf_counter()
    graph = None
    with session_scope() as s:
        store = get_vector_store(s)
        if q.mode == "fulltext":
            rows = store.fulltext_search(q.query, k=q.k)
        else:
            vec = get_embedder().embed([q.query])[0]
            if q.mode == "vector":
                rows = store.vector_search(vec, k=q.k)
            else:
                rows = store.hybrid_search(q.query, vec, k=q.k)
        if q.mode == "graphrag":
            ents = link_entities(q.query, _kg_nodes())
            edges = get_graph_store(s).neighborhood(ents, max_hops=1, limit=30) if ents else []
            ids = [r["id"] for r in rows]
            extra = [c for e in edges for c in e["chunk_ids"] if c not in ids]
            extra = list(dict.fromkeys(extra))[: max(0, q.k - 2)]
            linked = store.chunks_by_ids(extra)
            # GraphRAG: giữ 2 đoạn tốt nhất của truy xuất lai, bổ sung đoạn được nối từ đồ thị
            rows = rows[:2] + [{**c, "score": None, "via": "graph"} for c in linked] + rows[2:]
            rows = rows[: q.k]
            graph = {"entities": ents, "edges": [f"{e['head']} —{e['relation']}→ {e['tail']}" for e in edges[:15]]}
    return {"mode": q.mode, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), "graph": graph,
            "results": [{"chunk_id": r["id"], "title": r["title"], "heading": r["heading"],
                         "document_id": r["document_id"], "content": r["content"],
                         "score": None if r.get("score") is None else round(float(r["score"]), 4),
                         "via": r.get("via", q.mode)} for r in rows]}


@router.post("/ch5/search")
async def ch5_search(q: SearchIn) -> Dict[str, Any]:
    return await asyncio.to_thread(_search, q)


class AskIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)


@router.post("/ch5/ask")
async def ch5_ask(q: AskIn) -> Dict[str, Any]:
    """Hỏi đáp RAG đầy đủ qua hệ đa tác tử (Orchestrator → KnowledgeAgent → ResponseAgent)."""
    from src.api.deps import get_orchestrator
    from src.schemas.patient_profile import PatientProfile
    t0 = time.perf_counter()
    out = await get_orchestrator().run_turn(PatientProfile(), q.question, [])
    kb = out["data"].get("knowledge") or {}
    return {"answer": out["text"], "intent": out["data"].get("intent"), "citations": kb.get("citations", []),
            "passages": [{k: p[k] for k in ("title", "heading", "content")} for p in kb.get("passages", [])],
            "latency_s": round(time.perf_counter() - t0, 2)}


@router.get("/ch5/graph")
def ch5_graph() -> Dict[str, Any]:
    triples = json.loads((BASE_DIR / "data" / "kb_sources" / "curated_triples.json").read_text(encoding="utf-8"))
    nodes = {}
    for t in triples:
        nodes[t["head"]] = t["head_label"]
        nodes[t["tail"]] = t["tail_label"]
    return {"nodes": [{"id": n, "label": l} for n, l in nodes.items()],
            "edges": [{"source": t["head"], "target": t["tail"], "relation": t["relation"],
                       "evidence": t["evidence"], "source_doc": t["source_doc"]} for t in triples]}


# ------------------------------------------------------------------ Chương 6
@router.get("/ch6/paper")
async def ch6_paper(pid: Optional[str] = None, seed: Optional[int] = None) -> Dict[str, Any]:
    from src.models.deep_graph import get_pubmed_classifier
    clf = await asyncio.to_thread(get_pubmed_classifier)
    if clf is None:
        raise HTTPException(503, "Chưa có mô hình ch6_gat.pt")
    try:
        return clf.paper(pid, seed)
    except KeyError:
        raise HTTPException(404, f"Không có bài báo {pid} trong đồ thị Pubmed-Diabetes")


# ------------------------------------------------------------------ tổng hợp
@router.get("/summary")
def summary() -> Dict[str, Any]:
    out = {}
    for f in sorted(RESULTS.glob("ch*.json")):
        if f.stem.endswith("answers"):
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        out[f.stem] = {k: d[k] for k in d if k in ("dataset", "cv10", "classification", "models", "results",
                                                   "semi_supervised", "full_supervised", "k1", "k2", "k3",
                                                   "generation", "kg")}
    return out
