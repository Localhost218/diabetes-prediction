"""
Endpoint hệ thống: trạng thái, Knowledge Base, form đánh giá trực tiếp, Q-table, chuyển giọng nói thành văn bản.
"""

import asyncio
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text

from src.api.deps import get_orchestrator
from src.config import settings
from src.llm.client import Blob, llm
from src.schemas.patient_profile import PatientProfile
from src.storage.db import session_scope
from src.storage.graph_store import get_graph_store
from src.storage.models import ChatSession, Consultation, KbChunk, KbDocument

router = APIRouter(tags=["system"])


@router.get("/health")
def health():
    with session_scope() as s:
        s.execute(text("SELECT 1"))
    return {"status": "ok"}


# Mọi artifact trong notebook/model: tác tử nào dùng (None = chỉ thử nghiệm ở notebook, không chạy trong hệ tác tử)
MODEL_REGISTRY = [
    ("nhanes_gbdt.joblib", "HistGradientBoosting", "ML", "NHANES", "PredictionAgent · ensemble chính"),
    ("nhanes_mlp.pt", "PyTorch MLP 128-64-32", "DL", "NHANES", "PredictionAgent · ensemble chính"),
    ("brfss_gbdt.joblib", "HistGradientBoosting", "ML", "BRFSS 2021–2023", "PredictionAgent · ý kiến thứ hai"),
    ("brfss_mlp.pt", "PyTorch MLP", "DL", "BRFSS 2021–2023", "PredictionAgent · ý kiến thứ hai"),
    ("rnn_glucose_lstm.pt", "LSTM dự báo glucose", "DL · RNN", "Chuỗi CGM", "PredictionAgent · khi có ≥ 12 lần đo"),
    ("cnn_retinopathy_resnet18.pt", "ResNet-18 + Grad-CAM", "DL · CNN", "Ảnh đáy mắt", "PredictionAgent · khi có ảnh đáy mắt"),
    ("rl_qtable.npz", "Q-learning (24 trạng thái, γ = 0,9)", "RL", "Mô phỏng lâm sàng", "TreatmentRLAgent"),
    ("logistic_regression.joblib", "Logistic Regression", "ML", "NHANES", None),
    ("random_forest.joblib", "Random Forest", "ML", "NHANES", None),
    ("extra_trees.joblib", "Extra Trees", "ML", "NHANES", None),
    ("tabnet_attention.pt", "TabNet (attention)", "DL", "NHANES", None),
]


def _model_registry() -> List[Dict[str, Any]]:
    import csv

    board = {}
    path = settings.model_dir / "evaluation_leaderboard.csv"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if " + " not in row["Artifact (/notebook/model)"]:
                    board[row["Artifact (/notebook/model)"]] = row
    pred = get_orchestrator().bus.agents["PredictionAgent"]
    live = {"brfss_gbdt.joblib": pred.large and pred.large[1].metrics, "brfss_mlp.pt": pred.large and pred.large[2].metrics}
    out = []
    for file, name, kind, data, used_by in MODEL_REGISTRY:
        row, m = board.get(file), live.get(file) or {}
        num = lambda k, c: float(row[c]) if row and row.get(c) else m.get(k)  # noqa: E731
        out.append({"file": file, "name": name, "kind": kind, "dataset": data, "used_by": used_by,
                    "exists": (settings.model_dir / file).exists(),
                    "roc_auc": num("roc_auc", "ROC-AUC"), "pr_auc": num("pr_auc", "PR-AUC"), "brier": num("brier", "Brier Score")})
    return out


@router.get("/api/system")
def system_info():
    orc = get_orchestrator()
    pred = orc.bus.agents["PredictionAgent"]
    return {
        "agents": sorted(orc.bus.agents),
        "llm": {"enabled": llm.enabled, "provider": llm.provider, "model": llm.model,
                "fallback_models": list(settings.llm_fallback_models)},
        "storage": {"backend": settings.storage_backend, "endpoint": settings.s3_endpoint_url},
        "knowledge_graph": settings.kg_backend,
        "embedding": orc.bus.agents["KnowledgeAgent"].embedder.name,
        "prediction_dataset": pred.spec.title,
        "risk_models": {"ml": pred.gbdt.metrics, "dl": pred.mlp.metrics},
        "model_registry": _model_registry(),
    }


@router.get("/api/kb/stats")
def kb_stats():
    with session_scope() as s:
        docs = s.scalars(select(KbDocument).order_by(KbDocument.id)).all()
        return {
            "documents": [{"id": d.id, "title": d.title, "source": d.source, "url": d.url,
                           "object_key": d.object_key, "chunks": len(d.chunks)} for d in docs],
            "chunks": s.scalar(select(func.count()).select_from(KbChunk)),
            "graph": get_graph_store(s).stats(),
            "graph_backend": settings.kg_backend,
        }


@router.get("/api/kb/graph")
def kb_graph(center: str = "Đái tháo đường típ 2", hops: int = 2, limit: int = 60):
    """Vùng lân cận của một nút trong đồ thị tri thức để giao diện vẽ (tab Tri thức KB)."""
    with session_scope() as s:
        edges = get_graph_store(s).neighborhood([center], max_hops=max(1, min(hops, 3)), limit=max(1, min(limit, 200)))
    keys = ("head", "head_label", "relation", "tail", "tail_label", "hops")
    return {"entities": [center], "edges": [{k: e.get(k) for k in keys} for e in edges]}


class KbQuery(BaseModel):
    query: str
    k: int = 5


@router.post("/api/kb/search")
async def kb_search(q: KbQuery):
    knowledge = get_orchestrator().bus.agents["KnowledgeAgent"]
    return {"passages": await asyncio.to_thread(knowledge._search, q.query, q.k)}


class ConsultationForm(BaseModel):
    """Form nhập đủ chỉ số (dùng cho kiểm thử / tích hợp). Tên trường theo PatientProfile."""
    age_years: int = Field(..., ge=1, le=120)
    bmi: Optional[float] = None
    height_cm: Optional[float] = None
    weight_kg: Optional[float] = None
    sex: Optional[str] = Field(None, description="nam | nữ")
    waist_cm: Optional[float] = None
    glucose_fasting_mgdl: Optional[float] = None
    hba1c_percent: Optional[float] = None
    ogtt_2h_mgdl: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    high_bp: Optional[bool] = None
    high_chol: Optional[bool] = None
    family_history: Optional[bool] = None
    smoker: Optional[bool] = None
    phys_activity: Optional[bool] = None
    heart_disease: Optional[bool] = None
    glucose_history: Optional[List[float]] = None
    fundus_features: Optional[Dict[str, Any]] = None


@router.post("/api/consultation")
async def consultation(form: ConsultationForm):
    profile = PatientProfile()
    for name, value in form.model_dump(exclude={"glucose_history", "fundus_features"}).items():
        if value is not None:
            profile.set(name, value, "form")
    profile.glucose_history, profile.glucose_history_source = form.glucose_history, "form"
    profile.fundus_features = form.fundus_features
    profile.derive()
    if not profile.is_ready():
        raise HTTPException(422, f"Thiếu dữ liệu bắt buộc: {', '.join(profile.missing_required())}")
    result = await get_orchestrator().run_turn(profile, "Đánh giá nguy cơ đái tháo đường từ các chỉ số đã nhập.", [])
    return {"text": result["text"], "plan": result["plan"], "data": result["data"], "transcript": result["transcript"]}


@router.get("/api/agents/qtable")
def qtable():
    engine = get_orchestrator().bus.agents["TreatmentRLAgent"].rl_engine
    return {"q_table": engine.q_table.round(3).tolist(), "states": engine.n_states, "actions": engine.n_actions}


@router.get("/api/stats")
def stats():
    with session_scope() as s:
        levels = dict(s.execute(select(Consultation.risk_level, func.count()).group_by(Consultation.risk_level)).all())
        return {"sessions": s.scalar(select(func.count()).select_from(ChatSession)),
                "consultations": sum(levels.values()), "by_risk_level": levels}


@router.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    if not llm.enabled:
        raise HTTPException(501, "Chuyển giọng nói thành văn bản chưa khả dụng với cấu hình LLM hiện tại")
    data = await audio.read()
    mime = (audio.content_type or "audio/webm").split(";")[0]
    text_out = await llm.generate("Chép lại chính xác lời nói tiếng Việt trong đoạn ghi âm. Chỉ trả về văn bản đã chép, "
                                  "không thêm lời bình.", [Blob(data, mime)])
    if text_out is None:
        raise HTTPException(501, "Model LLM hiện tại (Ollama/Qwen text) không hỗ trợ trực tiếp đầu vào âm thanh; vui lòng dùng Web Speech API trên trình duyệt")
    return {"text": text_out}
