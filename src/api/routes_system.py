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
    result = await get_orchestrator().run_turn(profile, "", [])
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
    if not llm.enabled or llm.provider != "gemini":
        raise HTTPException(501, "Chuyển giọng nói thành văn bản cần cấu hình Gemini (LLM_PROVIDER=gemini)")
    data = await audio.read()
    mime = (audio.content_type or "audio/webm").split(";")[0]
    text_out = await llm.generate("Chép lại chính xác lời nói tiếng Việt trong đoạn ghi âm. Chỉ trả về văn bản đã chép, "
                                  "không thêm lời bình.", [Blob(data, mime)])
    if text_out is None:
        raise HTTPException(502, "Không chuyển được giọng nói thành văn bản, vui lòng thử lại")
    return {"text": text_out}
