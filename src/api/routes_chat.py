"""
Endpoint chat: nhận text và/hoặc file bệnh án, chạy một lượt hội chẩn đa tác tử, lưu kết quả.
"""

import asyncio
import hashlib
import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import select

from src.api.deps import get_orchestrator, message_dict, session_dict
from src.config import settings
from src.document_processing.loaders import detect_kind
from src.schemas.patient_profile import PatientProfile
from src.storage.db import session_scope
from src.storage.models import AgentMessageLog, ChatMessage, ChatSession, Consultation, Document
from src.storage.object_storage import get_storage

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["chat"])
MAX_FILES = 5


class ChatRequest(BaseModel):
    session_id: Optional[str] = None
    text: str


@router.post("/chat")
async def chat(req: ChatRequest):
    if not req.text.strip():
        raise HTTPException(400, "Tin nhắn trống")
    return await _run_turn(req.session_id, req.text, [])


@router.post("/chat/upload")
async def chat_upload(files: List[UploadFile] = File(...), text: str = Form(""), session_id: Optional[str] = Form(None)):
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"Tối đa {MAX_FILES} file mỗi tin nhắn")
    uploads = []
    for f in files:
        data = await f.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"{f.filename} vượt quá {settings.max_upload_mb} MB")
        if not detect_kind(f.filename or "", f.content_type or ""):
            raise HTTPException(415, f"Định dạng không hỗ trợ: {f.filename}. Hỗ trợ PDF, DOCX, ảnh JPG/PNG/WEBP, CSV, TXT.")
        uploads.append((f.filename or "file", f.content_type or "application/octet-stream", data))
    return await _run_turn(session_id, text, uploads)


@router.get("/documents/{document_id}/content")
async def document_content(document_id: str):
    with session_scope() as s:
        doc = s.get(Document, document_id)
        if doc is None:
            raise HTTPException(404, "Không tìm thấy tài liệu")
        bucket, key, mime, name = doc.bucket, doc.object_key, doc.mime_type, doc.filename
    data = await asyncio.to_thread(get_storage().get, bucket, key)
    return Response(data, media_type=mime, headers={"Content-Disposition": f'inline; filename="{_ascii(name)}"'})


# ---------------------------------------------------------------------------
async def _run_turn(session_id: Optional[str], text: str, uploads: List[tuple]) -> Dict[str, Any]:
    with session_scope() as s:
        session = s.get(ChatSession, session_id) if session_id else None
        if session is None:
            session = ChatSession(title="Cuộc trò chuyện mới", profile={})
            s.add(session)
            s.flush()
        session_id = session.id
        profile = PatientProfile.model_validate(session.profile or {})
        history, last_assessment = _memory(s, session_id)

    documents = await _store_uploads(session_id, uploads)
    attachments = [{"document_id": d["document_id"], "filename": d["filename"], "mime_type": d["mime_type"]}
                   for d in documents]
    with session_scope() as s:
        user_msg = ChatMessage(session_id=session_id, sender="user", text=text, attachments=attachments)
        s.add(user_msg)
        s.flush()
        user_msg_id = user_msg.id

    try:
        result = await get_orchestrator().run_turn(profile, text, documents, history, last_assessment)
        bot_text, data, error = result["text"], _compact(result), None
    except Exception as exc:  # noqa: BLE001
        log.exception("Lượt hội chẩn thất bại")
        result, error = None, f"{type(exc).__name__}: {exc}"
        bot_text = "Xin lỗi, hệ thống gặp lỗi khi xử lý yêu cầu. Vui lòng thử lại."
        data = {"error": error}

    with session_scope() as s:
        session = s.get(ChatSession, session_id)
        if result is not None:
            session.profile = profile.model_dump()
        if session.title == "Cuộc trò chuyện mới":
            session.title = _title(text, documents)
        session.updated_at = datetime.now().astimezone()
        bot_msg = ChatMessage(session_id=session_id, sender="bot", text=bot_text, data=data)
        s.add(bot_msg)
        s.flush()
        if result is not None:
            _persist_result(s, session_id, bot_msg.id, profile, result)
        s.refresh(bot_msg)
        return {"session": session_dict(session), "user_message": message_dict(s.get(ChatMessage, user_msg_id)),
                "bot_message": message_dict(bot_msg)}


HISTORY_TURNS = 10


def _memory(s, session_id: str):
    """Bộ nhớ của phiên: các tin nhắn gần nhất + kết quả hội chẩn gần nhất (đọc từ PostgreSQL)."""
    rows = s.scalars(select(ChatMessage).where(ChatMessage.session_id == session_id)
                     .order_by(ChatMessage.created_at.desc()).limit(HISTORY_TURNS)).all()[::-1]
    history = []
    for m in rows:
        text = m.text
        if m.attachments:
            text = f"[Đính kèm: {', '.join(a['filename'] for a in m.attachments)}] {text}"
        history.append({"sender": m.sender, "text": text})
    last = s.scalar(select(ChatMessage).where(ChatMessage.session_id == session_id, ChatMessage.sender == "bot",
                                              ChatMessage.data["consensus"].isnot(None))
                    .order_by(ChatMessage.created_at.desc()).limit(1))
    last_assessment = None
    if last is not None:
        d = last.data
        last_assessment = {**d["consensus"], "treatment": d.get("treatment"), "citations": d.get("citations")}
    return history, last_assessment


async def _store_uploads(session_id: str, uploads: List[tuple]) -> List[Dict[str, Any]]:
    storage = get_storage()
    documents = []
    for filename, mime, data in uploads:
        sha = hashlib.sha256(data).hexdigest()
        with session_scope() as s:
            existing = s.scalar(select(Document).where(Document.session_id == session_id, Document.sha256 == sha))
            if existing is not None:
                documents.append(_doc_payload(existing))
                continue
        key = f"uploads/{session_id}/{uuid.uuid4().hex}-{_ascii(filename)}"
        await asyncio.to_thread(storage.put, settings.s3_bucket_uploads, key, data, mime)
        with session_scope() as s:
            doc = Document(session_id=session_id, filename=filename, mime_type=mime, size_bytes=len(data), sha256=sha,
                           bucket=settings.s3_bucket_uploads, object_key=key)
            s.add(doc)
            s.flush()
            documents.append(_doc_payload(doc))
    return documents


def _doc_payload(doc: Document) -> Dict[str, Any]:
    return {"document_id": doc.id, "filename": doc.filename, "mime_type": doc.mime_type,
            "bucket": doc.bucket, "object_key": doc.object_key}


def _persist_result(s, session_id: str, message_id: str, profile: PatientProfile, result: Dict[str, Any]) -> None:
    for d in result["data"].get("documents", []):
        doc = s.get(Document, d["document_id"])
        if doc is not None:
            doc.status = "failed" if d.get("error") else "parsed"
            doc.error = d.get("error")
            doc.extracted = d
    for m in result["transcript"]:
        s.add(AgentMessageLog(id=m["message_id"] + result["conversation_id"][:8], session_id=session_id,
                              conversation_id=m["conversation_id"], in_reply_to=m["in_reply_to"], sender=m["sender"],
                              receiver=m["receiver"], performative=m["performative"], msg_type=m["msg_type"],
                              payload=_jsonable(m["payload"]), elapsed_ms=m["elapsed_ms"],
                              created_at=datetime.fromisoformat(m["timestamp"]).astimezone()))
    consensus = result["data"].get("consensus")
    if consensus:
        s.add(Consultation(session_id=session_id, message_id=message_id, profile=profile.model_dump(),
                           risk_score=consensus["consensus_risk_score"], risk_level=consensus["risk_level"],
                           debate_rounds=consensus["debate_rounds"], result=_jsonable(result["data"])))


def _compact(result: Dict[str, Any]) -> Dict[str, Any]:
    """Dữ liệu gửi kèm tin nhắn bot cho giao diện (bản rút gọn; bản đầy đủ nằm trong consultations/agent_messages)."""
    d = result["data"]
    out: Dict[str, Any] = {
        "intent": d["intent"], "plan": result["plan"], "generated_by": result["generated_by"],
        "updated_fields": d.get("updated_fields", []), "documents": d.get("documents", []),
        "transcript": [{k: m[k] for k in ("sender", "receiver", "performative", "msg_type", "elapsed_ms")}
                       for m in result["transcript"]],
    }
    if "consensus" in d:
        c, ml = d["consensus"], d["prediction"]["risk_model"]
        out["consensus"] = {k: c[k] for k in ("risk_level", "risk_label", "consensus_risk_score", "glycemic_status",
                                              "confirmation_required", "positive_evidence_plus",
                                              "negative_contradictions_minus", "safety_guardrails", "debate_rounds")}
        out["consensus"]["debate"] = [{k: x[k] for k in ("round", "issue", "challenged", "challenge")}
                                      | {"statement": x["response"].get("statement")} for x in c["debate"]]
        out["ml"] = {k: ml[k] for k in ("risk_probability", "relative_risk", "top_risk_factors", "data_completeness",
                                        "model_metrics", "dataset", "disagreement")}
        out["ml"]["models"] = {k: {"name": v["model"], "kind": v["kind"], "risk_probability": v["risk_probability"]}
                               for k, v in ml["models"].items()}
        out["glycemia"] = d["knowledge"]["glycemia"]["findings"]
        out["graph_paths"] = d["knowledge"]["graph_rag"]["paths"][:12]
        out["citations"] = d["knowledge"]["citations"]
        out["treatment"] = {k: d["treatment"][k] for k in ("recommended_action", "intervention_details", "notes",
                                                           "overridden_by_consensus")}
        out["profile_summary"] = d.get("profile_summary", [])
    if "knowledge" in d and "consensus" not in d:
        out["citations"] = d["knowledge"].get("citations", [])
    if "missing_required" in d:
        out["missing_required"] = d["missing_required"]
    return _jsonable(out)


def _jsonable(obj: Any) -> Any:
    import json
    return json.loads(json.dumps(obj, ensure_ascii=False, default=lambda o: float(o) if hasattr(o, "__float__") else str(o)))


def _title(text: str, documents: List[Dict[str, Any]]) -> str:
    base = text.strip() or (f"Bệnh án: {documents[0]['filename']}" if documents else "Cuộc trò chuyện mới")
    return base[:60] + ("…" if len(base) > 60 else "")


def _ascii(name: str) -> str:
    import re
    import unicodedata
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9._-]+", "_", plain)[-120:] or "file"
