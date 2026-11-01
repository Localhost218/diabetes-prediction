"""
Quản lý phiên chat (không cần đăng nhập).
"""

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, Body, HTTPException
from sqlalchemy import select

from src.api.deps import session_dict
from src.storage.db import session_scope
from src.storage.models import ChatMessage, ChatSession, Document
from src.storage.object_storage import get_storage

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


@router.get("")
def list_sessions():
    with session_scope() as s:
        rows = s.scalars(select(ChatSession).order_by(ChatSession.updated_at.desc()).limit(100)).all()
        return [session_dict(r) for r in rows]


@router.post("")
def create_session(payload: Dict[str, Any] = Body(default={})):
    with session_scope() as s:
        session = ChatSession(title=(payload.get("title") or "Cuộc trò chuyện mới")[:200], profile={})
        s.add(session)
        s.flush()
        s.refresh(session)
        return {**session_dict(session), "messages": [], "profile_summary": []}


@router.get("/{session_id}")
def get_session(session_id: str):
    with session_scope() as s:
        session = s.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(404, "Không tìm thấy phiên chat")
        return session_dict(session, with_messages=True)


@router.delete("/{session_id}")
async def delete_session(session_id: str):
    with session_scope() as s:
        session = s.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(404, "Không tìm thấy phiên chat")
        objects = [(d.bucket, d.object_key) for d in s.scalars(select(Document).where(Document.session_id == session_id))]
        s.delete(session)
    # Xóa luôn bệnh án gốc trên object storage — dữ liệu y tế không giữ lại khi người dùng xóa phiên
    storage = get_storage()
    await asyncio.gather(*(asyncio.to_thread(storage.delete, b, k) for b, k in objects), return_exceptions=True)
    return {"status": "deleted", "documents_removed": len(objects)}


@router.delete("/{session_id}/profile")
def reset_profile(session_id: str):
    with session_scope() as s:
        session = s.get(ChatSession, session_id)
        if session is None:
            raise HTTPException(404, "Không tìm thấy phiên chat")
        session.profile = {}
    return {"status": "reset"}


@router.patch("/{session_id}/messages/{message_id}/feedback")
def feedback(session_id: str, message_id: str, payload: Dict[str, Any] = Body(...)):
    with session_scope() as s:
        message = s.get(ChatMessage, message_id)
        if message is None or message.session_id != session_id:
            raise HTTPException(404, "Không tìm thấy tin nhắn")
        message.is_helpful = bool(payload.get("isHelpful"))
        message.is_useless = bool(payload.get("isUseless"))
    return {"status": "ok"}
