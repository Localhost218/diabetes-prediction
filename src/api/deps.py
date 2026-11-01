"""
Phụ thuộc dùng chung của tầng API.
"""

from functools import lru_cache
from typing import Any, Dict

from src.agents.orchestrator import OrchestratorAgent
from src.storage.models import ChatMessage, ChatSession


@lru_cache(maxsize=1)
def get_orchestrator() -> OrchestratorAgent:
    return OrchestratorAgent()


def message_dict(m: ChatMessage) -> Dict[str, Any]:
    return {
        "id": m.id,
        "sender": m.sender,
        "text": m.text,
        "data": m.data,
        "attachments": m.attachments or [],
        "is_error": bool(m.data and m.data.get("error")),
        "isHelpful": m.is_helpful,
        "isUseless": m.is_useless,
        "created_at": m.created_at.isoformat() if m.created_at else None,
    }


def session_dict(s: ChatSession, with_messages: bool = False) -> Dict[str, Any]:
    out = {"id": s.id, "title": s.title, "updated_at": s.updated_at.isoformat() if s.updated_at else None}
    if with_messages:
        from src.schemas.patient_profile import PatientProfile
        out["messages"] = [message_dict(m) for m in s.messages]
        out["profile_summary"] = PatientProfile.model_validate(s.profile or {}).summary_vi()
    return out
