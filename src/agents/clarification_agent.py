"""
Tác tử Làm rõ (Clarification Agent)
Quyết định có cần hỏi lại người dùng trước/kèm kết luận hay không, theo hai tầng:
  - Tầng luật (src/knowledge/scope.py): mâu thuẫn nguồn, đơn vị phải đoán, glucose chưa rõ lúc đói,
    thai phụ, đã chẩn đoán, HbA1c lệch glucose... — xác định, kiểm thử được.
  - Tầng LLM (SCOPE_CHECK): tìm tình huống mơ hồ / ngoài phạm vi mà luật chưa lường trước. Mỗi vấn đề phải kèm
    đoạn trích nguyên văn từ tin nhắn (không có thì loại), và LLM chỉ được chặn luồng với thai phụ / trẻ em.
Các vấn đề nhóm OTHER được trả về trong `uncovered` (lưu ở audit trail) để bổ sung luật sau này.
"""

import re
from typing import Any, Dict, List

from src.communication.message_bus import AgentMessage, BaseAgent
from src.knowledge.scope import rule_questions, select_questions
from src.llm.client import llm
from src.llm.prompts import SCOPE_SYSTEM, ScopeCheck, format_history
from src.schemas.patient_profile import PatientProfile

REASON_FIELD = {
    "PREGNANCY": "pregnant", "ALREADY_DIAGNOSED": "diagnosed_diabetes", "ON_DIABETES_MEDS": "glucose_lowering_meds",
    "HBA1C_UNRELIABLE": "hba1c_interference", "ACUTE_ILLNESS_OR_DRUG": "acute_illness",
}
LLM_MAY_BLOCK = {"PREGNANCY", "PEDIATRIC"}


class ClarificationAgent(BaseAgent):
    name = "ClarificationAgent"
    handles = {"SCOPE_CHECK", "PLAN_QUESTIONS"}

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        if msg.msg_type == "SCOPE_CHECK":
            return self.reply(msg, "SCOPE_ISSUES", await _scope_check(msg.payload["text"], msg.payload.get("history")))

        profile = PatientProfile.model_validate(msg.payload["profile"])
        candidates = rule_questions(profile)
        uncovered = []
        for issue in msg.payload.get("scope_issues", []):
            field = REASON_FIELD.get(issue["reason"])
            if field and profile.has(field):
                continue  # người dùng đã nói rõ trong chính tin nhắn → không hỏi lại
            if issue["reason"] == "PEDIATRIC" and profile.has("age_years"):
                continue
            if issue["reason"] == "OTHER":
                uncovered.append(issue)
            candidates.append({
                "key": field or f"{issue['reason'].lower()}:{_norm(issue['evidence'])[:40]}",
                "field": field, "question": issue["question"], "reason": f"\"{issue['evidence']}\"",
                "priority": 2, "blocking": issue["blocking"] and issue["reason"] in LLM_MAY_BLOCK, "source": "llm",
            })
        return self.reply(msg, "CLARIFICATION_PLAN", {
            "questions": select_questions(profile, candidates),
            "candidates": len(candidates),
            "uncovered": uncovered,
        })


async def _scope_check(text: str, history) -> Dict[str, Any]:
    if not llm.enabled or not text:
        return {"issues": [], "method": "disabled"}
    context = format_history(history or [])
    prompt = (f"Lịch sử hội thoại gần đây:\n{context}\n\n" if context else "") + f"Tin nhắn mới của người dùng:\n{text}"
    result = await llm.parse(SCOPE_SYSTEM, [prompt], ScopeCheck)
    if result is None:
        return {"issues": [], "method": "failed"}
    # Chống bịa: bằng chứng phải xuất hiện nguyên văn trong tin nhắn
    grounded = [i.model_dump() for i in result.issues if _norm(i.evidence) and _norm(i.evidence) in _norm(text)]
    return {"issues": grounded[:2], "rejected": len(result.issues) - len(grounded), "method": "llm"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip(" \"'“”….").lower())
