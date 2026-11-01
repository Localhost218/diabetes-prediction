"""
Tác tử Điều phối Trung tâm (Orchestrator)
Mỗi lượt chat: phân loại ý định → lập kế hoạch → điều phối các tác tử qua MessageBus.
Kế hoạch là động: câu hỏi kiến thức chỉ đi nhánh RAG; thiếu dữ liệu thì hỏi lại; đủ dữ liệu thì
chạy song song nhánh Prediction và Knowledge rồi đưa vào hội đồng tranh biện.
"""

import asyncio
import re
import uuid
from typing import Any, Dict, List, Optional

from src.agents.consensus_critic_agent import ConsensusCriticAgent
from src.agents.conversation_agent import ConversationAgent
from src.agents.document_agent import DocumentAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.prediction_agent import PredictionAgent
from src.agents.response_agent import ResponseAgent
from src.agents.treatment_rl_agent import TreatmentRLAgent
from src.communication.message_bus import AgentMessage, BaseAgent, MessageBus
from src.llm.client import llm
from src.llm.prompts import INTENT_SYSTEM, IntentDecision, format_history
from src.schemas.patient_profile import PatientProfile

GREETING = re.compile(r"^\s*(xin chào|chào|hello|hi|alo|cảm ơn|cám ơn|thanks|ok|oke|bạn là ai)\b", re.I)
QUESTION = re.compile(r"\?|là gì|thế nào|như thế nào|bao nhiêu|có nên|tại sao|vì sao|cách nào|làm sao|nên ăn|có phải", re.I)
ASSESS = re.compile(r"nguy cơ|dự đoán|đánh giá|có bị (tiểu đường|đái tháo đường)|kiểm tra giúp|phân tích", re.I)
# Câu hỏi về chính hồ sơ / kết quả trước đó của người dùng
SELF_REFERENCE = re.compile(r"\b(tôi|của tôi|mình|em|con)\b.*(bao nhiêu|là gì|thế nào|không|chưa|ra sao|nào)"
                            r"|kết quả|vừa rồi|lúc nãy|ở trên|tại sao|vì sao|giải thích|nghĩa là|vậy thì|thế thì", re.I)


class OrchestratorAgent(BaseAgent):
    name = "Orchestrator"
    handles: set = set()

    def __init__(self):
        super().__init__(MessageBus())
        for agent_cls in (ConversationAgent, DocumentAgent, PredictionAgent, KnowledgeAgent,
                          ConsensusCriticAgent, TreatmentRLAgent, ResponseAgent):
            agent_cls(self.bus)

    async def run_turn(self, profile: PatientProfile, text: str, documents: List[Dict[str, Any]],
                       history: Optional[List[Dict[str, str]]] = None,
                       last_assessment: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        history: các lượt chat gần nhất [{sender, text}] (bộ nhớ hội thoại ngắn hạn)
        last_assessment: kết quả hội chẩn gần nhất của phiên (để trả lời câu hỏi tiếp theo mà không chạy lại hội đồng)
        """
        conv = uuid.uuid4().hex
        plan: List[str] = []
        text = (text or "").strip()
        history = history or []
        profile.conflicts = []  # mâu thuẫn chỉ báo ở lượt phát sinh
        last_bot = next((h["text"] for h in reversed(history) if h["sender"] == "bot"), "")

        # 1. Tiếp nhận song song: phân loại ý định + trích xuất từ text + đọc từng file
        intent_task = self._classify(text, history, profile) if text else _none()
        text_task = self.ask("ConversationAgent", "EXTRACT_FROM_TEXT",
                             {"text": text, "previous_bot_message": last_bot[-600:]}, conv) if text else _none()
        doc_tasks = [self.ask("DocumentAgent", "PARSE_DOCUMENT", d, conv) for d in documents]
        llm_intent, text_reply, *doc_replies = await asyncio.gather(intent_task, text_task, *doc_tasks)
        plan.append("intake: " + ", ".join(filter(None, ["text" if text else None, f"{len(documents)} file" if documents else None])))

        # Gộp file trước, tin nhắn sau: lời người dùng nói trực tiếp (ví dụ đính chính tuổi) được ưu tiên hơn file
        updated: List[str] = []
        doc_results = []
        for d, reply in zip(documents, doc_replies):
            result = {"document_id": d["document_id"], "filename": d["filename"], **reply.payload}
            result.pop("profile_delta", None)
            if reply.performative == "INFORM":
                updated += profile.merge(PatientProfile.model_validate(reply.payload["profile_delta"]))
            doc_results.append(result)
        if text_reply is not None and text_reply.performative == "INFORM":
            updated += profile.merge(PatientProfile.model_validate(text_reply.payload["profile_delta"]))
        updated = list(dict.fromkeys(updated))

        intent = self._decide_intent(text, llm_intent, updated, documents, profile, last_assessment)
        plan.append(f"intent: {intent}" + (" (LLM)" if llm_intent else " (luật)"))
        base = {"intent": intent, "updated_fields": updated, "documents": doc_results}

        # 2. Rẽ nhánh theo ý định
        if intent == "SMALL_TALK":
            reply = await self.ask("ResponseAgent", "COMPOSE_SMALL_TALK", {"text": text, "history": history}, conv)
            return self._finish(conv, plan + ["respond: small talk"], reply, base)

        if intent == "FOLLOW_UP":
            # Không có dữ liệu mới → trả lời từ bộ nhớ (hồ sơ + kết quả lần trước + lịch sử), không chạy lại hội đồng
            reply = await self.ask("ResponseAgent", "COMPOSE_CONTEXTUAL_ANSWER", {
                "question": text, "history": history, "profile_summary": profile.summary_vi(),
                "missing_required": profile.missing_required(), "missing_recommended": profile.missing_recommended(),
                "last_assessment": last_assessment,
            }, conv)
            return self._finish(conv, plan + ["memory: hồ sơ + lịch sử + kết quả trước", "respond: follow-up"], reply,
                                {**base, "profile_summary": profile.summary_vi()})

        if intent == "KNOWLEDGE_QUESTION":
            kb = await self.ask("KnowledgeAgent", "ANSWER_QUESTION", {"question": text}, conv)
            reply = await self.ask("ResponseAgent", "COMPOSE_KNOWLEDGE_ANSWER",
                                   {"question": text, "history": history, **kb.payload}, conv)
            return self._finish(conv, plan + ["RAG: KnowledgeAgent", "respond: knowledge"], reply,
                                {**base, "knowledge": kb.payload})

        if not profile.is_ready():
            reply = await self.ask("ResponseAgent", "COMPOSE_FOLLOW_UP", {
                "missing_required": profile.missing_required(),
                "understood": profile.summary_vi() if updated else [],
                "documents": doc_results,
            }, conv)
            return self._finish(conv, plan + [f"slot filling: thiếu {', '.join(profile.missing_required())}"], reply,
                                {**base, "missing_required": profile.missing_required()})

        # 3. Hai nhánh song song: DB → ML/DL và KB → RAG
        snapshot = profile.model_dump()
        pred, know = await asyncio.gather(
            self.ask("PredictionAgent", "PREDICT_RISK", {"profile": snapshot}, conv),
            self.ask("KnowledgeAgent", "RETRIEVE_EVIDENCE", {"profile": snapshot}, conv),
        )
        failed = [r for r in (pred, know) if r.performative == "FAILURE"]
        if failed:
            raise RuntimeError(f"{failed[0].sender}: {failed[0].payload.get('error')}")
        plan.append("parallel: PredictionAgent ‖ KnowledgeAgent")

        # 4. Hội đồng tranh biện (+/-), Critic tự chất vấn các tác tử khác qua bus
        consensus = await self.ask("ConsensusCriticAgent", "DELIBERATE",
                                   {"profile": snapshot, "prediction": pred.payload, "knowledge": know.payload}, conv)
        plan.append(f"debate: {consensus.payload['debate_rounds']} vòng")

        # 5. Gợi ý can thiệp (RL) → 6. Viết câu trả lời
        treatment = await self.ask("TreatmentRLAgent", "RECOMMEND_TREATMENT",
                                   {"profile": snapshot, "consensus": consensus.payload, "knowledge": know.payload}, conv)
        assessment = {"prediction": pred.payload, "knowledge": know.payload, "consensus": consensus.payload,
                      "treatment": treatment.payload}
        reply = await self.ask("ResponseAgent", "COMPOSE_ASSESSMENT", {
            **assessment, "missing_recommended": profile.missing_recommended(), "updated_fields": updated,
            "question": text, "history": history,
        }, conv)
        return self._finish(conv, plan + ["RL: TreatmentRLAgent", "respond: assessment"], reply,
                            {**base, **assessment, "profile_summary": profile.summary_vi()})

    # ------------------------------------------------------------------
    async def _classify(self, text: str, history: List[Dict[str, str]], profile: PatientProfile) -> Optional[str]:
        context = format_history(history)
        prompt = (f"Lịch sử hội thoại gần đây:\n{context or '(chưa có)'}\n\n"
                  f"Hồ sơ đã ghi nhận: {'; '.join(profile.summary_vi()) or '(trống)'}\n\n"
                  f"Tin nhắn mới cần phân loại: {text}")
        decision = await llm.parse(INTENT_SYSTEM, [prompt], IntentDecision)
        return decision.intent if decision else None

    @staticmethod
    def _decide_intent(text: str, llm_intent: Optional[str], updated: List[str], documents: List[Dict[str, Any]],
                       profile: PatientProfile, last_assessment: Optional[Dict[str, Any]]) -> str:
        # Có dữ liệu mới của bệnh nhân (từ text hoặc file) thì luôn chạy lại hội đồng
        if documents or updated:
            return "RISK_ASSESSMENT"
        if llm_intent:
            # Không có dữ liệu mới mà hội đồng đã kết luận rồi → chạy lại cũng ra cùng kết quả, trả lời từ bộ nhớ
            if llm_intent == "RISK_ASSESSMENT" and last_assessment and profile.is_ready():
                return "FOLLOW_UP"
            return llm_intent
        if GREETING.search(text) and len(text) < 40:
            return "SMALL_TALK"
        if profile.fields and (SELF_REFERENCE.search(text) or last_assessment and ASSESS.search(text)):
            return "FOLLOW_UP"
        if ASSESS.search(text):
            return "RISK_ASSESSMENT"
        if QUESTION.search(text):
            return "KNOWLEDGE_QUESTION"
        return "FOLLOW_UP" if profile.fields else "KNOWLEDGE_QUESTION"

    def _finish(self, conv: str, plan: List[str], reply: AgentMessage, data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "conversation_id": conv,
            "text": reply.payload["text"],
            "generated_by": reply.payload.get("generated_by"),
            "plan": plan,
            "data": data,
            "transcript": self.bus.pop_transcript(conv),
        }


async def _none():
    return None
