"""
Tác tử Hội thoại (Conversation / Text Intake Agent)
Bóc tách chỉ số lâm sàng từ tin nhắn tự do của người dùng thành PatientProfile.
LLM (structured output) là nhánh chính; regex là nhánh dự phòng.
"""

import re

from src.communication.message_bus import AgentMessage, BaseAgent
from src.document_processing.lab_extractor import extract_profile
from src.llm.client import llm
from src.llm.prompts import EXTRACTION_SYSTEM, ExtractedClinicalData


class ConversationAgent(BaseAgent):
    name = "ConversationAgent"
    handles = {"EXTRACT_FROM_TEXT"}

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        text = msg.payload["text"]
        previous = msg.payload.get("previous_bot_message")
        # Kèm câu chatbot vừa hỏi để hiểu câu trả lời cụt (ví dụ bot hỏi tuổi, người dùng chỉ gõ "62")
        prompt = (f"Câu chatbot vừa hỏi người dùng:\n{previous}\n\n" if previous else "") + f"Tin nhắn của người dùng:\n{text}"
        extracted = await llm.parse(EXTRACTION_SYSTEM, [prompt], ExtractedClinicalData)
        if extracted is not None:
            profile, method = extracted.to_profile("text"), "llm"
        else:
            profile, method = extract_profile(text, "text"), "rules"
            # Câu trả lời chỉ là một con số cho câu hỏi tuổi vừa rồi
            if previous and "bao nhiêu tuổi" in previous.lower() and re.fullmatch(r"\s*(\d{1,3})\s*", text):
                profile.set("age_years", int(text), "text", 0.9, f"Trả lời câu hỏi tuổi: {text}")

        return self.reply(msg, "PATIENT_DATA_EXTRACTED", {
            "profile_delta": profile.model_dump(),
            "fields": sorted(profile.fields),
            "method": method,
        })
