"""
Tác tử Hội thoại (Conversation / Text Intake Agent)
Bóc tách chỉ số lâm sàng từ tin nhắn tự do của người dùng thành PatientProfile.
LLM (Ollama / Qwen structured output) là nhánh chính; regex luôn chạy song song và bù các trường LLM bỏ sót,
nên cùng một tin nhắn luôn cho ít nhất cùng một tập chỉ số chuẩn xác.

VẤN ĐÁP — VỊ TRÍ TRONG LUỒNG:
  Nhận: EXTRACT_FROM_TEXT từ Orchestrator (bước tiếp nhận, chạy song song với Clarification/Document)
  Trả:  PATIENT_DATA_EXTRACTED {profile_delta, fields, method} → Orchestrator gộp vào hồ sơ
  Chống LLM bịa số: ExtractedClinicalData.to_profile(source_text=…) chỉ nhận con số có trong tin nhắn.
  _answer_to_question: hiểu câu trả lời cụt ("chắc khoảng 165 mg/dL") cho đúng chỉ số chatbot vừa hỏi.
  Tác tử loại: phản xạ có mô hình (model-based reflex) — dùng câu hỏi trước đó làm ngữ cảnh.
"""

import re

from src.communication.message_bus import AgentMessage, BaseAgent
from src.document_processing.lab_extractor import extract_profile
from src.document_processing.unit_normalizer import normalize, unit_flags
from src.schemas.patient_profile import FIELD_SPECS
from src.llm.client import llm
from src.llm.prompts import EXTRACTION_SYSTEM, ExtractedClinicalData


# Câu trả lời cụt cho câu chatbot vừa hỏi: "chắc khoảng 165mg/dL", "62", "6,8 %"
BARE_NUMBER = re.compile(r"(\d+(?:[.,]\d+)?)\s*(mmol/?l|mg/?dl|%|cm|kg|tuổi|tiếng|giờ)?", re.I)
ASKABLE = {"bao nhiêu tuổi": "age_years", "đường huyết lúc đói": "glucose_fasting_mgdl", "hba1c": "hba1c_percent",
           "vòng eo": "waist_cm", "cholesterol": "total_cholesterol_mgdl", "egfr": "egfr", "ngủ khoảng mấy": "sleep_hours"}


ANSWER_UNITS = {"age_years": {"", "tuổi"}, "glucose_fasting_mgdl": {"", "mmol/l", "mmoll", "mg/dl", "mgdl"},
                "hba1c_percent": {"", "%"}, "waist_cm": {"", "cm"}, "total_cholesterol_mgdl": {"", "mmol/l", "mmoll", "mg/dl", "mgdl"},
                "egfr": {""}, "sleep_hours": {"", "tiếng", "giờ"}}


def _answer_to_question(previous, text: str, profile, asked_before=()) -> None:
    """Gán con số duy nhất trong tin nhắn cho trường mà chatbot vừa hỏi, nếu luật thường chưa bắt được trường đó.
    asked_before: các trường đã hỏi ở lượt trước — "165 mg/dL" gõ muộn vẫn hiểu là đường huyết đã được hỏi."""
    numbers = BARE_NUMBER.findall(text)
    questions = " ".join(line for line in (previous or "").lower().splitlines() if "?" in line)  # bỏ phần "đã ghi nhận"
    asked = [f for key, f in ASKABLE.items() if key in questions and not profile.has(f)]
    if not asked and len(numbers) == 1 and re.fullmatch(r"mmol/?l|mg/?dl", numbers[0][1] or "", re.I) \
            and "glucose_fasting_mgdl" in asked_before and not profile.has("glucose_fasting_mgdl"):
        asked = ["glucose_fasting_mgdl"]
    if len(asked) != 1 or len(numbers) != 1:
        return  # mơ hồ (hỏi nhiều chỉ số hoặc trả lời nhiều số) → để LLM / người dùng nói rõ
    field, (raw, unit) = asked[0], numbers[0]
    # Chỉ nhận câu trả lời ngắn ("chắc khoảng 165 mg/dL") và đơn vị hợp với chỉ số được hỏi —
    # "ít ăn rau, ngủ khoảng 5 tiếng" không phải câu trả lời cho câu hỏi đường huyết
    if len(text.split()) > 6 or (unit or "").lower() not in ANSWER_UNITS.get(field, {""}):
        return
    value, conf = normalize(field, float(raw.replace(",", ".")), unit or None)
    profile.set(field, value, "text", 0.85 * conf, f"Trả lời câu hỏi {FIELD_SPECS[field].label_vi}: {text.strip()}",
                unit_flags(field, unit or None))


class ConversationAgent(BaseAgent):
    name = "ConversationAgent"
    handles = {"EXTRACT_FROM_TEXT"}

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        # ĐIỂM VÀO: MessageBus gọi khi Orchestrator gửi EXTRACT_FROM_TEXT
        text = msg.payload["text"]
        previous = msg.payload.get("previous_bot_message")
        # Kèm câu chatbot vừa hỏi để hiểu câu trả lời cụt (ví dụ bot hỏi tuổi, người dùng chỉ gõ "62")
        prompt = (f"Câu chatbot vừa hỏi người dùng:\n{previous}\n\n" if previous else "") + f"Tin nhắn của người dùng:\n{text}"
        extracted = await llm.parse(EXTRACTION_SYSTEM, [prompt], ExtractedClinicalData)
        rules = extract_profile(text, "text")
        _answer_to_question(previous, text, rules, msg.payload.get("asked_fields", []))
        if extracted is None:
            profile, method = rules, "rules"
        else:
            profile, method = extracted.to_profile("text", source_text=text), "llm"
            filled = [n for n in rules.fields if n not in profile.fields]
            for name in filled:  # LLM ưu tiên; regex chỉ bù trường LLM bỏ sót
                profile.fields[name] = rules.fields[name]
            if filled:
                method = "llm+rules"
                profile.derive()

        return self.reply(msg, "PATIENT_DATA_EXTRACTED", {
            "profile_delta": profile.model_dump(),
            "fields": sorted(profile.fields),
            "method": method,
        })
