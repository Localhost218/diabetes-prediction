import asyncio

from src.agents.conversation_agent import ConversationAgent
from src.communication.message_bus import AgentMessage, MessageBus

ASK_FPG = "Để hội đồng đánh giá, mình cần thêm:\n1. Đường huyết lúc đói gần nhất của bạn (ghi rõ mmol/L hay mg/dL)?"


def _extract(text, previous, asked=()):
    agent = ConversationAgent(MessageBus())
    msg = AgentMessage(sender="t", receiver="ConversationAgent", performative="REQUEST", msg_type="EXTRACT_FROM_TEXT",
                       payload={"text": text, "previous_bot_message": previous, "asked_fields": list(asked)})
    return asyncio.run(agent.handle(msg)).payload["profile_delta"]["fields"]


def test_hedged_bare_glucose_answer_is_recorded():
    f = _extract("chắc là khoảng 165mg/dL", ASK_FPG)
    assert f["glucose_fasting_mgdl"]["value"] == 165


def test_bare_mmol_answer_is_converted():
    assert round(_extract("7,2 mmol/L", ASK_FPG)["glucose_fasting_mgdl"]["value"]) == 130


def test_bare_age_answer_still_works_and_ambiguous_answers_are_ignored():
    assert _extract("62", "Bạn bao nhiêu tuổi?")["age_years"]["value"] == 62
    assert "glucose_fasting_mgdl" not in _extract("khoảng 120 hoặc 130", ASK_FPG)
    assert _extract("165", None) == {}


def test_late_glucose_answer_after_bot_moved_on_to_other_questions():
    later = "Để kết quả chính xác hơn, mình cần hỏi thêm:\n1. Bạn có hay ăn đồ chiên rán không?"
    assert _extract("chắc là khoảng 165mg/dL", later, ["glucose_fasting_mgdl"])["glucose_fasting_mgdl"]["value"] == 165
    assert "glucose_fasting_mgdl" not in _extract("chắc là khoảng 165mg/dL", later, [])  # chưa từng hỏi → không đoán


def test_numbers_inside_a_longer_sentence_or_with_another_unit_are_not_taken_as_the_asked_value():
    assert "glucose_fasting_mgdl" not in _extract("tôi ít ăn rau, mỗi đêm chỉ ngủ khoảng 5 tiếng", ASK_FPG)
    assert "glucose_fasting_mgdl" not in _extract("5 tiếng", ASK_FPG)
    assert _extract("khoảng 6 tiếng", "Bình thường mỗi đêm bạn ngủ khoảng mấy tiếng?")["sleep_hours"]["value"] == 6
