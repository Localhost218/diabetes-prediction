"""Khi model LLM dự phòng chỉ trích được một phần tin nhắn, regex bù các trường còn thiếu
→ cùng một tin nhắn luôn cho cùng tập chỉ số tối thiểu (lỗi gặp thực tế khi Gemini quá tải)."""

import asyncio

from src.agents.conversation_agent import ConversationAgent
from src.communication.message_bus import AgentMessage, MessageBus
from src.llm.prompts import ExtractedClinicalData

TEXT = "Tôi 52 tuổi, nam, cao 1m65 nặng 72kg, đường huyết lúc đói 7.2 mmol/L, bố bị tiểu đường. Nguy cơ của tôi thế nào?"


def _extract(monkeypatch, llm_result):
    from src.agents import conversation_agent as mod

    async def fake_parse(*_a, **_k):
        return llm_result
    monkeypatch.setattr(mod.llm, "parse", fake_parse)
    agent = ConversationAgent(MessageBus())
    msg = AgentMessage(sender="Orchestrator", receiver=agent.name, performative="REQUEST",
                       msg_type="EXTRACT_FROM_TEXT", payload={"text": TEXT})
    return asyncio.run(agent.handle(msg)).payload


def test_partial_llm_result_is_completed_by_rules(monkeypatch):
    out = _extract(monkeypatch, ExtractedClinicalData(age_years=52, sex="male"))
    assert out["method"] == "llm+rules"
    for f in ("age_years", "sex", "height_cm", "weight_kg", "bmi", "glucose_fasting_mgdl", "family_history"):
        assert f in out["fields"], f
    glucose = out["profile_delta"]["fields"]["glucose_fasting_mgdl"]["value"]
    assert 129 <= glucose <= 130  # 7,2 mmol/L


def test_llm_value_wins_when_both_present(monkeypatch):
    out = _extract(monkeypatch, ExtractedClinicalData(age_years=53, sex="male"))
    assert out["profile_delta"]["fields"]["age_years"]["value"] == 53


def test_same_fields_with_or_without_llm(monkeypatch):
    partial = set(_extract(monkeypatch, ExtractedClinicalData(age_years=52, sex="male"))["fields"])
    rules_only = set(_extract(monkeypatch, None)["fields"])
    assert rules_only <= partial
