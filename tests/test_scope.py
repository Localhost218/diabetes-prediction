"""Câu hỏi làm rõ và phạm vi áp dụng (nhánh luật, không cần DB / LLM)."""

import asyncio

from src.agents.clarification_agent import ClarificationAgent
from src.communication.message_bus import AgentMessage, MessageBus
from src.document_processing.lab_extractor import extract_profile
from src.knowledge.scope import MAX_QUESTIONS_PER_TURN, apply_yes_no, rule_questions, scope_notes, select_questions
from src.schemas.patient_profile import PatientProfile


def _keys(questions):
    return [q["key"] for q in questions]


def test_pregnancy_question_blocks_for_woman_with_abnormal_glucose():
    p = extract_profile("tôi là nữ, 30 tuổi, BMI 24, đường huyết lúc đói 110 mg/dL", "text")
    qs = select_questions(p, rule_questions(p))
    assert qs[0]["key"] == "pregnant" and qs[0]["blocking"]


def test_no_pregnancy_question_for_men_or_normal_glucose():
    men = extract_profile("tôi là nam, 30 tuổi, BMI 24, đường huyết lúc đói 110 mg/dL", "text")
    normal = extract_profile("tôi là nữ, 30 tuổi, BMI 24, đường huyết lúc đói 85 mg/dL", "text")
    assert "pregnant" not in _keys(rule_questions(men)) + _keys(rule_questions(normal))


def test_known_pregnancy_becomes_scope_alert_not_question():
    p = extract_profile("tôi là nữ, 30 tuổi, đang mang thai 26 tuần, BMI 24, đường huyết lúc đói 110 mg/dL", "text")
    assert p.get("pregnant") is True
    assert "pregnant" not in _keys(rule_questions(p))
    alerts, minus = scope_notes(p)
    assert alerts and alerts[0]["level"] == "scope" and minus


def test_unclear_glucose_asks_fasting_and_unit():
    p = extract_profile("tôi 50 tuổi BMI 25, đường huyết 6,4", "text")
    keys = _keys(rule_questions(p))
    assert "fasting_confirmed" in keys
    assert any(k.startswith("unit:glucose_fasting_mgdl") for k in keys)


def test_fasting_answer_resolves_flag_or_moves_value():
    yes = extract_profile("tôi 50 tuổi BMI 25, đường huyết 6,4", "text")
    yes.set("fasting_confirmed", True, "text")
    yes.derive()
    assert "fasting_unclear" not in yes.fields["glucose_fasting_mgdl"].flags

    no = extract_profile("tôi 50 tuổi BMI 25, đường huyết 6,4", "text")
    no.set("fasting_confirmed", False, "text")
    no.derive()
    assert not no.has("glucose_fasting_mgdl") and no.has("glucose_random_mgdl")


def test_questions_are_capped_and_never_repeated():
    p = extract_profile("tôi là nữ, 30 tuổi BMI 25, đường huyết 6,4", "text")
    first = select_questions(p, rule_questions(p))
    assert len(first) <= MAX_QUESTIONS_PER_TURN
    p.asked_questions += _keys(first)
    assert not set(_keys(first)) & set(_keys(select_questions(p, rule_questions(p))))


def test_short_yes_no_answer_fills_single_pending_question():
    p = PatientProfile(pending_questions=["diagnosed_diabetes"])
    assert apply_yes_no(p, "chưa") == ["diagnosed_diabetes"] and p.get("diagnosed_diabetes") is False
    # Hai câu hỏi đang chờ thì "không" là mơ hồ → không ghi
    p2 = PatientProfile(pending_questions=["pregnant", "diagnosed_diabetes"])
    assert apply_yes_no(p2, "không") == []


def test_context_regex_distinguishes_self_from_family():
    family = extract_profile("mẹ tôi bị tiểu đường, tôi bị thiếu máu cơ tim", "text")
    assert family.get("family_history") is True
    assert not family.has("diagnosed_diabetes") and not family.has("hba1c_interference")

    me = extract_profile("tôi bị tiểu đường 5 năm, đang tiêm insulin", "text")
    assert me.get("diagnosed_diabetes") is True and me.get("glucose_lowering_meds") is True
    assert extract_profile("tôi không mang thai", "text").get("pregnant") is False


def test_hba1c_glucose_discordance_asks_about_interference():
    p = extract_profile("tôi 55 tuổi BMI 23, đường huyết lúc đói 90 mg/dL, HbA1c 6.8%", "text")
    assert "hba1c_interference" in _keys(rule_questions(p))


def _plan(profile, issues):
    agent = ClarificationAgent(MessageBus())
    msg = AgentMessage("Orchestrator", agent.name, "REQUEST", "PLAN_QUESTIONS",
                       {"profile": profile.model_dump(), "scope_issues": issues})
    return asyncio.run(agent.handle(msg)).payload


def test_llm_issues_cannot_block_except_pregnancy_and_are_logged_when_uncovered():
    p = extract_profile("tôi 45 tuổi BMI 24", "text")
    out = _plan(p, [{"reason": "OTHER", "evidence": "đang nằm viện", "question": "Bạn nằm viện vì bệnh gì?",
                     "blocking": True}])
    assert out["questions"][0]["source"] == "llm" and not out["questions"][0]["blocking"]
    assert out["uncovered"][0]["evidence"] == "đang nằm viện"


def test_llm_issue_skipped_when_fact_already_known():
    p = extract_profile("tôi là nữ, 30 tuổi, không mang thai, BMI 24", "text")
    out = _plan(p, [{"reason": "PREGNANCY", "evidence": "mang thai", "question": "Bạn có mang thai không?",
                     "blocking": True}])
    assert out["questions"] == []
