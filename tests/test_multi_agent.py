"""
Kiểm thử tích hợp toàn bộ hội đồng đa tác tử (cần DB + Qdrant + Neo4j và KB đã nạp, chạy không LLM).
    docker compose up -d neo4j qdrant && python scripts/ingest_kb.py && python scripts/train_model.py
"""

import asyncio

import pytest
from sqlalchemy import text

from src.schemas.patient_profile import PatientProfile


def _db_ready() -> bool:
    try:
        from src.storage.db import engine
        with engine.connect() as c:
            return c.execute(text("SELECT count(*) FROM kb_chunks")).scalar() > 0
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(not _db_ready(), reason="Cần DB và Knowledge Base đã nạp")


@pytest.fixture(scope="module")
def orchestrator():
    from src.agents.orchestrator import OrchestratorAgent
    return OrchestratorAgent()


def _turn(orc, profile, text):
    return asyncio.run(orc.run_turn(profile, text, []))


def test_slot_filling_then_assessment(orchestrator):
    profile = PatientProfile()
    r1 = _turn(orchestrator, profile, "Tôi 52 tuổi, nam, cao 1m65 nặng 72kg")
    assert r1["data"]["intent"] == "RISK_ASSESSMENT"
    assert r1["data"]["missing_required"] == ["glucose_fasting_mgdl"]

    r2 = _turn(orchestrator, profile, "đường huyết lúc đói 7.2 mmol/L, HbA1c 6.9%")
    c = r2["data"]["consensus"]
    assert c["glycemic_status"] == "diabetes_range" and not c["confirmation_required"]
    assert c["risk_level"] == "very_high"
    performatives = {m["performative"] for m in r2["transcript"]}
    assert {"REQUEST", "INFORM", "CHALLENGE", "RESPONSE"} <= performatives


def test_two_round_debate_when_ml_disagrees_with_labs(orchestrator):
    r = _turn(orchestrator, PatientProfile(), "tôi 58 tuổi nữ BMI 29 vòng eo 95, tăng huyết áp, mỡ máu cao, "
                                              "mẹ bị tiểu đường, ít vận động, đường huyết đói 92 mg/dL")
    c = r["data"]["consensus"]
    assert c["glycemic_status"] == "normal" and c["ml_probability"] >= 0.30
    assert c["debate_rounds"] == 2
    assert [d["issue"] for d in c["debate"]] == ["ml_high_labs_normal", "screening_plan"]
    # Xét nghiệm bình thường → RL bị hội đồng giới hạn ở can thiệp lối sống
    assert r["data"]["treatment"]["action_id"] == 0


def test_knowledge_question_skips_prediction(orchestrator):
    r = _turn(orchestrator, PatientProfile(), "HbA1c bao nhiêu thì là tiền tiểu đường?")
    assert r["data"]["intent"] == "KNOWLEDGE_QUESTION"
    assert "PREDICT_RISK" not in {m["msg_type"] for m in r["transcript"]}
    assert any(c["document_id"] == "ada-2024-tien-dtd" for c in r["data"]["knowledge"]["citations"])


def test_guardrail_overrides_for_very_high_glucose(orchestrator):
    r = _turn(orchestrator, PatientProfile(), "tôi 40 tuổi BMI 22, glucose lúc đói 320 mg/dL")
    c = r["data"]["consensus"]
    assert any(g["level"] == "urgent" for g in c["safety_guardrails"])
    assert r["data"]["treatment"]["action_id"] == 3


def test_follow_up_uses_memory_instead_of_rerunning_council(orchestrator):
    """Tình huống thực tế: đã đánh giá xong, người dùng hỏi lại chỉ số chưa từng khai."""
    profile = PatientProfile()
    first = _turn(orchestrator, profile, "Tôi 45 tuổi, BMI 27, HbA1c 6.1%, tăng huyết áp")
    last_assessment = {**first["data"]["consensus"], "treatment": first["data"]["treatment"]}
    history = [{"sender": "user", "text": "Tôi 45 tuổi, BMI 27, HbA1c 6.1%, tăng huyết áp"},
               {"sender": "bot", "text": first["text"]}]

    r = asyncio.run(orchestrator.run_turn(profile, "tôi cao bao nhiêu và nặng bao nhiêu", [], history, last_assessment))
    assert r["data"]["intent"] == "FOLLOW_UP"
    assert "PREDICT_RISK" not in {m["msg_type"] for m in r["transcript"]}  # không chạy lại hội đồng
    assert "chưa cung cấp" in r["text"]

    r = asyncio.run(orchestrator.run_turn(profile, "tôi bao nhiêu tuổi?", [], history, last_assessment))
    assert "45" in r["text"]


def test_short_answer_to_bot_question(orchestrator):
    profile = PatientProfile()
    r1 = _turn(orchestrator, profile, "BMI 24, đường huyết lúc đói 5.5 mmol/L")
    assert r1["data"]["missing_required"] == ["age_years"]
    history = [{"sender": "user", "text": "BMI 24, đường huyết lúc đói 5.5 mmol/L"}, {"sender": "bot", "text": r1["text"]}]
    r2 = asyncio.run(orchestrator.run_turn(profile, "62", [], history, None))
    assert profile.get("age_years") == 62 and "consensus" in r2["data"]


def test_ml_dl_models_and_disagreement_debate(orchestrator):
    pred = orchestrator.bus.agents["PredictionAgent"]
    young, old = PatientProfile(), PatientProfile()
    for p, age, bmi, flags in ((young, 28, 21, {}), (old, 64, 32, {"high_bp": True, "family_history": True})):
        p.set("age_years", age, "form")
        p.set("bmi", bmi, "form")
        for k, v in flags.items():
            p.set(k, v, "form")
    r_young, r_old = pred._ensemble(young), pred._ensemble(old)
    for r in (r_young, r_old):
        assert {r["models"]["ml"]["kind"], r["models"]["dl"]["kind"]} == {"ML", "DL"}
    assert r_old["risk_probability"] > 3 * r_young["risk_probability"]

    # Hồ sơ bất thường (BMI 42 nhưng vòng eo 80 cm): hai mô hình ngoại suy khác nhau → hội đồng chất vấn
    r = _turn(orchestrator, PatientProfile(), "Tôi 75 tuổi, BMI 42, vòng eo 80 cm, không tăng huyết áp, mẹ bị tiểu đường, "
                                              "đường huyết lúc đói 95 mg/dL")
    issues = [d["issue"] for d in r["data"]["consensus"]["debate"]]
    assert "ml_dl_disagree" in issues
    assert any("ML và DL bất đồng" in m for m in r["data"]["consensus"]["negative_contradictions_minus"])


def test_clarification_blocks_then_resumes_after_short_answer(orchestrator):
    profile = PatientProfile()
    r1 = _turn(orchestrator, profile, "tôi là nữ, 32 tuổi, BMI 26, đường huyết lúc đói 118 mg/dL")
    assert "consensus" not in r1["data"]
    assert [q["key"] for q in r1["data"]["clarification"]["questions"]] == ["pregnant"]
    assert "PREDICT_RISK" not in {m["msg_type"] for m in r1["transcript"]}

    r2 = _turn(orchestrator, profile, "không")
    assert profile.get("pregnant") is False and "consensus" in r2["data"]
    assert "pregnant" not in [q["key"] for q in r2["data"]["clarification"]["questions"]]
