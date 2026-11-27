from src.agents.consensus_critic_agent import diet_findings
from src.document_processing.lab_extractor import extract_profile
from src.knowledge.scope import rule_questions, select_questions
from src.schemas.patient_profile import LIFESTYLE_FIELDS, PatientProfile


def test_lifestyle_is_extracted_from_free_text():
    p = extract_profile("tôi ít ăn rau, ăn trái cây mỗi ngày, mỗi đêm chỉ ngủ khoảng 5 tiếng", "text")
    assert p.get("vegetables_daily") is False and p.get("fruits_daily") is True
    assert p.get("sleep_hours") == 5


def test_findings_split_risky_and_protective_habits():
    p = PatientProfile()
    p.apply_form({"vegetables_daily": "no", "fruits_daily": "yes", "sleep_hours": "5"})
    bad, good = diet_findings(p)
    assert bad == ["Ít ăn rau xanh", "Ngủ ít (5 giờ/đêm)"] and good == ["Ăn trái cây mỗi ngày"]
    p.apply_form({"sleep_hours": "7.5"})
    assert "Ngủ đủ (7.5 giờ/đêm)" in diet_findings(p)[1]


def test_lifestyle_questions_are_asked_gradually_two_per_turn():
    p = PatientProfile()
    p.apply_form({"age_years": "40", "bmi": "24", "hba1c_percent": "5.5"})
    asked = []
    for _ in range(2):
        qs = select_questions(p, rule_questions(p))
        assert len(qs) <= 2
        asked += [q["key"] for q in qs]
        p.asked_questions += [q["key"] for q in qs]
    assert asked == LIFESTYLE_FIELDS
