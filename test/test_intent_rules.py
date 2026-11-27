from src.agents.orchestrator import OrchestratorAgent
from src.schemas.patient_profile import PatientProfile

decide = OrchestratorAgent._decide_intent


def _profile():
    p = PatientProfile()
    p.apply_form({"age_years": "21", "height_cm": "165", "weight_kg": "75"})
    return p


def test_symptom_question_with_saved_profile_runs_assessment_even_if_llm_says_follow_up():
    for text in ["Tôi có dấu hiệu của bệnh tiểu đường không", "tôi có bị tiểu đường không?", "Mình có triệu chứng đái tháo đường không"]:
        assert decide(text, None, [], [], _profile(), None) == "RISK_ASSESSMENT", text
        assert decide(text, "FOLLOW_UP", [], [], _profile(), None) == "RISK_ASSESSMENT", text


def test_after_an_assessment_the_same_question_is_answered_from_memory():
    assert decide("Tôi có dấu hiệu của bệnh tiểu đường không", None, [], [], _profile(), {"consensus": {}}) == "FOLLOW_UP"


def test_general_questions_and_profile_lookups_unchanged():
    assert decide("Dấu hiệu sớm của bệnh tiểu đường là gì?", None, [], [], PatientProfile(), None) == "KNOWLEDGE_QUESTION"
    assert decide("tôi cao bao nhiêu?", None, [], [], _profile(), None) == "FOLLOW_UP"
