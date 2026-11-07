"""Đối chiếu tiêu chuẩn ADA và ghép hồ sơ nhiều nguồn."""

from src.knowledge.clinical_criteria import evaluate_glycemia, graph_entities
from src.schemas.patient_profile import PatientProfile


def _profile(**fields):
    p = PatientProfile()
    for k, v in fields.items():
        p.set(k, v, "form")
    return p


def test_single_abnormal_test_needs_confirmation():
    g = evaluate_glycemia(_profile(glucose_fasting_mgdl=130))
    assert g["status"] == "diabetes_range" and g["needs_confirmation"]


def test_two_abnormal_tests_meet_criteria():
    g = evaluate_glycemia(_profile(glucose_fasting_mgdl=130, hba1c_percent=6.8))
    assert g["status"] == "diabetes_range" and not g["needs_confirmation"]


def test_prediabetes_and_normal():
    assert evaluate_glycemia(_profile(hba1c_percent=6.0))["status"] == "prediabetes_range"
    assert evaluate_glycemia(_profile(glucose_fasting_mgdl=92))["status"] == "normal"


def test_graph_entities_from_profile():
    p = _profile(age_years=58, bmi=29, waist_cm=95, sex="nữ", high_bp=True, family_history=True)
    entities = graph_entities(p, "prediabetes_range")
    assert {"Tiền đái tháo đường", "Thừa cân/béo phì", "Béo bụng", "Tăng huyết áp",
            "Tiền sử gia đình ĐTĐ", "Tuổi ≥ 35"} <= set(entities)


def test_merge_detects_conflict_between_sources():
    p = _profile(age_years=47)
    other = PatientProfile()
    other.set("age_years", 49, "text")
    p.merge(other)
    assert p.get("age_years") == 49
    assert p.conflicts and p.conflicts[0].field == "age_years"
