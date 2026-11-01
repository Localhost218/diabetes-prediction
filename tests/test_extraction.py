"""Trích xuất chỉ số từ tin nhắn tiếng Việt và quy đổi đơn vị (nhánh luật, không LLM)."""

import pytest

from src.document_processing.lab_extractor import extract_profile
from src.document_processing.unit_normalizer import glucose_to_mgdl, hba1c_to_percent


def test_free_text_message():
    p = extract_profile("Tôi 52 tuổi, nam, cao 1m65 nặng 72kg, đường huyết lúc đói 7.2 mmol/L, "
                        "bố bị tiểu đường, không hút thuốc", "text")
    assert p.get("age_years") == 52
    assert p.get("sex") == 1
    assert p.get("bmi") == pytest.approx(26.4, abs=0.1)
    assert p.get("glucose_fasting_mgdl") == pytest.approx(129.7, abs=0.2)
    assert p.get("family_history") is True
    assert p.get("smoker") is False
    assert p.is_ready()


def test_negations_and_no_defaults():
    p = extract_profile("tôi không bị tăng huyết áp, nhà không ai bị tiểu đường", "text")
    assert p.get("high_bp") is False
    assert p.get("family_history") is False
    # Không tự điền giá trị mặc định cho trường lâm sàng
    assert not p.has("glucose_fasting_mgdl") and not p.has("bmi")
    assert set(p.missing_required()) == {"age_years", "bmi", "glucose_fasting_mgdl"}


def test_question_is_not_patient_data():
    p = extract_profile("HbA1c bao nhiêu thì là tiền tiểu đường?", "text")
    assert not p.has("family_history")


def test_lab_report_units():
    p = extract_profile("Năm sinh: 1975  Giới tính: Nữ\nHbA1c: 48 mmol/mol\nCreatinin 88 umol/L\n"
                        "Huyết áp: 150/95 mmHg", "file:phieu.pdf")
    assert p.get("sex") == 0
    assert p.get("hba1c_percent") == pytest.approx(6.54, abs=0.02)
    assert p.get("creatinine_mgdl") == pytest.approx(1.0, abs=0.01)
    assert p.get("high_bp") is True  # suy ra từ huyết áp đo được
    assert p.fields["high_bp"].source == "derived"


def test_unit_inference_is_flagged():
    p = extract_profile("đường máu lúc chưa ăn được 6,4", "text")
    fv = p.fields["glucose_fasting_mgdl"]
    assert fv.value == pytest.approx(115.3, abs=0.2)
    assert "unit_inferred" in fv.flags and "fasting_unclear" not in fv.flags


@pytest.mark.parametrize("value,unit,expected", [(7.0, "mmol/L", 126.1), (126, "mg/dL", 126), (5.5, None, 99.1)])
def test_glucose_conversion(value, unit, expected):
    assert glucose_to_mgdl(value, unit)[0] == pytest.approx(expected, abs=0.2)


def test_hba1c_ifcc_conversion():
    assert hba1c_to_percent(48, "mmol/mol")[0] == pytest.approx(6.54, abs=0.02)
