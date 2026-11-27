from src.schemas.patient_profile import PatientProfile


def test_apply_form_coerces_strings_skips_blanks_and_rejects_out_of_range():
    p = PatientProfile()
    rejected = p.apply_form({"age_years": "52", "sex": "1", "height_cm": "165", "weight_kg": "72",
                             "family_history": "yes", "smoker": "no", "waist_cm": "", "hba1c_percent": "45",
                             "unknown_field": "x"})
    assert sorted(rejected) == ["hba1c_percent", "unknown_field"]
    assert p.get("age_years") == 52 and p.get("sex") == 1
    assert p.get("family_history") is True and p.get("smoker") is False
    assert not p.has("waist_cm")
    assert round(p.get("bmi"), 1) == 26.4  # suy ra từ chiều cao/cân nặng
    assert "Tuổi: 52 tuổi (nhập form)" in p.summary_vi()
