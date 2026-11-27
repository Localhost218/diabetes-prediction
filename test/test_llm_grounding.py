from src.llm.prompts import ExtractedClinicalData, Measurement


def test_llm_numbers_not_in_message_are_dropped():
    data = ExtractedClinicalData(
        measurements=[Measurement(field="glucose_fasting", value=126, unit="mg/dL", evidence="x"),
                      Measurement(field="hba1c", value=6.8, unit="%", evidence="HbA1c 6,8%")],
        glucose_series=[110, 120, 130, 140, 150, 160, 170, 180, 190, 200])
    p = data.to_profile("text", source_text="tôi hay ăn khuya và ít ăn rau, HbA1c 6,8%")
    assert p.glucose_history is None  # chuỗi đo bịa ra không được nhận
    assert not p.has("glucose_fasting_mgdl") and p.get("hba1c_percent") == 6.8


def test_without_source_text_values_are_kept():
    data = ExtractedClinicalData(glucose_series=[100, 110, 120, 130, 140])
    assert data.to_profile("file:x").glucose_history == [100, 110, 120, 130, 140]
