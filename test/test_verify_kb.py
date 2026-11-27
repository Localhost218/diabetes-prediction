import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("verify_kb", Path(__file__).resolve().parent.parent / "scripts" / "verify_kb.py")
verify_kb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify_kb)


def test_numbers_match_across_decimal_styles():
    assert verify_kb.missing_numbers("FPG ≥ 126 mg/dL (7,0 mmol/L).", ["FPG ≥126 mg/dL (≥7.0 mmol/L)"]) == set()


def test_mistranslated_number_is_caught():
    assert verify_kb.missing_numbers("HbA1c ≥ 6,8%", ["A1C ≥6.5% (48 mmol/mol)"]) == {6.8}
