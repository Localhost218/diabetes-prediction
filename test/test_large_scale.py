"""Ý kiến thứ hai quy mô lớn (BRFSS): hội đồng chỉ chất vấn khi hai mô hình nằm ở hai phía ngược nhau của mức trung bình."""

import pytest

from src.agents.consensus_critic_agent import _apply_response, _detect_issues


def _state(rr_main, rr_large):
    ml = {"risk_probability": 0.2, "relative_risk": rr_main, "data_completeness": 0.8, "disagreement": 0.0,
          "top_risk_factors": [],
          "large_scale": {"dataset": "BRFSS", "n_train": 2_220_973, "relative_risk": rr_large}}
    return {"ml": ml, "ml_confidence": 0.85, "glycemia": {"status": "prediabetes_range"}, "plus": [], "minus": [],
            "knowledge": {"graph_rag": {"start_entities": []}}}


@pytest.mark.parametrize("rr_main, rr_large, expected", [
    (2.0, 0.5, True),    # chính: cao, quy mô lớn: thấp → chất vấn
    (0.5, 2.0, True),
    (2.0, 1.2, False),   # cùng phía (đều ≥ 1) → không chất vấn
    (1.3, 0.9, False),   # lệch nhưng chưa đủ xa ngưỡng 1,5 / 0,8
])
def test_cohort_disagreement_detection(rr_main, rr_large, expected):
    issues = [i[0] for i in _detect_issues(_state(rr_main, rr_large))]
    assert ("cohort_disagree" in issues) is expected


def test_cohort_response_lowers_confidence_and_adds_minus():
    state = _state(2.0, 0.5)
    _apply_response(state, "cohort_disagree", {"statement": "giải trình", "adjusted_confidence": 0.6})
    assert state["ml_confidence"] == 0.6
    assert "quy mô lớn bất đồng" in state["minus"][0]


def test_no_large_scale_model_means_no_issue():
    state = _state(2.0, 0.5)
    state["ml"]["large_scale"] = None
    assert "cohort_disagree" not in [i[0] for i in _detect_issues(state)]
