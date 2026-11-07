"""Tóm tắt thông điệp cho Phòng hội chẩn: mỗi loại thông điệp ra một câu tiếng Việt, không lỗi khi payload thiếu trường."""

from src.communication.summary import MAX_LEN, summarize


def _msg(perf, msg_type, payload=None):
    return {"sender": "A", "receiver": "B", "performative": perf, "msg_type": msg_type, "payload": payload or {}}


def test_request_and_challenge_use_fixed_phrases():
    assert "ML + DL" in summarize(_msg("REQUEST", "PREDICT_RISK"))
    assert "khác nhau" in summarize(_msg("CHALLENGE", "CHALLENGE_ML_VS_LABS"))
    assert summarize(_msg("REQUEST", "COMPOSE_ASSESSMENT")).startswith("Viết câu trả lời")


def test_inform_payloads():
    extracted = summarize(_msg("INFORM", "PATIENT_DATA_EXTRACTED",
                               {"profile_delta": {"fields": {"age_years": {}, "hba1c_percent": {}}}}))
    assert "Tuổi" in extracted and "HbA1c" in extracted
    risk = summarize(_msg("INFORM", "RISK_PREDICTION", {"risk_model": {
        "risk_probability": 0.47, "relative_risk": 2.38,
        "models": {"ml": {"risk_probability": 0.45}, "dl": {"risk_probability": 0.5}}}}))
    assert "47%" in risk and "ML 45%" in risk and "DL 50%" in risk
    verdict = summarize(_msg("INFORM", "CONSENSUS_RESULT", {"risk_label": "Cao", "positive_evidence_plus": [1, 2],
                                                            "negative_contradictions_minus": [1], "debate_rounds": 2}))
    assert "2 (+)" in verdict and "1 (−)" in verdict and "2 vòng" in verdict


def test_long_statement_is_truncated_and_failure_reported():
    long = summarize(_msg("RESPONSE", "CHALLENGE_RESPONSE", {"statement": "x " * 300}))
    assert len(long) <= MAX_LEN and long.endswith("…")
    assert summarize(_msg("FAILURE", "PREDICT_RISK_FAILED", {"error": "boom"})) == "Lỗi: boom"


def test_unknown_or_malformed_payload_falls_back_to_msg_type():
    assert summarize(_msg("INFORM", "SOMETHING_NEW", {"weird": 1})) == "SOMETHING_NEW"
    assert summarize(_msg("INFORM", "RISK_PREDICTION", {"risk_model": {}})) == "RISK_PREDICTION"
