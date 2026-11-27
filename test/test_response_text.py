from src.agents.response_agent import _whole_sentences


def test_cut_off_llm_sentence_is_dropped():
    assert _whole_sentences("Bạn nên đo lại. Hãy ăn uống lành mạnh hơn, tăng thời gian") == "Bạn nên đo lại."
    assert _whole_sentences("Không có dấu câu") == "Không có dấu câu"
