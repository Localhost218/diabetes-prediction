import asyncio
from types import SimpleNamespace

import src.agents.clarification_agent as ca


def _run(text, issues, monkeypatch):
    async def fake_parse(*_):
        return SimpleNamespace(issues=[SimpleNamespace(**i, model_dump=lambda i=i: dict(i)) for i in issues])
    monkeypatch.setattr(ca.llm, "enabled", True, raising=False)
    monkeypatch.setattr(ca.llm, "parse", fake_parse)
    return asyncio.run(ca._scope_check(text, []))["issues"]


def test_other_issue_quoting_whole_message_is_dropped(monkeypatch):
    text = "Tôi có dấu hiệu bị tiểu đường hay không"
    assert _run(text, [{"reason": "OTHER", "evidence": text, "question": "Bạn có đang điều trị cho một người khác không?",
                        "blocking": False}], monkeypatch) == []


def test_specific_other_issue_is_kept(monkeypatch):
    text = "tôi 45 tuổi, đang nằm viện vì viêm phổi, đường huyết 180"
    kept = _run(text, [{"reason": "OTHER", "evidence": "đang nằm viện", "question": "Bạn nằm viện vì bệnh gì?",
                        "blocking": False}], monkeypatch)
    assert len(kept) == 1


def test_llm_question_repeating_a_rule_question_is_skipped():
    cands = [{"field": "glucose_fasting_mgdl"}]
    assert ca._repeats_rule_question("Bạn có thể cung cấp đường huyết lúc đói gần nhất?", cands)
    assert not ca._repeats_rule_question("Bạn nằm viện vì bệnh gì?", cands)
