"""Suy luận Chương 7 cho Web App: giải bài toán lời văn bằng một trong bốn hệ (direct, cot, pal, multi)."""
import time

from .ch7_agents import SYSTEMS

SYSTEM_IDS = ["multi", "pal", "cot", "direct"]
TOOLS = {"direct": ["LLM"], "cot": ["LLM (chain-of-thought)"], "pal": ["LLM", "Python sandbox"],
         "multi": ["Planner", "Solver-PAL", "Solver-CoT", "Python sandbox", "Verifier", "Coordinator"]}
EXAMPLE = ("Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her "
           "friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. "
           "How much in dollars does she make every day at the farmers' market?")

SPECS = {"ch7_math": {
    "title": "Hệ đa tác tử giải toán lời văn (GSM8K) với Qwen2.5-1.5B cục bộ",
    "kind": "chat",
    "inputs": [{"name": "question", "label": "Bài toán (tiếng Anh)", "type": "text", "default": EXAMPLE},
               {"name": "system", "label": "Hệ giải", "type": "select", "options": SYSTEM_IDS, "default": "multi"}],
    "models": SYSTEM_IDS,
    "example": {"question": EXAMPLE, "system": "multi"},
}}

_LLM = {}


def _llm():
    if "m" not in _LLM:
        from .ch7_llm import load_lowmem
        _LLM["m"] = load_lowmem()
    return _LLM["m"]


def predict(exp_id, payload, model_id=None):
    """payload {"question": str, "system": direct|cot|pal|multi} -> đáp án, vết từng tác tử, công cụ đã dùng."""
    if exp_id != "ch7_math":
        raise KeyError(exp_id)
    system = model_id or payload.get("system", "multi")
    if system not in SYSTEMS:
        raise ValueError(f"hệ không hợp lệ: {system}")
    q = (payload.get("question") or "").strip()
    if not q:
        raise ValueError("thiếu câu hỏi")
    t0 = time.perf_counter()
    answer, tr = SYSTEMS[system](_llm(), q)
    return {"answer": "không tìm được đáp án" if answer is None else str(answer),
            "trace": [{"step": s["role"], "text": s["text"]} for s in tr.steps],
            "tools": TOOLS[system],
            "details": {"calls": tr.calls, "tokens_in": tr.tokens_in, "tokens_out": tr.tokens_out},
            "model": system, "latency_ms": round(1000 * (time.perf_counter() - t0), 1)}
