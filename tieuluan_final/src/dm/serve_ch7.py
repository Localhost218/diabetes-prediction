"""Web App — Chương 7 (mạch tiểu đường): trả lời trắc nghiệm y khoa đái tháo đường bằng 4 hệ tác tử (direct, cot, rag, multi)."""
import json
import re
import time
from functools import lru_cache
from pathlib import Path

from . import ch7_medqa as M

TLF = Path(__file__).resolve().parents[2]
ROWS_DIR = TLF / "outputs" / "ch7"
SYSTEM_IDS = ["multi", "rag", "cot", "direct"]
SYSTEM_NAMES = {
    "multi": "Đa tác tử (Solver-CoT ‖ Solver-RAG + Critic + Vote)",
    "rag": "RAG-agent (BM25 MedQuAD-NIDDK + LLM)",
    "cot": "Một tác tử suy luận từng bước (CoT)",
    "direct": "Một tác tử trả lời trực tiếp (Direct)",
}
TOOLS = {
    "direct": ["LLM (Qwen2.5-1.5B)"],
    "cot": ["LLM (Chain-of-Thought)"],
    "rag": ["LLM", "BM25 Search (MedQuAD-NIDDK)"],
    "multi": ["Solver-CoT", "Solver-RAG", "BM25 Search (MedQuAD-NIDDK)", "Critic", "Solver-CoT#2", "Coordinator"],
}
EXAMPLE_Q = (
    "Question: Glucose mediated insulin release is mediated through:\n"
    "A. ATP dependent K-t channels\n"
    "B. cAMP\n"
    "C. Carrier modulators\n"
    "D. Receptor phosphorylation"
)

SPECS = {
    "ch7_dm_medqa": {
        "title": "Hệ đa tác tử giải trắc nghiệm y khoa đái tháo đường (MedMCQA + MedQuAD)",
        "kind": "chat",
        "description": "So sánh 4 hệ (Direct, CoT, RAG-agent, Đa tác tử tranh luận) dùng Qwen2.5-1.5B-Instruct cục bộ và kho tri thức MedQuAD-NIDDK.",
        "inputs": [
            {"name": "question", "label": "Câu hỏi trắc nghiệm (kèm phương án A–D)", "type": "text", "default": EXAMPLE_Q},
            {"name": "sample", "label": "Hoặc chọn mã câu trong bộ 150 câu kiểm tra (0–149, để trống nếu dùng ô trên)",
             "type": "text", "default": ""},
            {"name": "system", "label": "Hệ tác tử", "type": "select", "options": SYSTEM_IDS, "default": "multi"},
        ],
        "models": SYSTEM_IDS,
        "model_names": SYSTEM_NAMES,
        "example": {"question": EXAMPLE_Q, "sample": "", "system": "multi"},
    }
}

_STATE = {}


@lru_cache(maxsize=1)
def _questions():
    qs = M.load_questions(150)
    by_text = {M.format_q(q).strip(): q for q in qs}
    by_qonly = {q["question"].strip(): q for q in qs}
    return qs, by_text, by_qonly


def _cached_row(system, qid):
    p = ROWS_DIR / f"rows_{system}.jsonl"
    if not p.exists():
        return None
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("id") == qid:
            return r
    return None


def _parse_custom(text):
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    opts = []
    q_lines = []
    for l in lines:
        m = re.match(r"^[ABCD][\.\)]\s*(.+)$", l)
        if m:
            opts.append(m.group(1).strip())
        else:
            q_lines.append(re.sub(r"^Question:\s*", "", l, flags=re.I))
    while len(opts) < 4:
        opts.append("—")
    return {"id": -1, "mid": "custom", "question": " ".join(q_lines) or text.strip(),
            "options": opts[:4], "gold": None, "subject": "Tự nhập"}


def _llm_and_tool():
    if "llm" not in _STATE:
        from tl3.ch7_llm import load_lowmem
        _STATE["llm"] = load_lowmem()
    if "tool" not in _STATE:
        _STATE["tool"] = M.KnowledgeTool()
    return _STATE["llm"], _STATE["tool"]


def predict(exp_id, payload, model_id=None):
    if exp_id != "ch7_dm_medqa":
        raise KeyError(exp_id)
    t0 = time.perf_counter()
    system = model_id or payload.get("system") or "multi"
    if system not in M.SYSTEMS:
        raise ValueError(f"Hệ không hợp lệ: {system}")
    qs, by_text, by_qonly = _questions()
    raw_sample = str(payload.get("sample") or "").strip()
    raw_q = str(payload.get("question") or "").strip()

    matched = None
    if raw_sample.isdigit() and 0 <= int(raw_sample) < len(qs):
        matched = qs[int(raw_sample)]
    elif raw_q in by_text:
        matched = by_text[raw_q]
    elif raw_q in by_qonly:
        matched = by_qonly[raw_q]

    if matched is not None:
        cached = _cached_row(system, matched["id"])
        if cached is not None:
            pred = cached.get("pred")
            gold = matched["gold"]
            opt_txt = matched["options"][M.LETTERS.index(pred)] if pred in M.LETTERS else "Không xác định"
            ans_str = f"Đáp án: {pred} ({opt_txt}) — {'✓ Đúng' if pred == gold else f'✗ Sai (đáp án đúng: {gold})'}"
            return {
                "answer": ans_str,
                "label": pred or "—",
                "trace": [{"step": s["role"], "text": s["text"]} for s in cached.get("steps", [])],
                "tools": TOOLS[system],
                "details": {
                    "true_label": f"{gold} ({matched['options'][M.LETTERS.index(gold)]})",
                    "correct": pred == gold,
                    "calls": cached.get("calls", 1),
                    "tokens_in": cached.get("tokens_in", 0),
                    "tokens_out": cached.get("tokens_out", 0),
                    "dataset": f"MedMCQA – {matched.get('subject', '')} (câu #{matched['id']})",
                },
                "model": system,
                "latency_ms": round(cached.get("wall_seconds", time.perf_counter() - t0) * 1000, 1),
            }
        row = matched
    else:
        if not raw_q:
            matched = by_text.get(EXAMPLE_Q.strip(), qs[0])
            row = matched
        else:
            row = _parse_custom(raw_q)

    llm, tool = _llm_and_tool()
    pred, tr = M.SYSTEMS[system](llm, row, tool)
    gold = row.get("gold")
    opt_txt = row["options"][M.LETTERS.index(pred)] if pred and pred in M.LETTERS else ""
    ans_str = f"Đáp án: {pred or '—'}" + (f" ({opt_txt})" if opt_txt else "")
    details = {"calls": tr.calls, "tokens_in": tr.tokens_in, "tokens_out": tr.tokens_out}
    if gold:
        details["true_label"] = gold
        details["correct"] = pred == gold
    return {
        "answer": ans_str,
        "label": pred or "—",
        "trace": [{"step": s["role"], "text": s["text"]} for s in tr.steps],
        "tools": TOOLS[system],
        "details": details,
        "model": system,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
    }
