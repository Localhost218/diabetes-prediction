"""Chương 5 — so sánh bốn hệ hỏi đáp trên tập con câu hỏi có đáp án của một miền:
    LLM thuần (không truy xuất) · Naive RAG (dense top-5) · Advanced RAG (RRF + đồ thị + rerank) · Tác tử định tuyến.

    python scripts/ch5_eval_agent.py <miền>
Ghi outputs/metrics/ch5_agent_<miền>.json: độ chính xác đáp án theo hệ và loại câu hỏi, độ trễ, số token, vết
suy luận mẫu (kể cả câu hỏi mở không có đáp án).
"""
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import data_dir, limit_threads, save_json, seed_all  # noqa: E402
from tl3.gpu import heavy  # noqa: E402
from tl3.ch5_agent import Ch5Agent, Retrievers, load_domain  # noqa: E402
from tl3.ch5_qa import OPEN_QUESTIONS, is_correct  # noqa: E402
from tl3.llm import LocalLLM  # noqa: E402

SYSTEMS = {"llm_only": "LLM thuần", "naive_rag": "Naive RAG", "advanced_rag": "Advanced RAG", "agent": "Tác tử"}
PER_TYPE = 8


def subset(qs, seed=7):
    """Lấy tối đa PER_TYPE câu mỗi (loại, kiểu diễn đạt) để tập con cân bằng."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for q in qs:
        groups[(q["type"], q["style"])].append(q)
    out = []
    for k in sorted(groups):
        out += rng.sample(groups[k], min(PER_TYPE, len(groups[k])))
    return out


def main(dom):
    limit_threads(4)
    seed_all()
    qs = subset(json.loads((data_dir("ch5") / f"qa_{dom}.json").read_text(encoding="utf-8")))
    rows, traces = [], []
    with heavy(f"ch5 tác tử {dom}"):
        # nạp LLM trước: bản sao trọng số bf16 tạm trên CPU được giải phóng trước khi KG và chỉ mục chiếm RAM
        llm = LocalLLM(max_new_tokens=220)
        D = load_domain(dom)
        A = Ch5Agent(D, Retrievers(D), llm)
        for q in qs:
            for s in SYSTEMS:
                out = A.run(q["question"]) if s == "agent" else getattr(A, s)(q["question"])
                rows.append({"system": s, "qid": q["qid"], "type": q["type"], "style": q["style"],
                             "correct": bool(is_correct(q, out["answer"])), "ms": out["latency_ms"],
                             "tools": out["tools"], "answer": out["answer"][:400],
                             "gold_in_sources": bool(set(q["gold"]) & set(out["sources"]))})
            print(q["qid"], [r["correct"] for r in rows[-4:]], flush=True)
        # vết suy luận mẫu: một câu tra cứu, một câu tổng hợp nhóm/giai đoạn, ba câu hỏi mở
        demo = [next(q for q in qs if q["type"] == "record")["question"],
                next(q for q in qs if q["type"] != "record")["question"]] + OPEN_QUESTIONS[dom]
        for qq in demo:
            out = A.run(qq)
            traces.append({"question": qq, "trace": [list(t) for t in out["trace"]], "tools": out["tools"],
                           "sources": out["sources"], "ms": out["latency_ms"]})
    acc = {}
    for s in SYSTEMS:
        rs = [r for r in rows if r["system"] == s]
        acc[s] = {"acc": float(np.mean([r["correct"] for r in rs])), "ms": float(np.mean([r["ms"] for r in rs])),
                  "ms_median": float(np.median([r["ms"] for r in rs])),
                  "gold_in_sources": float(np.mean([r["gold_in_sources"] for r in rs])), "n": len(rs)}
        for t in sorted({r["type"] for r in rs}):
            acc[s][f"acc_{t}"] = float(np.mean([r["correct"] for r in rs if r["type"] == t]))
        for st in ("lex", "para"):
            sub = [r["correct"] for r in rs if r["style"] == st]
            if sub:
                acc[s][f"acc_{st}"] = float(np.mean(sub))
    tool_use = defaultdict(int)
    for r in rows:
        if r["system"] == "agent":
            tool_use[r["tools"][0]] += 1
    save_json(f"ch5_agent_{dom}", {"n_questions": len(qs), "systems": acc, "tool_use": dict(tool_use),
                                    "rows": rows, "traces": traces})


if __name__ == "__main__":
    main(sys.argv[1])
