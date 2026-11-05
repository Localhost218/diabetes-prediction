"""Chương 7 — chạy bốn hệ trên n câu trắc nghiệm tiểu đường, ghi từng câu vào outputs/ch7/rows_<hệ>.jsonl
(chạy lại thì tiếp tục từ câu chưa làm) và tóm tắt vào outputs/metrics/ch7_medqa.json.

    python scripts/run_ch7_medqa.py --n 150 --systems direct,cot,rag,multi
"""
import argparse
import json
import sys
import time
from pathlib import Path

TLF = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(TLF / "src"), str(TLF / "src" / "tl03")]
from dm.ch7_medqa import KnowledgeTool, load_questions, run_question  # noqa: E402

ROWS = TLF / "outputs" / "ch7"
OUT = TLF / "outputs" / "metrics"
ROWS.mkdir(parents=True, exist_ok=True)
OUT.mkdir(parents=True, exist_ok=True)


def run_system(llm, tool, system, questions):
    path = ROWS / f"rows_{system}.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    with path.open("a", encoding="utf-8") as f:
        for q in questions:
            if q["id"] in done:
                continue
            r = run_question(llm, system, q, tool)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{time.strftime('%H:%M:%S')}] {system} #{q['id']}: {r['pred']} / {r['gold']} "
                  f"{'✓' if r['correct'] else '✗'} {r['calls']} lượt {r['wall_seconds']:.1f}s", flush=True)


def summarize(systems, n):
    res = {"n": n, "systems": {}}
    for s in systems:
        rows = [json.loads(l) for l in (ROWS / f"rows_{s}.jsonl").read_text().splitlines()]
        k = len(rows)
        errs = {}
        for r in rows:
            if r["error"]:
                errs[r["error"]] = errs.get(r["error"], 0) + 1
        res["systems"][s] = {"n": k, "accuracy": sum(r["correct"] for r in rows) / k,
                             "no_answer": sum(r["pred"] is None for r in rows) / k,
                             "calls": sum(r["calls"] for r in rows) / k,
                             "tokens_out": sum(r["tokens_out"] for r in rows) / k,
                             "seconds": sum(r["wall_seconds"] for r in rows) / k, "errors": errs}
    (OUT / "ch7_medqa.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--systems", default="direct,cot,rag,multi")
    a = ap.parse_args()
    systems = a.systems.split(",")
    questions = load_questions(a.n)
    tool = KnowledgeTool()
    print(f"{len(questions)} câu, kho MedQuAD {len(tool.docs)} đoạn", flush=True)
    from tl3.ch7_llm import load_lowmem
    llm = load_lowmem()
    for s in systems:
        run_system(llm, tool, s, questions)
    print(json.dumps(summarize(systems, a.n), ensure_ascii=False, indent=1))
    print("XONG", flush=True)


if __name__ == "__main__":
    main()
