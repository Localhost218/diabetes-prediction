"""Tổng hợp kết quả GSM8K của bốn hệ (outputs/logs/ch7_rows_*.jsonl) -> ch7_gsm8k.json, ch7_traces.json."""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import LOGS, save_json  # noqa: E402
from tl3.ch7_agents import load_gsm8k, same, to_number  # noqa: E402

SYSTEMS = ["direct", "cot", "pal", "multi"]
ERRS = ["no_answer", "arithmetic", "reasoning", "coordination"]


def load_rows():
    rows = {}
    for s in SYSTEMS:
        p = LOGS / f"ch7_rows_{s}.jsonl"
        rows[s] = {r["id"]: r for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())} if p.exists() else {}
    common = set.intersection(*(set(r) for r in rows.values() if r)) if all(rows.values()) else set()
    return rows, sorted(common)


def bootstrap_ci(x, n=2000, seed=0):
    """Khoảng tin cậy 95 % của accuracy bằng bootstrap trên các câu hỏi."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def steps_of(answer):
    """Độ khó xấp xỉ: số phép tính <<...>> trong lời giải mẫu của GSM8K."""
    return len(re.findall(r"<<", answer))


def multi_analysis(rows, ids):
    """Cổng đồng thuận của hệ đa tác tử: bao nhiêu câu chốt ngay khi hai Solver khớp, độ đúng ở mỗi nhánh, cận trên oracle."""
    agree, dis, oracle = [], [], 0
    for i in ids:
        r = rows["multi"][i]
        disagreed = any(s["role"] == "Coordinator" for s in r["steps"])   # Coordinator chỉ xuất hiện khi bất đồng
        (dis if disagreed else agree).append(r["correct"])
        cands = [to_number(s["text"].split("=")[-1]) for s in r["steps"]
                 if s["role"] == "Tool:python" and s["text"].startswith("kết quả")]
        cands += [to_number(s["text"]) for s in r["steps"] if s["role"] in ("Solver-CoT", "Verifier")
                  and not s["text"].startswith("hai solver")]
        oracle += any(same(c, r["gold"]) for c in cands)
    n = len(ids)
    # lần chạy công cụ đầu tiên của Solver-PAL (nhận kế hoạch của Planner) thiếu hàm solution() -> phải gọi lại
    first_tool = [next((s["text"] for s in rows["multi"][i]["steps"] if s["role"] == "Tool:python"), "") for i in ids]
    first_pal = [next((s["text"] for s in rows["pal"][i]["steps"] if s["role"] == "Tool:python"), "") for i in ids]
    # chi phí từng nhánh; lượt gọi theo thiết kế = Planner + Solver-PAL + Solver-CoT (+ Verifier + PAL#2 khi bất đồng),
    # phần vượt quá là lượt tự sửa mã
    by = {"agree": [], "disagree": []}
    for i in ids:
        r = rows["multi"][i]
        by["disagree" if any(s["role"] == "Coordinator" for s in r["steps"]) else "agree"].append(r)
    cost = {f"{k}_{m}": float(np.mean([r[f] for r in v])) if v else 0.0
            for k, v in by.items() for m, f in (("calls", "calls"), ("seconds", "wall_seconds"))}
    repair = [r["calls"] - (5 if any(s["role"] == "Coordinator" for s in r["steps"]) else 3)
              for r in (rows["multi"][i] for i in ids)]

    def rescued(steps, stop_role=None):
        """Lần chạy đầu thiếu solution() nhưng một lần chạy sau (trước stop_role) cho ra số."""
        cut = next((k for k, s in enumerate(steps) if s["role"] == stop_role), len(steps)) if stop_role else len(steps)
        runs = [s["text"] for s in steps[:cut] if s["role"] == "Tool:python"]
        if not runs or "solution()" not in runs[0]:
            return None
        return any(t.startswith("kết quả") and not t.endswith("None") for t in runs[1:])
    res_m = [x for x in (rescued(rows["multi"][i]["steps"], "Solver-CoT") for i in ids) if x is not None]
    res_p = [x for x in (rescued(rows["pal"][i]["steps"]) for i in ids) if x is not None]
    return {"first_nofunc_multi": sum("solution()" in t for t in first_tool) / n,
            "first_nofunc_pal": sum("solution()" in t for t in first_pal) / n,
            "rescued_multi": float(np.mean(res_m)) if res_m else 0.0,
            "rescued_pal": float(np.mean(res_p)) if res_p else 0.0,
            "repair_calls": float(np.mean(repair)), **cost,
            "agree_n": len(agree), "agree_share": len(agree) / n, "agree_acc": float(np.mean(agree)) if agree else 0.0,
            "disagree_n": len(dis), "disagree_acc": float(np.mean(dis)) if dis else 0.0, "oracle_acc": oracle / n}


def main():
    rows, ids = load_rows()
    qs = {q["id"]: q for q in load_gsm8k(250)}
    out = {"n": len(ids), "systems": {}}
    # mỗi hệ có thể đã chạy nhiều câu hơn tập chung; báo cả n và accuracy trên toàn bộ số câu đã chạy của từng hệ
    out["full"] = {s: {"n": len(rows[s]), "acc": float(np.mean([r["correct"] for r in rows[s].values()]))}
                   for s in SYSTEMS if rows[s]}
    for s in SYSTEMS:
        rr = [rows[s][i] for i in ids]
        acc = [r["correct"] for r in rr]
        lo, hi = bootstrap_ci(acc)
        errs = Counter(r["error"] for r in rr if not r["correct"])
        out["systems"][s] = {
            "acc": float(np.mean(acc)), "ci_lo": lo, "ci_hi": hi, "correct": int(sum(acc)),
            "calls": float(np.mean([r["calls"] for r in rr])), "tokens_in": float(np.mean([r["tokens_in"] for r in rr])),
            "tokens_out": float(np.mean([r["tokens_out"] for r in rr])),
            "tokens": float(np.mean([r["tokens_in"] + r["tokens_out"] for r in rr])),
            "seconds": float(np.mean([r["wall_seconds"] for r in rr])),
            "seconds_median": float(np.median([r["wall_seconds"] for r in rr])),
            "seconds_p90": float(np.percentile([r["wall_seconds"] for r in rr], 90)),
            "tok_per_s": sum(r["tokens_out"] for r in rr) / max(sum(r["llm_seconds"] for r in rr), 1e-9),
            "errors": {e: int(errs.get(e, 0)) for e in ERRS},
            "series": {"seconds": [r["wall_seconds"] for r in rr], "tokens_out": [r["tokens_out"] for r in rr],
                       "calls": [r["calls"] for r in rr], "correct": acc},
        }
        tool = [st["text"] for r in rr for st in r["steps"] if st["role"] == "Tool:python"]
        if tool:
            kinds = Counter("ok" if t.startswith("kết quả") and not t.endswith("None") else
                            "timeout" if "timeout" in t else "import" if "not allowed" in t else
                            "no_function" if "solution()" in t else "runtime" for t in tool)
            out["systems"][s]["tool"] = {"runs": len(tool), **{k: int(kinds.get(k, 0)) for k in
                                                                ("ok", "timeout", "import", "no_function", "runtime")}}
    base = out["systems"]
    out["multi"] = multi_analysis(rows, ids)
    for ref in ("pal", "cot"):
        fixed = sum(rows["multi"][i]["correct"] and not rows[ref][i]["correct"] for i in ids)
        broken = sum(not rows["multi"][i]["correct"] and rows[ref][i]["correct"] for i in ids)
        extra_calls = (base["multi"]["calls"] - base[ref]["calls"]) * len(ids)
        gain = base["multi"]["correct"] - base[ref]["correct"]
        out[f"vs_{ref}"] = {"fixed": int(fixed), "broken": int(broken), "net": int(gain),
                            "gain": base["multi"]["acc"] - base[ref]["acc"],
                            "calls_per_extra_correct": extra_calls / gain if gain > 0 else None,
                            "seconds_ratio": base["multi"]["seconds"] / base[ref]["seconds"]}
    out["cot_vs_direct"] = base["cot"]["acc"] - base["direct"]["acc"]
    out["pal_vs_cot"] = base["pal"]["acc"] - base["cot"]["acc"]
    # ma trận "đúng ở hệ hàng nhưng sai ở hệ cột"
    out["only_correct"] = {"rows": [[int(sum(rows[a][i]["correct"] and not rows[b][i]["correct"] for i in ids))
                                     for b in SYSTEMS] for a in SYSTEMS]}
    # độ chính xác theo độ khó
    bins = [(1, 2), (3, 3), (4, 4), (5, 20)]
    diff = []
    for lo, hi in bins:
        sel = [i for i in ids if lo <= steps_of(qs[i]["answer"]) <= hi]
        if not sel:
            continue
        diff.append({"bin": f"{lo}-{hi}" if hi < 20 and lo != hi else str(lo) if lo == hi else f">={lo}", "n": len(sel),
                     **{s: float(np.mean([rows[s][i]["correct"] for i in sel])) if sel else None for s in SYSTEMS}})
    out["difficulty"] = {"rows": diff}
    save_json("ch7_gsm8k", out)
    save_json("ch7_traces", {"traces": pick_traces(rows, ids, qs)})
    print({s: round(base[s]["acc"], 3) for s in SYSTEMS})


def pick_traces(rows, ids, qs):
    """Chọn vết mẫu: đa tác tử đồng thuận đúng (ngắn), bất đồng được Verifier sửa đúng, lỗi phối hợp, PAL tự sửa lỗi mã."""
    def multi_kind(r):
        return "disagree" if any(s["role"] == "Coordinator" for s in r["steps"]) else "agree"
    want = {}
    for i in ids:
        m = rows["multi"][i]
        if "agree_ok" not in want and multi_kind(m) == "agree" and m["correct"] and len(qs[i]["question"]) < 260:
            want["agree_ok"] = m
        if "disagree_ok" not in want and multi_kind(m) == "disagree" and m["correct"] and not rows["pal"][i]["correct"]:
            want["disagree_ok"] = m
        if "coordination" not in want and m["error"] == "coordination":
            want["coordination"] = m
        p = rows["pal"][i]
        if "pal_fix" not in want and p["correct"] and sum(s["role"] == "PAL" for s in p["steps"]) > 1:
            want["pal_fix"] = p
    return {k: {**v, "question": qs[v["id"]]["question"]} for k, v in want.items()}


if __name__ == "__main__":
    main()
