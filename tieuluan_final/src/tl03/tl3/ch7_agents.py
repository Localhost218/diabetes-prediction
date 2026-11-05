"""Chương 7 — Hệ đa tác tử trên GSM8K với LLM cục bộ (Qwen2.5-1.5B-Instruct).

Bốn hệ được so sánh trên cùng tập câu hỏi:
  direct : LLM trả lời ngay một con số
  cot    : một tác tử suy luận từng bước (chain-of-thought)
  pal    : một tác tử viết chương trình Python, công cụ thực thi trong hộp cát trả kết quả
  multi  : Planner -> Solver(PAL) ‖ Solver(CoT) -> Verifier; bất đồng thì Critic phân xử + biểu quyết đa số
"""
import json
import random
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field

from .common import DATA, SEED

GSM8K = DATA / "ch7" / "gsm8k_test.jsonl"
NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def load_gsm8k(n=250, seed=SEED):
    """Tập con cố định n câu của GSM8K test; đáp án vàng là số sau '####'."""
    rows = [json.loads(l) for l in GSM8K.read_text(encoding="utf-8").splitlines() if l.strip()]
    for i, r in enumerate(rows):
        r["id"] = i
        r["gold"] = to_number(r["answer"].split("####")[-1])
    rng = random.Random(seed)
    return sorted(rng.sample(rows, n), key=lambda r: r["id"])


def to_number(text):
    """Lấy số cuối cùng trong văn bản ('1,250.50' -> 1250.5); None nếu không có."""
    if text is None:
        return None
    nums = NUM_RE.findall(str(text).replace("$", ""))
    if not nums:
        return None
    s = nums[-1].replace(",", "").rstrip(".")
    try:
        v = float(s)
    except ValueError:
        return None
    return int(v) if v.is_integer() else round(v, 4)


def same(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) < 1e-4


# ---------------------------------------------------------------- công cụ: thực thi Python trong hộp cát
SANDBOX_PRELUDE = r'''
import builtins, math, fractions, decimal, itertools, statistics, datetime
_ALLOWED = {"math", "fractions", "decimal", "itertools", "statistics", "datetime"}
_real_import = builtins.__import__
def _safe_import(name, *a, **k):
    if name.split(".")[0] not in _ALLOWED:
        raise ImportError("module not allowed in sandbox: " + name)
    return _real_import(name, *a, **k)
builtins.__import__ = _safe_import
for _n in ("open", "input", "exec", "eval", "compile", "breakpoint", "help"):
    builtins.__dict__.pop(_n, None)
'''


def run_python(code, timeout=5.0):
    """Chạy mã do LLM sinh trong tiến trình con riêng (-I, thư mục tạm, chặn import ngoài danh sách, giới hạn
    thời gian). Mã phải định nghĩa solution(); trả (giá trị, lỗi)."""
    program = SANDBOX_PRELUDE + "\n" + code + "\n\nprint(repr(solution()))\n"
    with tempfile.TemporaryDirectory() as tmp:
        try:
            p = subprocess.run([sys.executable, "-I", "-c", program], cwd=tmp, capture_output=True, text=True,
                               timeout=timeout, env={"PYTHONIOENCODING": "utf-8", "SYSTEMROOT": "C:\\Windows"})
        except subprocess.TimeoutExpired:
            return None, "timeout"
    if p.returncode != 0:
        return None, (p.stderr.strip().splitlines() or ["lỗi"])[-1][:200]
    return to_number(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else None), None


def extract_code(text):
    """Lấy khối mã Python đầu tiên trong câu trả lời của LLM."""
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.S)
    code = m.group(1) if m else text
    return code if "def solution" in code else None


# ---------------------------------------------------------------- lời nhắc của từng vai trò
SYS_DIRECT = "You are a math assistant. Reply with only the final numeric answer, no words, no units."
SYS_COT = ("You are a careful math tutor. Solve the problem step by step, writing each calculation explicitly. "
           "Finish with a line of the form 'The answer is N.' where N is a number.")
SYS_PAL = ("You solve math word problems by writing Python. Write a function solution() that computes and returns the "
           "final numeric answer. Use clear variable names. Output only one ```python code block.")
SYS_PLANNER = ("You are the Planner agent. Read the problem and list the quantities given and the 2-5 calculation "
               "steps needed, as a short numbered plan. Do not compute the final answer.")
SYS_CRITIC = ("You are the Verifier agent. Two solver agents disagree. Check each solution against the problem "
              "statement, find the mistake, then give the correct result. Finish with 'The answer is N.'")

PAL_SHOT = '''Problem: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
```python
def solution():
    money_initial = 23
    bagels = 5
    bagel_cost = 3
    money_spent = bagels * bagel_cost
    return money_initial - money_spent
```'''


@dataclass
class Trace:
    """Vết của một lần giải: các bước (vai trò, nội dung), số lượt gọi LLM, token, thời gian."""
    steps: list = field(default_factory=list)
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0

    def add(self, role, text):
        self.steps.append({"role": role, "text": text})


class Agent:
    """Một tác tử = một vai trò (system prompt) dùng chung LLM; mỗi lần act() là một lượt gọi được ghi vết."""

    def __init__(self, llm, role, system, max_new_tokens=320, temperature=0.0):
        self.llm, self.role, self.system = llm, role, system
        self.max_new_tokens, self.temperature = max_new_tokens, temperature

    def act(self, message, trace):
        text, n_out, sec = self.llm.chat(message, system=self.system, max_new_tokens=self.max_new_tokens,
                                         temperature=self.temperature)
        trace.calls += 1
        trace.tokens_in += len(self.llm.tok.encode(self.system + message))
        trace.tokens_out += n_out
        trace.seconds += sec
        trace.add(self.role, text)
        return text


def solve_direct(llm, q):
    tr = Trace()
    out = Agent(llm, "Direct", SYS_DIRECT, max_new_tokens=24).act(q, tr)
    return to_number(out), tr


def solve_cot(llm, q):
    tr = Trace()
    out = Agent(llm, "CoT", SYS_COT, max_new_tokens=400).act(q, tr)
    return to_number(out), tr


def pal_once(agent, q, trace, plan=None, retries=1):
    """Tác tử PAL: sinh mã -> gọi công cụ Python; lỗi thực thi được gửi lại để tác tử tự sửa (tối đa retries lần)."""
    msg = f"{PAL_SHOT}\n\nProblem: {q}\n" + (f"Plan from the Planner agent:\n{plan}\n" if plan else "")
    for attempt in range(retries + 1):
        out = agent.act(msg, trace)
        code = extract_code(out)
        if code is None:
            val, err = None, "không có hàm solution()"
        else:
            val, err = run_python(code)
        trace.add("Tool:python", f"kết quả = {val}" if err is None else f"lỗi: {err}")
        if err is None and val is not None:
            return val
        msg += f"\nYour previous code failed with: {err}. Fix it and output a corrected code block."
    return None


def solve_pal(llm, q):
    tr = Trace()
    return pal_once(Agent(llm, "PAL", SYS_PAL, max_new_tokens=360), q, tr), tr


_COT_BY_Q = {}


def _get_cot_by_q():
    from .common import LOGS
    p = LOGS / "ch7_rows_cot.jsonl"
    if not _COT_BY_Q and p.exists():
        qs = {r["id"]: r["question"] for r in load_gsm8k(250)}
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["id"] in qs and r.get("steps"):
                _COT_BY_Q[qs[r["id"]]] = r
    return _COT_BY_Q


def solve_multi(llm, q):
    """Planner -> hai Solver độc lập (PAL theo kế hoạch, CoT) -> Verifier.
    Đồng thuận: chấp nhận ngay. Bất đồng: Critic phân xử, thêm một mẫu PAL nhiệt độ 0,7, rồi biểu quyết đa số."""
    tr = Trace()
    plan = Agent(llm, "Planner", SYS_PLANNER, max_new_tokens=160).act(q, tr)
    a_pal = pal_once(Agent(llm, "Solver-PAL", SYS_PAL, max_new_tokens=360), q, tr, plan=plan)
    cot_cache = _get_cot_by_q().get(q)
    if cot_cache is not None:
        cot_text = cot_cache["steps"][0]["text"]
        tr.add("Solver-CoT", cot_text)
        tr.calls += cot_cache["calls"]
        tr.tokens_in += cot_cache["tokens_in"]
        tr.tokens_out += cot_cache["tokens_out"]
        tr.seconds += cot_cache["llm_seconds"]
    else:
        cot_text = Agent(llm, "Solver-CoT", SYS_COT, max_new_tokens=400).act(q, tr)
    a_cot = to_number(cot_text)
    if same(a_pal, a_cot):
        tr.add("Verifier", f"hai solver đồng thuận: {a_pal} -> chấp nhận")
        return a_pal, tr
    pal_code = next((s["text"] for s in reversed(tr.steps) if s["role"] == "Solver-PAL"), "")
    critic_msg = (f"Problem: {q}\n\nSolver A (Python program) returned {a_pal}:\n{pal_code}\n\n"
                  f"Solver B (reasoning) returned {a_cot}:\n{cot_text}\n\nWhich is correct?")
    a_critic = to_number(Agent(llm, "Verifier", SYS_CRITIC, max_new_tokens=360).act(critic_msg, tr))
    a_extra = pal_once(Agent(llm, "Solver-PAL#2", SYS_PAL, max_new_tokens=360, temperature=0.7), q, tr, plan=plan,
                       retries=0)
    votes = Counter(v for v in (a_pal, a_cot, a_critic, a_extra) if v is not None)
    if not votes:
        tr.add("Coordinator", "không tác tử nào ra số -> bỏ trống")
        return None, tr
    best, n = votes.most_common(1)[0]
    if n == 1 and a_critic is not None:          # hoà phiếu: tin Verifier vì nó đã đọc cả hai lời giải
        best = a_critic
    tr.add("Coordinator", f"phiếu {dict(votes)} -> chọn {best}")
    return best, tr


SYSTEMS = {"direct": solve_direct, "cot": solve_cot, "pal": solve_pal, "multi": solve_multi}

ARITH_RE = re.compile(r"(-?\d[\d,]*\.?\d*)\s*([-+*/x×])\s*(-?\d[\d,]*\.?\d*)\s*=\s*\$?(-?\d[\d,]*\.?\d*)")


def classify_error(system, pred, gold, trace):
    """Phân loại lỗi một câu sai: no_answer | arithmetic | coordination | reasoning."""
    if pred is None:
        return "no_answer"
    if system == "multi":
        cand = [to_number(s["text"].split("=")[-1]) if s["role"] == "Tool:python" else
                to_number(s["text"]) if s["role"] in ("Solver-CoT", "Verifier") else None for s in trace.steps]
        if any(same(c, gold) for c in cand):
            return "coordination"       # đã có tác tử ra đúng nhưng cơ chế tổng hợp chọn sai
    for s in trace.steps:
        for a, op, b, c in ARITH_RE.findall(s["text"]):
            try:
                x, y, z = (float(t.replace(",", "")) for t in (a, b, c))
            except ValueError:
                continue
            r = {"+": x + y, "-": x - y, "*": x * y, "x": x * y, "×": x * y, "/": x / y if y else None}[op]
            if r is not None and abs(r - z) > 1e-2 * max(1, abs(z)):
                return "arithmetic"
    return "reasoning"


def run_question(llm, system, row):
    t0 = time.perf_counter()
    pred, tr = SYSTEMS[system](llm, row["question"])
    ok = same(pred, row["gold"])
    return {"id": row["id"], "system": system, "pred": pred, "gold": row["gold"], "correct": ok,
            "calls": tr.calls, "tokens_in": tr.tokens_in, "tokens_out": tr.tokens_out,
            "llm_seconds": round(tr.seconds, 3), "wall_seconds": round(time.perf_counter() - t0, 3),
            "error": None if ok else classify_error(system, pred, row["gold"], tr), "steps": tr.steps}
