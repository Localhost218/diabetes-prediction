"""Chương 7 (mạch tiểu đường) — hệ đa tác tử trả lời câu hỏi trắc nghiệm y khoa về đái tháo đường.

Dữ liệu: MedMCQA (Pal và cộng sự, CHIL 2022) — đề thi tuyển sinh sau đại học y khoa Ấn Độ, lọc 1.421 câu một đáp án
có chủ đề tiểu đường/insulin/HbA1c/hạ đường huyết/metformin; lấy mẫu cố định n câu. Kho tri thức cho tác tử truy xuất:
MedQuAD — phần NIDDK (Ben Abacha & Demner-Fushman, 2019), 1.192 cặp hỏi–đáp, chỉ mục BM25.

Bốn hệ, cùng một LLM cục bộ (Qwen2.5-1.5B-Instruct, giải mã tham lam):
  direct : LLM chọn ngay một chữ cái A–D
  cot    : một tác tử suy luận từng bước rồi chọn
  rag    : tác tử dùng công cụ tra cứu (BM25 trên MedQuAD) rồi trả lời dựa trên bằng chứng
  multi  : Solver-CoT ‖ Solver-RAG -> Verifier; đồng thuận thì chấp nhận, bất đồng thì Critic đọc cả hai lập luận
           và bằng chứng để phân xử, thêm một mẫu CoT nhiệt độ 0,7, rồi biểu quyết đa số (hoà phiếu tin Critic)
Khung Agent/Trace kế thừa tl3.ch7_agents (thành viên nhóm 07); bài toán, dữ liệu, công cụ và giao thức là của tiểu luận final.
"""
import glob
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd

from tl3.ch7_agents import Agent, Trace

ROOT = Path(__file__).resolve().parents[3]
TLF = ROOT / "tieuluan_final"
MCQ = TLF / "data" / "ch7" / "medmcqa_diabetes.parquet"
MEDQUAD = ROOT / "lab" / "data" / "medquad" / "MedQuAD-master" / "5_NIDDK_QA"
LETTERS = "ABCD"
ON_TOPIC = re.compile(r"diabet|insulin|HbA1c|glycated|hypoglyc|metformin|sulfonylurea|ketoacidosis", re.I)


def load_questions(n=150, seed=42):
    df = pd.read_parquet(MCQ)
    df = df[df.question.str.len().between(20, 600)]
    df = df.sample(n=n, random_state=seed).reset_index(drop=True)
    rows = []
    for i, r in df.iterrows():
        rows.append({"id": int(i), "mid": r["id"], "question": r["question"].strip(),
                     "options": [r.opa, r.opb, r.opc, r.opd], "gold": LETTERS[int(r.cop)],
                     "subject": r.subject_name, "exp": r.exp})
    return rows


def format_q(row):
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(row["options"]))
    return f"Question: {row['question']}\n{opts}"


ANS_RE = re.compile(r"(?:answer|đáp án)\s*(?:is|:)?\s*\(?\**([ABCD])\b", re.I)


def to_letter(text):
    """Lấy chữ cái đáp án: ưu tiên 'Answer: X', sau đó chữ cái đứng riêng cuối cùng."""
    if not text:
        return None
    m = ANS_RE.findall(text)
    if m:
        return m[-1].upper()
    m = re.findall(r"\b([ABCD])\b", text)
    return m[-1] if m else None


# ---------------------------------------------------------------- công cụ: tra cứu MedQuAD-NIDDK (BM25)
class KnowledgeTool:
    def __init__(self):
        from rank_bm25 import BM25Okapi
        docs = []
        for f in sorted(glob.glob(str(MEDQUAD / "*.xml"))):
            r = ET.parse(f).getroot()
            focus = r.findtext("Focus") or ""
            for p in r.iter("QAPair"):
                a = " ".join((p.findtext("Answer") or "").split())
                q = (p.findtext("Question") or "").strip()
                if a:
                    docs.append({"title": f"{focus} — {q}", "text": a})
        self.docs = docs
        self.bm25 = BM25Okapi([self._tok(d["title"] + " " + d["text"]) for d in docs])

    @staticmethod
    def _tok(s):
        return re.findall(r"[a-z0-9]+", s.lower())

    def search(self, query, k=3, max_chars=700):
        scores = self.bm25.get_scores(self._tok(query))
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [(self.docs[i]["title"], self.docs[i]["text"][:max_chars], float(scores[i])) for i in top]


SYS_DIRECT = ("You are a medical exam assistant. Reply with only the letter of the single best option "
              "in the form 'Answer: X'.")
SYS_COT = ("You are a careful physician answering a multiple-choice exam question about diabetes and metabolism. "
           "Think step by step in at most 6 short sentences, then end with 'Answer: X' where X is A, B, C or D.")
SYS_RAG = ("You are a physician. You are given reference passages from NIH/NIDDK. Use them when relevant, "
           "otherwise rely on standard medical knowledge. Reason briefly (at most 5 sentences), cite passages as [1], [2], "
           "then end with 'Answer: X'.")
SYS_CRITIC = ("You are a senior endocrinologist reviewing two colleagues who disagree on an exam question. "
              "Check each argument against the evidence and medical facts, point out the error, then end with "
              "'Answer: X' for the correct option.")


def evidence_block(tool, row, trace, k=3):
    query = row["question"] + " " + " ".join(row["options"])
    hits = tool.search(query, k=k)
    trace.add("Tool:search", "; ".join(f"[{i + 1}] {t[:80]} (BM25 {s:.1f})" for i, (t, _, s) in enumerate(hits)))
    return "\n".join(f"[{i + 1}] {t}\n{x}" for i, (t, x, _) in enumerate(hits))


def solve_direct(llm, row, tool=None):
    tr = Trace()
    return to_letter(Agent(llm, "Direct", SYS_DIRECT, max_new_tokens=8).act(format_q(row), tr)), tr


def solve_cot(llm, row, tool=None):
    tr = Trace()
    return to_letter(Agent(llm, "CoT", SYS_COT, max_new_tokens=320).act(format_q(row), tr)), tr


def solve_rag(llm, row, tool):
    tr = Trace()
    ev = evidence_block(tool, row, tr)
    out = Agent(llm, "RAG", SYS_RAG, max_new_tokens=320).act(f"Reference passages:\n{ev}\n\n{format_q(row)}", tr)
    return to_letter(out), tr


_CACHE_COT_RAG = {}


def _get_cot_rag_cache():
    if not _CACHE_COT_RAG:
        import json
        for tag in ("cot", "rag"):
            p = TLF / "outputs" / "ch7" / f"rows_{tag}.jsonl"
            if p.exists():
                _CACHE_COT_RAG[tag] = {json.loads(l)["id"]: json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()}
    return _CACHE_COT_RAG.get("cot", {}), _CACHE_COT_RAG.get("rag", {})


def solve_multi(llm, row, tool):
    tr = Trace()
    q = format_q(row)
    qid = row.get("id", -1)
    cot_map, rag_map = _get_cot_rag_cache() if qid >= 0 else ({}, {})
    if qid in cot_map and qid in rag_map:
        rc, rr = cot_map[qid], rag_map[qid]
        cot_text = rc["steps"][0]["text"]
        tr.add("Solver-CoT", cot_text)
        a_cot = to_letter(cot_text)
        ev = evidence_block(tool, row, tr)
        rag_text = rr["steps"][1]["text"]
        tr.add("Solver-RAG", rag_text)
        a_rag = to_letter(rag_text)
        tr.calls += rc["calls"] + rr["calls"]
        tr.tokens_in += rc["tokens_in"] + rr["tokens_in"]
        tr.tokens_out += rc["tokens_out"] + rr["tokens_out"]
        tr.seconds += rc["llm_seconds"] + rr["llm_seconds"]
    else:
        cot_text = Agent(llm, "Solver-CoT", SYS_COT, max_new_tokens=320).act(q, tr)
        a_cot = to_letter(cot_text)
        ev = evidence_block(tool, row, tr)
        rag_text = Agent(llm, "Solver-RAG", SYS_RAG, max_new_tokens=320).act(f"Reference passages:\n{ev}\n\n{q}", tr)
        a_rag = to_letter(rag_text)
    if a_cot and a_cot == a_rag:
        tr.add("Verifier", f"hai solver đồng thuận: {a_cot} -> chấp nhận")
        return a_cot, tr
    critic_msg = (f"{q}\n\nReference passages:\n{ev}\n\nColleague 1 chose {a_cot}:\n{cot_text}\n\n"
                  f"Colleague 2 chose {a_rag}:\n{rag_text}\n\nWho is right?")
    a_critic = to_letter(Agent(llm, "Critic", SYS_CRITIC, max_new_tokens=320).act(critic_msg, tr))
    a_extra = to_letter(Agent(llm, "Solver-CoT#2", SYS_COT, max_new_tokens=320, temperature=0.7).act(q, tr))
    votes = Counter(v for v in (a_cot, a_rag, a_critic, a_extra) if v)
    if not votes:
        tr.add("Coordinator", "không tác tử nào chọn đáp án -> bỏ trống")
        return None, tr
    best, n = votes.most_common(1)[0]
    if n <= 1 or (n == 2 and len(votes) == 2 and a_critic):   # hoà phiếu: tin Critic vì đã đọc cả hai lập luận
        best = a_critic or best
    tr.add("Coordinator", f"phiếu {dict(votes)} -> chọn {best}")
    return best, tr


SYSTEMS = {"direct": solve_direct, "cot": solve_cot, "rag": solve_rag, "multi": solve_multi}


def classify_error(system, pred, gold, trace):
    if pred is None:
        return "no_answer"
    if system == "multi":
        picks = [to_letter(s["text"]) for s in trace.steps if s["role"] in ("Solver-CoT", "Solver-RAG", "Critic")]
        if gold in picks:
            return "coordination"        # đã có tác tử chọn đúng nhưng cơ chế tổng hợp chọn sai
    return "knowledge"


def run_question(llm, system, row, tool):
    t0 = time.perf_counter()
    pred, tr = SYSTEMS[system](llm, row, tool)
    ok = pred == row["gold"]
    wall = max(time.perf_counter() - t0, tr.seconds + 0.02)
    return {"id": row["id"], "mid": row["mid"], "system": system, "pred": pred, "gold": row["gold"], "correct": ok,
            "subject": row["subject"], "calls": tr.calls, "tokens_in": tr.tokens_in, "tokens_out": tr.tokens_out,
            "llm_seconds": round(tr.seconds, 3), "wall_seconds": round(wall, 3),
            "error": None if ok else classify_error(system, pred, row["gold"], tr), "steps": tr.steps}
