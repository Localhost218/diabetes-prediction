"""Bảng (generated/ch7_*.tex) và vết suy luận mẫu (generated/ch7_trace_*.tex) của Chương 7."""
import json
import re
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.common import GEN, LOGS, load_json  # noqa: E402
from tl3.texfmt import best, num, pct, table, tex_escape  # noqa: E402
from tl3.ch7_agents import load_gsm8k  # noqa: E402

SYS = ["direct", "cot", "pal", "multi"]
NAMES = {"direct": "Trực tiếp", "cot": "CoT", "pal": "PAL + Python", "multi": "Đa tác tử"}
ERRS = [("no_answer", "Không ra số"), ("arithmetic", "Sai số học"), ("reasoning", "Sai lập luận"),
        ("coordination", "Lỗi phối hợp")]
STEP = {"Direct": ("cSlate", "TRẢ LỜI"), "CoT": ("cBlue", "CoT"), "PAL": ("cGreen", "PAL"),
        "Planner": ("cIndigo", "PLANNER"), "Solver-PAL": ("cGreen", "SOLVER-PAL"),
        "Solver-PAL#2": ("cTeal", "PAL\\#2"), "Solver-CoT": ("cBlue", "SOLVER-CoT"),
        "Tool:python": ("cOrange", "PYTHON"), "Verifier": ("cRed", "VERIFIER"),
        "Coordinator": ("cPurple", "ĐIỀU PHỐI")}
TRACE_TITLE = {"agree_ok": "hai Solver đồng thuận, Verifier chấp nhận ngay",
               "disagree_ok": "hai Solver bất đồng, Verifier và biểu quyết chọn đúng (PAL đơn lẻ sai)",
               "coordination": "lỗi phối hợp -- một tác tử đã ra đáp án đúng nhưng bị biểu quyết loại",
               "pal_fix": "PAL tự sửa sau khi mã chạy lỗi trong sandbox"}


def t_main(R):
    S = R["systems"]
    acc = [S[s]["acc"] for s in SYS]
    rows = []
    for s, a in zip(SYS, best(acc, [pct(S[s]["acc"], 1) for s in SYS])):
        rows.append([NAMES[s], a, f"[{pct(S[s]['ci_lo'], 1)}; {pct(S[s]['ci_hi'], 1)}]", num(S[s]["calls"], 2),
                     num(S[s]["tokens_in"], 0), num(S[s]["tokens_out"], 0), num(S[s]["seconds_median"], 1),
                     num(S[s]["seconds"], 1)])
    table("ch7_main", "lrcrrrrr", ["Hệ", "Acc (\\%)", "KTC 95 \\%", "Lượt gọi", "Token vào", "Token ra",
                                   "Giây (trung vị)", "Giây (TB)"], rows,
          note=f"Trung bình trên cùng {R['n']} câu GSM8K test (mẫu ngẫu nhiên, hạt giống 42), Qwen2.5-1.5B-Instruct "
               "chạy cục bộ, giải mã tham lam. KTC: khoảng tin cậy bootstrap 2.000 lần lấy mẫu lại.")


def t_errors(R):
    S = R["systems"]
    rows = [[NAMES[s]] + [str(S[s]["errors"][e]) for e, _ in ERRS] + [str(R["n"] - S[s]["correct"])] for s in SYS]
    table("ch7_errors", "lrrrrr", ["Hệ"] + [l for _, l in ERRS] + ["Tổng sai"], rows,
          note="Sai số học: lời giải có một phép tính viết sai kết quả; lỗi phối hợp (chỉ hệ đa tác tử): ít nhất một "
               "tác tử đã ra đáp án đúng nhưng bước tổng hợp chọn sai; còn lại là sai lập luận.")


def t_vs(R):
    rows = []
    for ref in ("pal", "cot"):
        v = R[f"vs_{ref}"]
        cpe = num(v["calls_per_extra_correct"], 1) if v["calls_per_extra_correct"] else "--"
        rows.append([f"Đa tác tử so với {NAMES[ref]}", str(v["fixed"]), str(v["broken"]), num(v["net"], 0, sign=True),
                     num(100 * v["gain"], 1, sign=True), cpe, num(v["seconds_ratio"], 2)])
    table("ch7_vs", "lrrrrrr", ["So sánh", "Sửa đúng", "Làm hỏng", "Ròng", "Acc (điểm \\%)", "Lượt gọi / câu đúng thêm",
                                "Thời gian (lần)"], rows,
          note="Sửa đúng: câu hệ tham chiếu sai nhưng đa tác tử đúng; làm hỏng: ngược lại. Lượt gọi / câu đúng thêm: "
               "số lượt gọi LLM phụ trội trên toàn tập chia cho số câu đúng tăng thêm.")


def t_difficulty(R):
    rows = []
    for r in R["difficulty"]["rows"]:
        vals = [r[s] for s in SYS]
        rows.append([f"{r['bin']} phép", str(r["n"])] + best(vals, [pct(v, 1) for v in vals]))
    table("ch7_difficulty", "lr" + "r" * 4, ["Độ khó", "Số câu"] + [NAMES[s] for s in SYS], rows,
          note="Độ khó xấp xỉ bằng số phép tính đánh dấu <<...>> trong lời giải mẫu của GSM8K; accuracy tính theo \\%.")


def t_answers(ids):
    qs = {q["id"]: q for q in load_gsm8k(250)}
    rows_by = {s: {r["id"]: r for r in map(json.loads, (LOGS / f"ch7_rows_{s}.jsonl").read_text(
        encoding="utf-8").splitlines())} for s in SYS}
    picked = [i for i in ids if len(qs[i]["question"]) < 300 and len({rows_by[s][i]["correct"] for s in SYS}) == 2][:4]
    rows = []
    for i in picked:
        rows.append(["\\multicolumn{3}{L{15.0cm}}{\\textbf{Câu " + str(i) + ".} " + tex_escape(qs[i]["question"])
                     + " \\emph{(đáp án: " + tex_escape(qs[i]["gold"]) + ")}}"])
        for s in SYS:
            r = rows_by[s][i]
            rows.append([NAMES[s], tex_escape("không ra số" if r["pred"] is None else r["pred"]),
                         "\\textcolor{cGreen}{đúng}" if r["correct"] else "\\textcolor{cRed}{sai}"])
        rows.append("\\midrule")
    table("ch7_answers", "L{4.0cm}L{8.0cm}c", ["Hệ", "Đáp án đưa ra", "Chấm"], rows[:-1])


LATEX_SUBS = [(r"\\(?:text|boxed|mathrm|textbf)\{([^{}]*)\}", r"\1"), (r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)"),
              (r"\\[()\[\]]", ""), (r"\\times", "×"), (r"\\cdot", "·"), (r"\\div", "÷"), (r"\\approx", "≈"),
              (r"\\(?:quad|,|;|!)", " "), (r"\\%", "%"), (r"\\\$", "$")]


def strip_latex(text):
    """Bỏ cú pháp LaTeX mà LLM hay chèn vào lời giải (\\( \\), \\times, \\frac{a}{b}...) để vết đọc được như văn bản."""
    for pat, rep in LATEX_SUBS:
        text = re.sub(pat, rep, text)
    return text


def fmt_step(role, text):
    """Mã Python giữ xuống dòng (\\texttt); lời văn gộp một đoạn và cắt bớt khi quá dài."""
    text = strip_latex(text.strip())
    if "def solution" in text or role == "Tool:python" and "\n" in text:
        lines = [l for l in text.splitlines() if not l.strip().startswith("```") and l.strip()][:22]
        # dòng dài (thường là lời văn mô hình chèn quanh mã) phải tự cắt: "~" không cho LaTeX ngắt dòng
        lines = [w for l in lines for w in textwrap.wrap(l, 74,subsequent_indent=" " * (len(l) - len(l.lstrip()) + 4),
                                                           drop_whitespace=False) or [l]]
        body = "\\\\\n".join("\\texttt{\\footnotesize " + tex_escape(l).replace(" ", "~") + "}" for l in lines)
        return body
    # lời văn: bỏ ký hiệu markdown (**đậm**, ### tiêu đề, rào ```) để hộp vết đọc như văn bản thường
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?m)^\s*#{1,6}\s*", "", text)
    text = re.sub(r"```[a-z]*", "", text)
    text = " ".join(text.split())
    if len(text) > 620:
        text = text[:600] + " ... [lược bớt]"
    return tex_escape(text)


def traces(T):
    for key, t in T["traces"].items():
        out = [f"\\begin{{vetsuyluan}}{{Vết tác tử -- {TRACE_TITLE[key]} (câu {t['id']}, đáp án {tex_escape(t['gold'])}, "
               f"hệ trả lời {tex_escape(t['pred'])}, {t['calls']} lượt gọi, {num(t['wall_seconds'], 1)} s)}}",
               "\\vetBuoc{cNavy}{CÂU HỎI}{" + tex_escape(t["question"]) + "}"]
        for st in t["steps"]:
            color, lab = STEP.get(st["role"], ("cSlate", tex_escape(st["role"]).upper()))
            out.append(f"\\vetBuoc{{{color}}}{{{lab}}}{{{fmt_step(st['role'], st['text'])}}}")
        out.append("\\end{vetsuyluan}")
        (GEN / f"ch7_trace_{key}.tex").write_text("\n".join(out) + "\n", encoding="utf-8", newline="\n")
        print("->", f"ch7_trace_{key}.tex")


def main():
    R = load_json("ch7_gsm8k")
    t_main(R)
    t_errors(R)
    t_vs(R)
    t_difficulty(R)
    rows_multi = [json.loads(l) for l in (LOGS / "ch7_rows_multi.jsonl").read_text(encoding="utf-8").splitlines()]
    common = sorted(set(r["id"] for r in rows_multi))
    t_answers(common)
    traces(load_json("ch7_traces"))


if __name__ == "__main__":
    main()
