"""Bảng, hộp kết quả và vết suy luận Chương 5 -> Report/TieuLuan 03/generated/ch5_*.tex

    python scripts/ch5_tables.py
Đọc outputs/metrics/ch5_kb_<miền>.json, ch5_retrieval.json, ch5_agent_<miền>.json và data/ch5/qa_<miền>.json.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd  # noqa: E402

from tl3.common import GEN, data_dir, load_json, save_json  # noqa: E402
from tl3.ch5_data import (DIAB_CSV, diabetes_doc, gold_doc, load_diabetes, load_gold, load_stock,  # noqa: E402
                          stock_doc)
from tl3.ch5_kg import diabetes_triples, gold_triples, market_triples  # noqa: E402
from tl3.texfmt import best, num, pct, table, tex_escape  # noqa: E402

DOMS = ["diabetes", "stock", "gold"]
DOM_VN = {"diabetes": "Tiểu đường", "stock": "Cổ phiếu Google", "gold": "Vàng (FINAL\\_USO)"}
RET = [("TF-IDF", "TF-IDF"), ("BM25", "BM25"), ("Dense minilm", "Dense MiniLM-L12"), ("Dense e5-small", "Dense e5-small"),
       ("Dense e5-base", "Dense e5-base"), ("Graph", "Đồ thị (KG)"), ("RRF BM25+Dense", "RRF BM25 + Dense"),
       ("RRF BM25+Dense+Graph", "RRF BM25 + Dense + KG"), ("RRF+Rerank (không đồ thị)", "RRF + Rerank"),
       ("RRF+Graph+Rerank", "RRF + KG + Rerank")]
SYS = [("llm_only", "LLM thuần (không truy xuất)"), ("naive_rag", "Naive RAG (dense top-5)"),
       ("advanced_rag", "Advanced RAG (lai + rerank)"), ("agent", "Tác tử chọn công cụ")]
TYPE_VN = {"record": "Tra cứu bản ghi", "group": "Tổng hợp nhóm", "month": "Tổng hợp tháng", "year": "Tổng hợp năm",
           "indicator": "Chỉ báo thị trường"}
STEP_COLOR = {"question": ("cSlate", "CÂU HỎI"), "thought": ("cPurple", "THOUGHT"), "action": ("cOrange", "ACTION"),
              "obs": ("cTeal", "OBSERV."), "answer": ("cGreen", "ANSWER")}


def write(name, text):
    (GEN / f"{name}.tex").write_text(text + "\n", encoding="utf-8", newline="\n")


def lines_box(lines):
    """Danh sách dòng văn bản thô -> nội dung hộp ketqua (escape TeX, giữ ngắt dòng)."""
    return "\n".join(tex_escape(s) + "\\par" for s in lines)


# ---------------------------------------------------------------- dữ liệu
def t_datasets(kb):
    src = {"diabetes": "Kaggle ghnshymsaini", "stock": "Kaggle jillanisofttech (V2)", "gold": "Kaggle sid321axn"}
    period = {"diabetes": "không có thời gian", "stock": "02/01/2015 -- 10/01/2024", "gold": "15/12/2011 -- 31/12/2018"}
    cols = {"diabetes": 9, "stock": 6, "gold": 81}
    rows = [[DOM_VN[d], src[d], period[d], num(kb[d]["n_rows"], 0), num(cols[d], 0), num(kb[d]["n_records"], 0),
             num(kb[d]["n_summaries"], 0), num(kb[d]["n_docs"], 0)] for d in DOMS]
    table("ch5_datasets", "lL{3.6cm}L{3.4cm}rrrrr", ["Bộ dữ liệu", "Nguồn", "Giai đoạn", "Số dòng", "Số cột",
                                                     "Tài liệu bản ghi", "Tài liệu tổng hợp", "Tổng tài liệu"], rows,
          note="Tiểu đường: sau khi bỏ dòng trùng lặp. Tài liệu tổng hợp: theo nhóm (tiểu đường) hoặc theo tháng, năm "
               "(thị trường).")


def t_dia_columns(df):
    meaning = {"gender": ("giới tính", "Female, Male, Other"), "age": ("tuổi (năm)", None),
               "hypertension": ("tăng huyết áp (0/1)", None), "heart_disease": ("bệnh tim (0/1)", None),
               "smoking_history": ("tiền sử hút thuốc", "never, No Info, current, former, ever, not current"),
               "bmi": ("chỉ số khối cơ thể (kg/m²)", None), "HbA1c_level": ("HbA1c (\\%)", None),
               "blood_glucose_level": ("đường huyết (mg/dL)", None), "diabetes": ("nhãn mắc tiểu đường", None)}
    rows = []
    for c, (m, cats) in meaning.items():
        s = df[c]
        if cats:
            rng = tex_escape(cats)
        else:
            rng = f"{num(float(s.min()), 2)} -- {num(float(s.max()), 2)}"
        mean = "--" if cats else num(float(s.mean()), 2)
        rows.append([f"\\texttt{{{tex_escape(c)}}}", m, rng, mean, num(int(s.nunique()), 0)])
    table("ch5_dia_columns", "lL{4.2cm}L{4.6cm}rr", ["Cột", "Ý nghĩa", "Miền giá trị", "Trung bình", "Số giá trị"], rows)


def t_dia_groups(df):
    """Tỷ lệ mắc theo khái niệm suy ra (nhóm tuổi, phân loại HbA1c) — các nút khái niệm của KG."""
    rows = []
    for col, lab in [("age_group", "Nhóm tuổi"), ("hba1c_class", "HbA1c (ADA)"), ("bmi_class", "BMI (WHO)"),
                     ("glucose_class", "Đường huyết")]:
        g = df.groupby(col)["diabetes"].agg(["size", "mean"])
        for i, (k, r) in enumerate(g.iterrows()):
            rows.append([lab if i == 0 else "", tex_escape(k), num(int(r["size"]), 0), pct(r["mean"])])
        rows.append("\\midrule")
    table("ch5_dia_groups", "llrr", ["Khái niệm", "Giá trị", "Số hồ sơ", "Tỷ lệ mắc (\\%)"], rows[:-1],
          note="HbA1c theo ADA: dưới 5,7\\% bình thường; 5,7--6,4\\% tiền tiểu đường; từ 6,5\\% ngưỡng chẩn đoán. "
               "BMI theo WHO.")


def t_stock_columns(df):
    rows = []
    for c, m in [("Date", "ngày giao dịch"), ("Open", "giá mở cửa (USD)"), ("High", "giá cao nhất (USD)"),
                 ("Low", "giá thấp nhất (USD)"), ("Close", "giá đóng cửa (USD, đã điều chỉnh chia tách)"),
                 ("Volume", "khối lượng (cổ phiếu)")]:
        s = df[c]
        if c == "Date":
            rows.append([f"\\texttt{{{c}}}", m, f"{s.min():%d/%m/%Y} -- {s.max():%d/%m/%Y}", "--"])
        else:
            rows.append([f"\\texttt{{{c}}}", m, f"{num(float(s.min()), 2)} -- {num(float(s.max()), 2)}",
                         num(float(s.mean()), 2)])
    table("ch5_stock_columns", "lL{5.4cm}L{4.6cm}r", ["Cột", "Ý nghĩa", "Miền giá trị", "Trung bình"], rows)


def t_gold_columns(df):
    groups = [("Open, High, Low, Close, Adj Close, Volume", "quỹ vàng SPDR Gold Shares (GLD)", "USD", "Close"),
              ("SP\\_*", "quỹ chỉ số S\\&P 500 (SPY)", "USD", "SP_close"),
              ("DJ\\_*", "chỉ số Dow Jones", "điểm", "DJ_close"),
              ("EG\\_*", "cổ phiếu Eldorado Gold", "USD", "EG_close"),
              ("EU\\_*", "tỷ giá EUR/USD", "", "EU_Price"),
              ("OF\\_*", "dầu Brent", "USD/thùng", "OF_Price"),
              ("OS\\_*", "dầu WTI", "USD/thùng", "OS_Price"),
              ("SF\\_*", "giá bạc (đơn vị không phải USD/ounce)", "?", "SF_Price"),
              ("USB\\_*", "lợi suất trái phiếu Mỹ", "\\%", "USB_Price"),
              ("PLT\\_*", "bạch kim", "USD/ounce", "PLT_Price"),
              ("PLD\\_*", "palađi", "USD/ounce", "PLD_Price"),
              ("RHO\\_PRICE", "rhodi (giá trị thiếu ghi là 0)", "USD/ounce", "RHO_PRICE"),
              ("USDI\\_*", "chỉ số đồng USD", "điểm", "USDI_Price"),
              ("GDX\\_*", "quỹ cổ phiếu mỏ vàng GDX", "USD", "GDX_Close"),
              ("USO\\_*", "quỹ dầu United States Oil Fund", "USD", "USO_Close")]
    last = df.iloc[-1]
    rows = []
    for cols, m, unit, c in groups:
        s = df[c]
        rows.append([f"\\texttt{{{cols}}}", m, unit, f"{num(float(s.min()), 2)} -- {num(float(s.max()), 2)}",
                     num(float(last[c]), 2), num(int((s == 0).sum()), 0)])
    table("ch5_gold_columns", "L{4.0cm}L{4.4cm}lrrr", ["Nhóm cột", "Ý nghĩa", "Đơn vị", "Miền (cột đại diện)",
                                                     "Ngày cuối", "Số ô = 0"], rows,
          note="Cột đại diện mỗi nhóm: giá đóng cửa hoặc giá. Ngày cuối: 31/12/2018. Close 121{,}25 USD khớp quỹ GLD, "
               "không phải giá vàng giao ngay (khoảng 1.280 USD/ounce cùng ngày).")


# ---------------------------------------------------------------- KB và KG
def t_kb_stats(kb):
    keys = [("n_docs", "Số tài liệu", 0), ("doc_len_mean", "Độ dài TB (âm tiết)", 1), ("doc_len_max", "Độ dài lớn nhất", 0),
            ("vocab", "Đặc trưng TF-IDF khác 0", 0), ("n_triples", "Số bộ ba", 0)]
    rows = [[lab] + [num(kb[d][k], dec) for d in DOMS] for k, lab, dec in keys]
    rows += [["Số nút KG"] + [num(kb[d]["graph"]["nodes"], 0) for d in DOMS],
             ["Số cạnh KG"] + [num(kb[d]["graph"]["edges"], 0) for d in DOMS],
             ["Thuộc tính literal trên nút"] + [num(kb[d]["graph"]["n_literals"], 0) for d in DOMS],
             ["Số loại quan hệ"] + [num(len(kb[d]["graph"]["relations"]), 0) for d in DOMS], "\\midrule",
             ["Văn bản hoá (s)"] + [num(kb[d]["corpus_s"], 2) for d in DOMS],
             ["TF-IDF (s)"] + [num(kb[d]["tfidf_s"], 2) for d in DOMS],
             ["BM25 (s)"] + [num(kb[d]["bm25_s"], 2) for d in DOMS],
             ["Dựng KG (s)"] + [num(kb[d]["build_kg_s"], 2) for d in DOMS]]
    for k, lab in [("minilm", "MiniLM-L12"), ("e5-small", "e5-small"), ("e5-base", "e5-base")]:
        rows.append([f"Embedding {lab} (s)"] + [num(kb[d].get(f"emb_{k}_s", float("nan")), 1) for d in DOMS])
    table("ch5_kb_stats", "lrrr", ["Chỉ số", "Tiểu đường", "Google", "Vàng"], rows,
          note="Thời gian đo trên CPU 4 luồng (thưa, đồ thị) và GPU RTX 8 GB (embedding). Độ dài tính theo khoảng trắng.")


def t_relations(kb):
    for d in DOMS:
        g = kb[d]["graph"]
        rel = list(g["relations"].items())
        kinds = list(g["kinds"].items())
        n = max(len(rel), len(kinds))
        rows = []
        for i in range(n):
            a = [f"\\texttt{{{tex_escape(rel[i][0])}}}", num(rel[i][1], 0)] if i < len(rel) else ["", ""]
            b = [tex_escape(kinds[i][0]), num(kinds[i][1], 0)] if i < len(kinds) else ["", ""]
            rows.append(a + b)
        table(f"ch5_rel_{d}", "lrlr", ["Quan hệ", "Số cạnh", "Loại nút", "Số nút"], rows)
        top = [[tex_escape(nd), num(deg, 0)] for nd, deg in g["top_in_degree"]]
        table(f"ch5_hub_{d}", "lr", ["Nút có bậc vào lớn nhất", "Bậc vào"], top)


def samples():
    """Hộp kết quả: tài liệu và bộ ba mẫu của mỗi miền (sinh lại bằng đúng hàm của pipeline)."""
    dia = load_diabetes()
    stock, gold = load_stock(), load_gold()
    write("ch5_doc_diabetes", lines_box([diabetes_doc(r) for r in dia.head(3).itertuples()]))
    write("ch5_doc_stock", lines_box([stock_doc(r) for r in stock[stock.Date.between("2020-03-13", "2020-03-17")]
                                     .itertuples()]))
    write("ch5_doc_gold", lines_box([gold_doc(r) for r in gold.tail(2).itertuples()]))
    fmt = lambda t: f"({t[0]}, {t[1]}, {t[2] if not isinstance(t[2], float) else round(t[2], 4)})"
    write("ch5_tri_diabetes", lines_box([fmt(t) for t in diabetes_triples(dia.head(1))]))
    st = stock[stock.Date == "2020-03-16"]
    write("ch5_tri_stock", lines_box([fmt(t) for t in market_triples(st.reset_index(drop=True), "TradingDay", "did")]))
    gd = gold.tail(1).reset_index(drop=True)
    write("ch5_tri_gold", lines_box([fmt(t) for t in gold_triples(gd)][:16]))
    return dia, stock, gold


def best_year(df, key):
    """Năm tăng mạnh nhất (đóng cửa phiên cuối so với phiên đầu năm) -- đáp án của câu hỏi mở "năm nào tăng mạnh nhất"."""
    g = df.groupby(df.Date.dt.year).Close.agg(["first", "last"])
    chg = 100 * (g["last"] / g["first"] - 1)
    return {f"{key}_best_year": int(chg.idxmax()), f"{key}_best_year_chg": float(chg.max()),
            f"{key}_worst_year": int(chg.idxmin()), f"{key}_worst_year_chg": float(chg.min())}


def t_sparql(kb):
    demo = kb["stock"]["rdf_demo"]
    rows = [[tex_escape(r[0].split("/")[-1]), num(float(r[1]), 2), num(float(r[2]), 2, sign=True)] for r in demo["sparql_rows"]]
    table("ch5_sparql", "lrr", ["Ngày (URI)", "Giá đóng cửa (USD)", "Thay đổi (\\%)"], rows)
    write("ch5_turtle", lines_box(demo["turtle_head"].splitlines()))
    write("ch5_sparql_query", lines_box(demo["sparql"].splitlines()))


# ---------------------------------------------------------------- bộ câu hỏi
def t_qa(qa):
    types = ["record", "group", "month", "year", "indicator"]
    rows = []
    for d in DOMS:
        rows.append([DOM_VN[d]] + [num(qa[d].get(t, 0), 0) for t in types] + [num(qa[d]["para"], 0), num(qa[d]["total"], 0)])
    rows.append(["Tổng"] + [num(sum(qa[d].get(t, 0) for d in DOMS), 0) for t in types] +
                [num(sum(qa[d]["para"] for d in DOMS), 0), num(sum(qa[d]["total"] for d in DOMS), 0)])
    table("ch5_qa_counts", "lrrrrrrr", ["Miền", "Bản ghi", "Nhóm", "Tháng", "Năm", "Chỉ báo", "Diễn đạt lại", "Tổng"], rows,
          note="Diễn đạt lại (para): câu hỏi dùng từ đồng nghĩa và định dạng ngày khác tài liệu, kiểm tra truy xuất ngữ nghĩa.")


def t_qa_examples(qs):
    rows = []
    for d in DOMS:
        seen = set()
        for q in qs[d]:
            key = (q["type"], q["style"])
            if key in seen:
                continue
            seen.add(key)
            ans = ("mắc" if q["answer"] else "không mắc") if q["answer_type"] == "label" else num(float(q["answer"]), 2)
            rows.append([DOM_VN[d] if len(seen) == 1 else "", TYPE_VN[q["type"]], q["style"], tex_escape(q["question"]),
                         ans, f"\\texttt{{{tex_escape(q['gold'][0])}}}"])
        rows.append("\\midrule")
    table("ch5_qa_examples", "lL{2.3cm}lL{6.6cm}rl", ["Miền", "Loại", "Kiểu", "Câu hỏi", "Đáp án", "Tài liệu đúng"], rows[:-1])


# ---------------------------------------------------------------- truy xuất
def ret_rows(agg, prefix=""):
    names = [n for n, _ in RET if f"{prefix}{n}" in agg]
    metrics = ["r@1", "r@5", "r@10", "mrr", "ndcg"]
    cols = {m: best([agg[f"{prefix}{n}"][m] for n in names], [pct(agg[f"{prefix}{n}"][m]) for n in names]) for m in metrics}
    ms = best([agg[f"{prefix}{n}"]["ms"] for n in names], [num(agg[f"{prefix}{n}"]["ms"], 1) for n in names], higher=False)
    lab = dict(RET)
    return [[lab[n]] + [cols[m][i] for m in metrics] + [ms[i]] for i, n in enumerate(names)]


HEAD = ["Bộ truy xuất", "R@1", "R@5", "R@10", "MRR", "nDCG@10", "ms / câu"]


def t_retrieval(R):
    table("ch5_ret_all", "lrrrrrr", HEAD, ret_rows(R["by_retriever"]),
          note="Trung bình trên toàn bộ câu hỏi của ba miền (\\%). In đậm: tốt nhất mỗi cột. Dense trong RRF là e5-small; "
               "rerank: bge-reranker-v2-m3 xếp lại 20 ứng viên.")
    for d in DOMS:
        table(f"ch5_ret_{d}", "lrrrrrr", HEAD, ret_rows(R["by_domain"], f"{d}|"),
              note=f"Miền {DOM_VN[d]}: {R['qa'][d]['total']} câu hỏi (\\%).")


def t_style(R):
    rows = []
    lab = dict(RET)
    for n, l in RET:
        r = [l]
        for d in DOMS:
            for s in ("lex", "para"):
                k = f"{d}|{s}|{n}"
                r.append(pct(R["by_style"][k]["mrr"]) if k in R["by_style"] else "--")
        rows.append(r)
    table("ch5_ret_style", "lrrrrrr", ["Bộ truy xuất", "TĐ khớp", "TĐ diễn đạt", "GG khớp", "GG diễn đạt", "Vàng khớp",
                                       "Vàng diễn đạt"], rows,
          note="MRR (\\%) theo kiểu câu hỏi. Khớp: dùng đúng từ và định dạng ngày như tài liệu; diễn đạt: đồng nghĩa, "
               "định dạng ngày khác. TĐ: tiểu đường; GG: Google.")
    del lab


def t_type(R):
    types = [t for t in TYPE_VN if any(k.startswith(t + "|") for k in R["by_type"])]
    rows = [[l] + [pct(R["by_type"][f"{t}|{n}"]["mrr"]) if f"{t}|{n}" in R["by_type"] else "--" for t in types]
            for n, l in RET]
    table("ch5_ret_type", "l" + "r" * len(types), ["Bộ truy xuất"] + [TYPE_VN[t] for t in types], rows,
          note="MRR (\\%) theo loại câu hỏi, gộp ba miền.")


def t_sweep(R):
    rows = []
    for d in DOMS:
        gw = [s for s in R["sweep_rows"] if s["domain"] == d and s["param"] == "graph_weight"]
        rn = [s for s in R["sweep_rows"] if s["domain"] == d and s["param"] == "rerank_n"]
        rows.append([DOM_VN[d]] + [pct(s["mrr"]) for s in gw] + [pct(s["mrr"]) for s in rn])
    gw = [s["value"] for s in R["sweep_rows"] if s["domain"] == "stock" and s["param"] == "graph_weight"]
    rn = [s["value"] for s in R["sweep_rows"] if s["domain"] == "stock" and s["param"] == "rerank_n"]
    table("ch5_ret_sweep", "l" + "r" * (len(gw) + len(rn)),
          ["Miền"] + [f"$w_G{{=}}{num(v, 1)}$" for v in gw] + [f"$n{{=}}{int(v)}$" for v in rn], rows,
          note="MRR (\\%). Trái: trọng số nhánh đồ thị trong RRF BM25 + Dense + KG. Phải: số ứng viên đưa vào cross-encoder "
               "(RRF BM25 + Dense + Rerank).")


# ---------------------------------------------------------------- tác tử
def t_agent(A):
    rows = []
    for d in DOMS:
        S = A[d]["systems"]
        accs = [S[s]["acc"] for s, _ in SYS]
        accf = best(accs, [pct(a) for a in accs])
        ms = best([S[s]["ms"] for s, _ in SYS], [num(S[s]["ms"], 0) for s, _ in SYS], higher=False)
        for i, (s, l) in enumerate(SYS):
            g = "--" if s == "llm_only" else pct(S[s]["gold_in_sources"])
            rows.append([DOM_VN[d] if i == 0 else "", l, accf[i], g, ms[i], num(S[s]["ms_median"], 0)])
        rows.append("\\midrule")
    table("ch5_agent", "lL{5.4cm}rrrr", ["Miền", "Hệ thống", "Độ chính xác (\\%)", "Nguồn đúng (\\%)", "Trễ TB (ms)",
                                         "Trễ trung vị (ms)"], rows[:-1],
          note="Nguồn đúng: tỷ lệ câu mà tài liệu đáp án có trong bằng chứng đưa cho LLM (hoặc nút KG tác tử dùng). "
               "LLM: Qwen2.5-1.5B-Instruct cục bộ, giải mã tham lam.")


def t_agent_type(A):
    rows = []
    for d in DOMS:
        S = A[d]["systems"]
        types = [t for t in TYPE_VN if f"acc_{t}" in S["agent"]]
        for j, t in enumerate(types):
            rows.append([DOM_VN[d] if j == 0 else "", TYPE_VN[t]] + [pct(S[s][f"acc_{t}"]) for s, _ in SYS])
        rows.append([DOM_VN[d] if not types else "", "Diễn đạt lại"] + [pct(S[s]["acc_para"]) for s, _ in SYS])
        rows.append("\\midrule")
    table("ch5_agent_type", "llrrrr", ["Miền", "Loại câu hỏi", "LLM thuần", "Naive RAG", "Advanced RAG", "Tác tử"], rows[:-1],
          note="Độ chính xác (\\%) theo loại câu hỏi; dòng cuối mỗi miền: riêng các câu diễn đạt lại.")


def t_tools(A):
    names = sorted({t for d in DOMS for t in A[d]["tool_use"]})
    rows = [[DOM_VN[d]] + [num(A[d]["tool_use"].get(t, 0), 0) for t in names] + [num(A[d]["n_questions"], 0)] for d in DOMS]
    table("ch5_tools", "l" + "r" * (len(names) + 1), ["Miền"] + [f"\\texttt{{{tex_escape(t)}}}" for t in names] + ["Số câu"],
          rows, note="Số câu hỏi đánh giá mà bộ định tuyến chuyển tới mỗi công cụ (công cụ đầu tiên được gọi).")


def t_answers(A, qs):
    """Bảng so sánh câu trả lời của bốn hệ thống trên vài câu hỏi mẫu."""
    for d in DOMS:
        byq = {q["qid"]: q for q in qs[d]}
        rows_by_q = {}
        for r in A[d]["rows"]:
            rows_by_q.setdefault(r["qid"], {})[r["system"]] = r
        picked = []
        for qid, rs in rows_by_q.items():
            if len(rs) == 4 and rs["agent"]["correct"] and not rs["llm_only"]["correct"]:
                picked.append(qid)
        picked = picked[:2] + [q for q, rs in rows_by_q.items() if len(rs) == 4 and not rs["naive_rag"]["correct"]
                               and q not in picked][:1]
        rows = []
        for qid in picked:
            q = byq[qid]
            ans = ("mắc" if q["answer"] else "không mắc") if q["answer_type"] == "label" else num(float(q["answer"]), 2)
            rows.append(["\\multicolumn{3}{L{15.4cm}}{\\textbf{" + tex_escape(q["question"]) + "} (đáp án: " + ans + ")}"])
            for s, l in SYS:
                r = rows_by_q[qid][s]
                a = r["answer"].replace("\n", " ")
                a = a if len(a) < 230 else a[:227] + "..."
                rows.append([l.split(" (")[0], tex_escape(a), "\\textcolor{cGreen}{đúng}" if r["correct"]
                             else "\\textcolor{cRed}{sai}"])
            rows.append("\\midrule")
        table(f"ch5_answers_{d}", "L{2.6cm}L{11.0cm}c", ["Hệ thống", "Câu trả lời (rút gọn)", "Chấm"], rows[:-1])


def traces(A):
    for d in DOMS:
        for i, t in enumerate(A[d]["traces"]):
            out = [f"\\begin{{vetsuyluan}}{{Vết suy luận của tác tử -- {DOM_VN[d]}, ví dụ {i + 1} "
                   f"({num(t['ms'], 0)} ms)}}"]
            for step in t["trace"]:
                kind, text = step[0], str(step[1])
                if kind not in STEP_COLOR:
                    continue
                if kind == "obs" and len(text) > 520:
                    text = text[:517] + "..."
                text = tex_escape(text.replace("\n", " "))
                if kind == "action" and len(step) > 2:
                    text += f" \\hfill{{\\footnotesize ({num(float(step[2]), 1)} ms)}}"
                color, lab = STEP_COLOR[kind]
                out.append(f"\\vetBuoc{{{color}}}{{{lab}}}{{{text}}}")
            out.append("\\end{vetsuyluan}")
            write(f"ch5_trace_{d}_{i + 1}", "\n".join(out))


def main():
    kb = {d: load_json(f"ch5_kb_{d}") for d in DOMS}
    dia, stock, gold = samples()
    raw = len(pd.read_csv(DIAB_CSV))
    last = gold.iloc[-1]
    save_json("ch5_facts", {
        "dia_raw": raw, "dia_rows": len(dia), "dia_dups": raw - len(dia), "dia_pos": int(dia.diabetes.sum()),
        "dia_prev": float(dia.diabetes.mean()), "dia_other": int((dia.gender == "Other").sum()),
        "dia_noinfo": float((dia.smoking_history == "No Info").mean()),
        "stock_rows": len(stock), "stock_first": f"{stock.Date.min():%d/%m/%Y}", "stock_last": f"{stock.Date.max():%d/%m/%Y}",
        "stock_min": float(stock.Close.min()), "stock_max": float(stock.Close.max()),
        "stock_maxdate": f"{stock.Date[stock.Close.idxmax()]:%d/%m/%Y}",
        "stock_mindate": f"{stock.Date[stock.Close.idxmin()]:%d/%m/%Y}",
        "stock_worst": float(stock.ret.min()), "stock_worstdate": f"{stock.Date[stock.ret.idxmin()]:%d/%m/%Y}",
        "stock_best": float(stock.ret.max()), "stock_bestdate": f"{stock.Date[stock.ret.idxmax()]:%d/%m/%Y}",
        "stock_vol": float(stock.ret.std()),
        "gold_rows": len(gold), "gold_cols": int(gold.shape[1] - 5), "gold_first": f"{gold.Date.min():%d/%m/%Y}",
        "gold_last": f"{gold.Date.max():%d/%m/%Y}", "gold_close_last": float(last.Close),
        "gold_sf_last": float(last.SF_Price), "gold_rho_zero": int((gold.RHO_PRICE == 0).sum()),
        "gold_min": float(gold.Close.min()), "gold_max": float(gold.Close.max()),
        "gold_maxdate": f"{gold.Date[gold.Close.idxmax()]:%d/%m/%Y}",
        "gold_mindate": f"{gold.Date[gold.Close.idxmin()]:%d/%m/%Y}",
        "gold_vol": float(gold.ret.std()),
        "corr_gld_gdx": float(gold.Close.pct_change().corr(gold.GDX_Close.pct_change())),
        "corr_gld_spy": float(gold.Close.pct_change().corr(gold.SP_close.pct_change())),
        "corr_gld_usdi": float(gold.Close.pct_change().corr(gold.USDI_Price.pct_change())),
        "corr_gld_uso": float(gold.Close.pct_change().corr(gold.USO_Close.pct_change())),
        **best_year(stock, "stock"), **best_year(gold, "gold"),
    })
    t_datasets(kb)
    t_dia_columns(dia)
    t_dia_groups(dia)
    t_stock_columns(stock)
    t_gold_columns(gold)
    t_kb_stats(kb)
    t_relations(kb)
    t_sparql(kb)
    qs = {d: json.loads((data_dir("ch5") / f"qa_{d}.json").read_text(encoding="utf-8")) for d in DOMS}
    try:
        R = load_json("ch5_retrieval")
    except FileNotFoundError:
        R = None
    if R:
        t_qa(R["qa"])
        t_qa_examples(qs)
        t_retrieval(R)
        t_style(R)
        t_type(R)
        t_sweep(R)
    A = {}
    for d in DOMS:
        try:
            A[d] = load_json(f"ch5_agent_{d}")
        except FileNotFoundError:
            pass
    if len(A) == 3:
        t_agent(A)
        t_agent_type(A)
        t_tools(A)
        t_answers(A, qs)
        traces(A)
    print("ch5 tables:", sorted(p.stem for p in GEN.glob("ch5_*.tex") if not p.stem.endswith("_note")).__len__())


if __name__ == "__main__":
    main()
