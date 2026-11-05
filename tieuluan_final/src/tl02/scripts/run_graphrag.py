"""GraphRAG: phát hiện cộng đồng trên KG hợp nhất, tóm tắt bằng LLM, và so sánh bốn cách dựng ngữ cảnh
trên các câu hỏi đa bước / tổng hợp mà một đoạn văn đơn lẻ không chứa đủ đáp án.

Độ phủ: tỷ lệ nhóm từ khóa bắt buộc (không dấu) xuất hiện trong ngữ cảnh, và trong câu trả lời do LLM viết.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2 import chunking as C  # noqa: E402
from tl2.common import OUT, save_json, seed_all  # noqa: E402
from tl2.corpus import load_documents  # noqa: E402
from tl2.graphrag import GraphRAG, communities, merged_graph, strip_accents, summarize_communities  # noqa: E402
from tl2.kg_finance import build_finance_kg  # noqa: E402
from tl2.kg_health import build_health_kg  # noqa: E402
from tl2.llm import LocalLLM, grounded_prompt  # noqa: E402
from tl2.retrieval import (BM25Retriever, DenseRetriever, Embedder, HybridRRF, Reranker,  # noqa: E402
                           RerankPipeline)

MULTIHOP = [
    ("Fed tăng lãi suất thì giá vàng SJC trong nước chịu tác động qua những kênh nào?",
     [["dxy", "usd manh"], ["chi phi co hoi"], ["ty gia"], ["1,2057", "quy doi"]]),
    ("Tỷ giá USD/VND tăng thì FPT và HPG bị ảnh hưởng ngược nhau ra sao?",
     [["xuat khau phan mem", "doanh thu xuat khau", "ngoai te"], ["quang sat", "than"], ["lo chenh lech ty gia", "lo ty gia"]]),
    ("Đồng USD mạnh lên tác động thế nào tới khối ngoại và chỉ số VN30?",
     [["ban rong"], ["khoi ngoai"], ["von hoa lon"]]),
    ("Nghị định 232/2025 ảnh hưởng thế nào tới chênh lệch giá vàng SJC?",
     [["bai bo"], ["doc quyen"], ["chenh lech"], ["tang cung", "nguon cung"]]),
    ("NHNN hạ lãi suất có lợi cho những cổ phiếu nào trong VN30?",
     [["vcb", "vietcombank", "ngan hang"], ["vic", "vinhomes", "bat dong san"], ["hpg", "hoa phat", "xay dung"]]),
    ("Tôi đang dùng metformin và hay đi nhậu, có nguy cơ biến chứng gì?",
     [["toan lactic"], ["ha duong huyet", "ha glucose"]]),
    ("Người tiểu đường ăn thịt gà và cơm thế nào cho đúng?",
     [["uc ga"], ["ga ran", "chat beo bao hoa"], ["bat con", "dinh luong"], ["an rau truoc", "rau truoc"]]),
    ("Bệnh nhân tiểu đường bị suy tim và suy thận nên dùng thuốc nào, tránh thuốc nào?",
     [["sglt2", "empagliflozin"], ["egfr"], ["metformin"]]),
    ("Dùng gliclazide mà uống rượu thì nguy hiểm thế nào và xử trí ra sao?",
     [["ha glucose huyet", "ha duong huyet"], ["15 g", "15-15"], ["glucagon", "15 phut"]]),
    ("Tiền đái tháo đường phòng ngừa bằng những biện pháp nào và hiệu quả bao nhiêu?",
     [["58%"], ["150 phut"], ["7%", "giam can"]]),
    ("Giá quặng sắt và đầu tư công tác động thế nào tới Hòa Phát?",
     [["quang sat"], ["dau tu cong"], ["bien loi nhuan"]]),
    ("Biến chứng của tiểu đường gồm những gì và cần theo dõi định kỳ ra sao?",
     [["tim mach"], ["vong mac", "kham mat"], ["than"], ["ban chan"]]),
]


def coverage(text, groups):
    t = strip_accents(text)
    hit = [any(strip_accents(k) in t for k in g) for g in groups]
    return float(np.mean(hit)), hit


def main():
    seed_all()
    hkg, fkg = build_health_kg(), build_finance_kg()
    kgs = [hkg, fkg]
    docs = load_documents()
    fixed = C.chunk_corpus(docs, C.fixed_size, size=400, overlap=60)
    struct = C.chunk_corpus(docs, C.structure_aware, size=900)
    llm = LocalLLM()                         # nạp LLM trước embedder (xem tl2/agent.py)
    emb, rr = Embedder("e5-base", half=True), Reranker()
    naive = DenseRetriever(fixed, emb)
    adv = RerankPipeline(HybridRRF([BM25Retriever(struct), DenseRetriever(struct, emb)]), rr)

    U = merged_graph(kgs)
    comms, Q = communities(U)
    t0 = time.perf_counter()
    reports = summarize_communities(kgs, comms, llm)
    sum_s = time.perf_counter() - t0
    g = GraphRAG(kgs, struct, emb, reports)
    rows = []
    for question, groups in MULTIHOP:
        ctx = {}
        ctx["naive"] = [("RAG:" + c.doc_id, c.text) for c, _ in naive.search(question, 5)]
        ctx["advanced"] = [("RAG:" + c.doc_id, c.text) for c, _ in adv.search(question, 5)]
        loc = g.local_context(question, adv, k_chunks=3, hops=1)
        kg_ev = [("KG:" + (s or "?"), f) for f, s in loc["facts"][:14]]
        ctx["graph_local"] = kg_ev + [("RAG:" + c.doc_id, c.text) for c in loc["passages"]]
        glob = [("Cộng đồng " + str(r["id"]), r["summary"]) for r, _ in g.global_search(question, 2)]
        ctx["graph_local_global"] = ctx["graph_local"] + glob
        row = {"q": question, "entities": [e for e, _, _ in loc["entities"]]}
        for name, ev in ctx.items():
            text = " ".join(t for _, t in ev)
            cov, hits = coverage(text, groups)
            ans, ntok, sec = llm.chat(grounded_prompt(question, ev))
            acov, _ = coverage(ans, groups)
            row[name] = {"ctx_cov": cov, "ctx_hits": hits, "ctx_chars": len(text), "ans_cov": acov, "answer": ans,
                         "gen_s": sec, "n_evidence": len(ev)}
        rows.append(row)
        print(question[:40], {k: round(row[k]["ctx_cov"], 2) for k in ctx}, flush=True)
    methods = ["naive", "advanced", "graph_local", "graph_local_global"]
    agg = {m: {k: float(np.mean([r[m][k] for r in rows])) for k in ("ctx_cov", "ans_cov", "ctx_chars", "gen_s")}
           for m in methods}
    # bố cục cộng đồng để vẽ hình
    import networkx as nx
    pos = nx.spring_layout(U, seed=7, k=0.45, weight="weight", iterations=200)
    member = {n: i for i, c in enumerate(comms) for n in c}
    layout = {"nodes": [{"id": n, "label": U.nodes[n]["label"], "x": float(pos[n][0]), "y": float(pos[n][1]),
                         "comm": member[n], "kg": U.nodes[n]["kg"], "deg": U.degree(n)} for n in U.nodes],
              "edges": [[a, b] for a, b in U.edges]}
    save_json("graphrag", {"modularity": Q, "n_comms": len(comms), "summary_s": sum_s,
                           "reports": [{k: v for k, v in r.items()} for r in reports], "rows": rows, "agg": agg,
                           "layout": layout})
    (OUT / "kg").mkdir(parents=True, exist_ok=True)
    print(agg)


if __name__ == "__main__":
    main()
