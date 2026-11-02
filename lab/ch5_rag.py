"""
Chương 5 — Đồ thị tri thức và RAG: đánh giá tầng truy xuất và sinh câu trả lời của KnowledgeAgent.

Hai bộ dữ liệu:
  (A) Knowledge Base tiếng Việt của hệ thống: 10 tài liệu (ADA 2024, QĐ 5481/QĐ-BYT), 36 đoạn trong pgvector,
      đồ thị tri thức 33 nút / 40 cạnh trong Apache AGE; bộ 60 câu hỏi tự xây có đoạn vàng và dữ kiện vàng.
  (B) MedQuAD — phần NIDDK (Ben Abacha & Demner-Fushman, 2019): 1.192 cặp hỏi–đáp của Viện ĐTĐ, Tiêu hóa
      và Thận Hoa Kỳ; 166 câu hỏi về ĐTĐ được dịch sang tiếng Việt để đo truy xuất chéo ngôn ngữ.

Bộ truy xuất: BM25 · dày (3 mô hình nhúng) · lai RRF · lai + GraphRAG · lai + xếp hạng lại (cross-encoder)
· truy vấn lai chạy thật trên PostgreSQL (pgvector + tsvector) như production.
Sinh câu trả lời (Gemini): chỉ LLM · RAG · GraphRAG — đo độ phủ dữ kiện, trích dẫn đúng, số liệu không có căn cứ.

    .venv/bin/python lab/ch5_rag.py              # toàn bộ
    .venv/bin/python lab/ch5_rag.py --no-llm     # bỏ phần sinh câu trả lời
"""

import argparse
import asyncio
import glob
import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from sqlalchemy import text

from common import ACCENT, DATA, GREY, PALETTE, RESULTS, ROOT, plt, r4, save_fig, save_results
from src.knowledge.entity_linking import fold, link_entities  # noqa: E402

EMBEDDERS = {
    "minilm": ("MiniLM-L12 đa ngữ (384)", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "", ""),
    "mpnet": ("MPNet đa ngữ (768)", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "", ""),
    "e5": ("multilingual-E5-large (1024)", "intfloat/multilingual-e5-large", "query: ", "passage: "),
}
RERANKER = "BAAI/bge-reranker-v2-m3-int8"
CACHE = DATA / "emb_cache"
CACHE.mkdir(exist_ok=True)


# ------------------------------------------------------------------ tiện ích
def tokenize(s: str):
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", fold(s))


class BM25:
    """Okapi BM25 (Robertson 1994): Σ IDF(t)·f(t,d)·(k1+1) / (f(t,d) + k1·(1 − b + b·|d|/avgdl))."""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avgdl = np.mean([len(t) for t in self.toks])
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def scores(self, q):
        qt = tokenize(q)
        out = np.zeros(len(self.tf))
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            norm = self.k1 * (1 - self.b + self.b * len(toks) / self.avgdl)
            out[i] = sum(self.idf.get(w, 0) * tf[w] * (self.k1 + 1) / (tf[w] + norm) for w in qt if w in tf)
        return out


def embed(key, texts, kind, tag):
    name, model, qp, pp = EMBEDDERS[key]
    path = CACHE / f"{tag}_{key}_{kind}.npy"
    if path.exists():
        return np.load(path)
    m = TextEmbedding(model_name=model)
    prefix = qp if kind == "q" else pp
    v = np.array(list(m.embed([prefix + t for t in texts], batch_size=32)), dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.save(path, v)
    return v


def rrf(rankings, k=60):
    """Reciprocal Rank Fusion (Cormack 2009): score(d) = Σ_r 1 / (k + rank_r(d))."""
    s = defaultdict(float)
    for ranking in rankings:
        for r, d in enumerate(ranking):
            s[d] += 1 / (k + r + 1)
    return sorted(s, key=lambda d: -s[d])


def metrics(ranked, gold):
    """ranked: danh sách id tài liệu đã xếp hạng; gold: tập id đúng."""
    first = next((i for i, d in enumerate(ranked[:10]) if d in gold), None)
    dcg = sum(1 / math.log2(i + 2) for i, d in enumerate(ranked[:10]) if d in gold)
    idcg = sum(1 / math.log2(i + 2) for i in range(min(len(gold), 10)))
    return {"hit1": float(first == 0), "hit3": float(first is not None and first < 3),
            "hit5": float(first is not None and first < 5),
            "recall5": len(set(ranked[:5]) & gold) / len(gold), "mrr10": 0 if first is None else 1 / (first + 1),
            "ndcg10": dcg / idcg}


def summarize(rows):
    return {k: r4(np.mean([r[k] for r in rows])) for k in rows[0]}


# ------------------------------------------------------------------ (A) KB tiếng Việt
def load_kb():
    from src.storage.db import engine
    with engine.connect() as c:
        rows = c.execute(text("""SELECT c.id, c.document_id, c.heading, c.content, d.title
                                 FROM kb_chunks c JOIN kb_documents d ON d.id = c.document_id ORDER BY c.id""")).all()
    return [dict(r._mapping) for r in rows]


def load_triples():
    return json.loads((ROOT / "data" / "kb_sources" / "curated_triples.json").read_text(encoding="utf-8"))


def graph_candidates(q, triples, chunk_of):
    """GraphRAG: nút được nhắc trong câu hỏi → các cạnh 1 bước → đoạn văn làm bằng chứng cho cạnh đó."""
    nodes = sorted({t["head"] for t in triples} | {t["tail"] for t in triples})
    ents = link_entities(q, nodes)
    cands = []
    for t in triples:
        if t["head"] in ents or t["tail"] in ents:
            cid = chunk_of.get((t["source_doc"], t["section"]))
            if cid and cid not in cands:
                cands.append(cid)
    return ents, cands


def postgres_hybrid(questions, k=10):
    """Truy vấn lai chạy trên PostgreSQL thật (SQL của src/storage/vector_store.py) — độ trễ gồm cả nhúng câu hỏi."""
    from src.knowledge.embedder import get_embedder
    from src.storage.db import session_scope
    from src.storage.vector_store import PgVectorStore
    emb = get_embedder()
    out, lat = [], []
    with session_scope() as s:
        store = PgVectorStore(s)
        for q in questions:
            t0 = time.perf_counter()
            rows = store.hybrid_search(q, emb.embed([q])[0], k=k, pool=30)
            lat.append((time.perf_counter() - t0) * 1000)
            out.append([r["id"] for r in rows])
    return out, float(np.median(lat))


def eval_vi(no_rerank=False):
    kb = load_kb()
    triples = load_triples()
    qs = json.loads((ROOT / "lab" / "kb_eval_vi.json").read_text(encoding="utf-8"))
    ids = [c["id"] for c in kb]
    passages = [f"{c['title']}. {c['heading']}. {c['content']}" for c in kb]
    chunk_of = {(c["document_id"], c["heading"]): c["id"] for c in kb}
    questions = [q["q"] for q in qs]
    gold = [set(q["gold"]) for q in qs]

    runs, latency = {}, {}
    bm = BM25(passages)
    t0 = time.perf_counter()
    runs["bm25"] = [[ids[i] for i in np.argsort(-bm.scores(q))] for q in questions]
    latency["bm25"] = (time.perf_counter() - t0) / len(qs) * 1000
    for key in EMBEDDERS:
        P = embed(key, passages, "p", "kb")
        t0 = time.perf_counter()
        Q = embed(key, questions, "q", "kbq")
        runs[key] = [[ids[i] for i in np.argsort(-(P @ q))] for q in Q]
        latency[key] = (time.perf_counter() - t0) / len(qs) * 1000
    runs["hybrid"] = [rrf([a, b]) for a, b in zip(runs["bm25"], runs["minilm"])]
    graph_info = [graph_candidates(q, triples, chunk_of) for q in questions]
    runs["graphrag"] = [rrf([h[:10], g[1]]) if g[1] else h for h, g in zip(runs["hybrid"], graph_info)]
    if not no_rerank:
        ce = TextCrossEncoder(RERANKER)
        t0 = time.perf_counter()
        rr = []
        for q, cand in zip(questions, runs["hybrid"]):
            top = cand[:15]
            sc = list(ce.rerank(q, [passages[ids.index(c)] for c in top]))
            rr.append([top[i] for i in np.argsort(sc)[::-1]] + cand[15:])
        runs["rerank"] = rr
        latency["rerank"] = (time.perf_counter() - t0) / len(qs) * 1000
    runs["postgres"], latency["postgres"] = postgres_hybrid(questions)

    res = {k: summarize([metrics(r, g) for r, g in zip(v, gold)]) for k, v in runs.items()}
    by_type = {}
    for k in runs:
        by_type[k] = {}
        for t in sorted({q["type"] for q in qs}):
            idx = [i for i, q in enumerate(qs) if q["type"] == t]
            by_type[k][t] = r4(np.mean([metrics(runs[k][i], gold[i])["hit3"] for i in idx]))
    linked = [len(g[0]) > 0 for g in graph_info]
    return {"results": res, "latency_ms": {k: round(v, 2) for k, v in latency.items()}, "by_type_hit3": by_type,
            "n_questions": len(qs), "type_counts": dict(Counter(q["type"] for q in qs)),
            "entity_link_rate": r4(np.mean(linked)), "n_chunks": len(kb),
            "examples": [{"q": questions[i], "entities": graph_info[i][0], "gold": sorted(gold[i]),
                          "bm25": runs["bm25"][i][:3], "minilm": runs["minilm"][i][:3],
                          "hybrid": runs["hybrid"][i][:3], "graphrag": runs["graphrag"][i][:3]}
                         for i in [0, 34, 56]]}, runs, kb, qs, graph_info


# ------------------------------------------------------------------ (B) MedQuAD
def chunk(textv, size, overlap=0.15):
    if size is None or len(textv) <= size:
        return [textv]
    step, out, i = int(size * (1 - overlap)), [], 0
    while i < len(textv):
        out.append(textv[i:i + size])
        i += step
    return out


def load_medquad():
    qa = []
    for f in sorted(glob.glob(str(DATA / "medquad" / "MedQuAD-master" / "5_NIDDK_QA" / "*.xml"))):
        r = ET.parse(f).getroot()
        focus = r.findtext("Focus")
        for p in r.iter("QAPair"):
            a = (p.findtext("Answer") or "").strip()
            q = p.find("Question")
            if a:
                qa.append({"focus": focus, "qtype": q.get("qtype"), "q": q.text.strip(), "a": " ".join(a.split())})
    for i, x in enumerate(qa):
        x["id"] = i
        x["diabetes"] = bool(re.search(r"diabet|insulin|glucose|hypoglyc", x["focus"] + x["q"], re.I))
    return qa


async def translate_questions(qs):
    """Dịch câu hỏi sang tiếng Việt bằng Gemini (lưu cache) — để đo truy xuất chéo ngôn ngữ Việt → Anh."""
    path = DATA / "medquad" / "diabetes_questions_vi.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    from pydantic import BaseModel
    from src.llm.client import llm

    class Out(BaseModel):
        translations: list[str]

    out = []
    for i in range(0, len(qs), 30):
        batch = qs[i:i + 30]
        for attempt in range(3):  # thử lại khi mạng / API lỗi tạm thời
            r = await llm.parse("Dịch từng câu hỏi y khoa sau sang tiếng Việt tự nhiên như người bệnh hỏi. "
                                "Giữ nguyên thứ tự và số lượng, không thêm giải thích.",
                                ["\n".join(f"{j + 1}. {q}" for j, q in enumerate(batch))], Out)
            if r and len(r.translations) == len(batch):
                break
            await asyncio.sleep(5 * (attempt + 1))
        assert r and len(r.translations) == len(batch), "dịch thiếu câu"
        out += r.translations
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def eval_medquad(no_llm=False):
    qa = load_medquad()
    sizes = {"400": 400, "800": 800, "1600": 1600, "full": None}
    out = {"n_qa": len(qa), "n_docs_focus": len({x["focus"] for x in qa}),
           "n_diabetes": sum(x["diabetes"] for x in qa), "qtype": dict(Counter(x["qtype"] for x in qa)),
           "answer_chars": {"mean": r4(np.mean([len(x["a"]) for x in qa])),
                            "median": r4(np.median([len(x["a"]) for x in qa]))}}
    questions = [x["q"] for x in qa]
    gold_of = [x["id"] for x in qa]

    def corpus(size):
        cids, texts = [], []
        for x in qa:
            for c in chunk(x["a"], size):
                cids.append(x["id"])
                texts.append(f"{x['focus']}. {c}")
        return cids, texts

    def ranked_docs(scores, cids):
        """Gộp đoạn → câu trả lời: hạng của câu trả lời = hạng của đoạn tốt nhất của nó."""
        seen, outl = set(), []
        for i in np.argsort(-scores):
            if cids[i] not in seen:
                seen.add(cids[i])
                outl.append(cids[i])
            if len(outl) >= 20:
                break
        return outl

    # 1) so sánh bộ truy xuất với đoạn 800 ký tự
    cids, texts = corpus(800)
    out["n_chunks_800"] = len(texts)
    bm = BM25(texts)
    runs = {"bm25": [ranked_docs(bm.scores(q), cids) for q in questions]}
    embs = {}
    for key in EMBEDDERS:
        P = embed(key, texts, "p", "mq800")
        Q = embed(key, questions, "q", "mqq")
        embs[key] = (P, Q)
        runs[key] = [ranked_docs(P @ q, cids) for q in Q]
    runs["hybrid"] = [rrf([a, b])[:20] for a, b in zip(runs["bm25"], runs["minilm"])]
    runs["hybrid_e5"] = [rrf([a, b])[:20] for a, b in zip(runs["bm25"], runs["e5"])]
    res = {k: summarize([metrics(r, {g}) for r, g in zip(v, gold_of)]) for k, v in runs.items()}
    dia = [i for i, x in enumerate(qa) if x["diabetes"]]
    res_dia = {k: summarize([metrics(runs[k][i], {gold_of[i]}) for i in dia]) for k in runs}
    out.update({"retrievers": res, "retrievers_diabetes": res_dia})

    # 2) kích thước đoạn (lai BM25 + MiniLM)
    chunk_res = {}
    for name, size in sizes.items():
        c2, t2 = corpus(size)
        b2 = BM25(t2)
        P = embed("minilm", t2, "p", f"mq{name}")
        Q = embed("minilm", questions, "q", "mqq")
        hy = [rrf([ranked_docs(b2.scores(q), c2), ranked_docs(P @ qv, c2)])[:20] for q, qv in zip(questions, Q)]
        chunk_res[name] = {"n_chunks": len(t2), **summarize([metrics(r, {g}) for r, g in zip(hy, gold_of)])}
        print("  đoạn", name, chunk_res[name])
    out["chunk_size"] = chunk_res

    # 3) truy xuất chéo ngôn ngữ: câu hỏi tiếng Việt → kho tiếng Anh
    if not no_llm:
        vi = asyncio.run(translate_questions([qa[i]["q"] for i in dia]))
        cross = {}
        for key in EMBEDDERS:
            P, _ = embs[key]
            Qv = embed(key, vi, "q", "mqvi")
            cross[key] = summarize([metrics(ranked_docs(P @ q, cids), {gold_of[i]}) for q, i in zip(Qv, dia)])
        cross["bm25"] = summarize([metrics(ranked_docs(bm.scores(q), cids), {gold_of[i]}) for q, i in zip(vi, dia)])
        out["cross_lingual_vi_en"] = cross
        out["cross_lingual_examples"] = [{"en": qa[i]["q"], "vi": v} for i, v in list(zip(dia, vi))[:5]]
    out["examples"] = [{k: qa[i][k] for k in ["focus", "qtype", "q"]} | {"a": qa[i]["a"][:220] + "…"}
                       for i in [0, dia[0], dia[5]]]
    return out


# ------------------------------------------------------------------ (C) sinh câu trả lời
def fact_hit(answer, fact):
    a = fold(answer).replace("–", "-")
    return any(fold(alt).replace("–", "-") in a for alt in fact.split("|"))


def numbers(s):
    return set(re.findall(r"\d+(?:[.,]\d+)?", s))


async def generate_all(qs, runs, kb, graph_info, triples):
    from src.llm.client import llm
    byid = {c["id"]: c for c in kb}
    sem = asyncio.Semaphore(4)
    system = ("Bạn là trợ lý y khoa về đái tháo đường. Trả lời ngắn gọn (≤ 4 câu) bằng tiếng Việt. ")
    rag_rule = ("CHỈ dùng thông tin trong các đoạn tài liệu được cung cấp; ghi trích dẫn dạng [1], [2] theo số đoạn. "
                "Nếu tài liệu không có thông tin thì nói không đủ dữ liệu.")

    async def one(i, mode):
        q = qs[i]["q"]
        ctx_ids, prompt = [], q
        if mode != "llm_only":
            ctx_ids = (runs["graphrag"] if mode == "graphrag" else runs["hybrid"])[i][:3]
            blocks = [f"[{j + 1}] ({byid[c]['title']} — {byid[c]['heading']}) {byid[c]['content']}"
                      for j, c in enumerate(ctx_ids)]
            facts = ""
            if mode == "graphrag" and graph_info[i][0]:
                ents = graph_info[i][0]
                rel = [f"{t['head']} —{t['relation']}→ {t['tail']}" for t in triples
                       if t["head"] in ents or t["tail"] in ents][:12]
                facts = "\nQuan hệ trong đồ thị tri thức:\n" + "\n".join(rel)
            prompt = "Tài liệu:\n" + "\n\n".join(blocks) + facts + f"\n\nCâu hỏi: {q}"
        async with sem:
            t0 = time.perf_counter()
            ans = ""
            for attempt in range(3):  # thử lại khi mạng / API lỗi tạm thời
                ans = await llm.generate(system + (rag_rule if ctx_ids else ""), [prompt]) or ""
                if ans:
                    break
                await asyncio.sleep(5 * (attempt + 1))
            lat = time.perf_counter() - t0
        ctx_text = " ".join(byid[c]["content"] for c in ctx_ids) + " " + q
        cited = [ctx_ids[int(n) - 1] for n in re.findall(r"\[(\d)\]", ans) if 0 < int(n) <= len(ctx_ids)]
        nums = numbers(ans) - {"1", "2", "3"}
        return {"i": i, "mode": mode, "answer": ans, "latency_s": round(lat, 2),
                "fact_recall": float(np.mean([fact_hit(ans, f) for f in qs[i]["facts"]])),
                "cites_gold": float(bool(set(cited) & set(qs[i]["gold"]))) if ctx_ids else None,
                "context_has_gold": float(bool(set(ctx_ids) & set(qs[i]["gold"]))) if ctx_ids else None,
                "unsupported_numbers": float(len(nums - numbers(ctx_text)) / max(len(nums), 1)) if ctx_ids else None,
                "abstain": float("không đủ dữ liệu" in ans.lower())}

    tasks = [one(i, m) for i in range(len(qs)) for m in ["llm_only", "rag", "graphrag"]]
    rows = await asyncio.gather(*tasks)
    summary = {}
    for m in ["llm_only", "rag", "graphrag"]:
        rs = [r for r in rows if r["mode"] == m and r["answer"]]
        summary[m] = {"n": len(rs), "fact_recall": r4(np.mean([r["fact_recall"] for r in rs])),
                      "latency_s_median": r4(np.median([r["latency_s"] for r in rs])),
                      "answer_chars": r4(np.mean([len(r["answer"]) for r in rs])),
                      "abstain_rate": r4(np.mean([r["abstain"] for r in rs]))}
        if m != "llm_only":
            for k in ["cites_gold", "context_has_gold", "unsupported_numbers"]:
                summary[m][k] = r4(np.mean([r[k] for r in rs]))
    return summary, rows


# ------------------------------------------------------------------ hình
def plot_kg(triples):
    G = nx.DiGraph()
    for t in triples:
        G.add_node(t["head"], label=t["head_label"])
        G.add_node(t["tail"], label=t["tail_label"])
        G.add_edge(t["head"], t["tail"], rel=t["relation"])
    colors = {"Disease": PALETTE[3], "RiskFactor": PALETTE[1], "Test": PALETTE[0], "Drug": PALETTE[4],
              "Complication": PALETTE[6], "Intervention": PALETTE[2]}
    pos = nx.kamada_kawai_layout(G.to_undirected())
    fig, ax = plt.subplots(figsize=(11, 7.5))
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#9ca3af", arrows=True, arrowsize=8, width=0.8,
                           connectionstyle="arc3,rad=0.05")
    for lab, c in colors.items():
        nodes = [n for n, d in G.nodes(data=True) if d["label"] == lab]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, node_color=c, node_size=[220 + 90 * G.degree(n) for n in nodes],
                               ax=ax, label=lab, alpha=0.9)
    nx.draw_networkx_labels(G, pos, font_size=7.5, ax=ax, font_family="serif")
    ax.legend(scatterpoints=1, markerscale=0.5, fontsize=8, loc="lower left")
    ax.set_title(f"Đồ thị tri thức ĐTĐ: {G.number_of_nodes()} nút, {G.number_of_edges()} cạnh, 7 loại quan hệ")
    ax.axis("off")
    save_fig(fig, "ch5_kg")
    return {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
            "node_types": dict(Counter(nx.get_node_attributes(G, "label").values())),
            "relations": dict(Counter(t["relation"] for t in triples)),
            "top_degree": sorted(((n, G.degree(n)) for n in G), key=lambda x: -x[1])[:6]}


def plot_kb_stats(kb, mq):
    fig, axes = plt.subplots(1, 3, figsize=(11, 2.9))
    axes[0].hist([len(c["content"]) for c in kb], bins=12, color=ACCENT)
    axes[0].set(title=f"KB tiếng Việt: độ dài {len(kb)} đoạn (ký tự)")
    qa = load_medquad()
    axes[1].hist([min(len(x["a"]), 8000) for x in qa], bins=40, color=GREY)
    axes[1].set(title="MedQuAD-NIDDK: độ dài câu trả lời", xlabel="ký tự (cắt ở 8.000)")
    qt = Counter(x["qtype"] for x in qa).most_common(8)
    axes[2].barh([k for k, _ in qt][::-1], [v for _, v in qt][::-1], color=PALETTE[2])
    axes[2].set(title="MedQuAD-NIDDK: loại câu hỏi")
    fig.tight_layout()
    save_fig(fig, "ch5_data")


def plot_retrieval(vi, mq):
    names = {"bm25": "BM25", "minilm": "Dày MiniLM", "mpnet": "Dày MPNet", "e5": "Dày E5-large",
             "hybrid": "Lai RRF", "graphrag": "Lai + GraphRAG", "rerank": "Lai + rerank", "postgres": "Lai (Postgres)"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.3))
    keys = [k for k in names if k in vi["results"]]
    x = np.arange(len(keys))
    for j, (m, lab) in enumerate([("hit1", "Hit@1"), ("hit3", "Hit@3"), ("mrr10", "MRR@10")]):
        axes[0].bar(x + (j - 1) * 0.27, [vi["results"][k][m] for k in keys], 0.27, color=PALETTE[j], label=lab)
    axes[0].set_xticks(x, [names[k] for k in keys], rotation=30, ha="right", fontsize=8)
    axes[0].set(title="KB tiếng Việt — 60 câu hỏi", ylim=(0, 1.05))
    axes[0].legend(fontsize=8, ncol=3)
    keys2 = ["bm25", "minilm", "mpnet", "e5", "hybrid", "hybrid_e5"]
    lab2 = ["BM25", "MiniLM", "MPNet", "E5-large", "Lai (MiniLM)", "Lai (E5)"]
    x = np.arange(len(keys2))
    for j, (m, lab) in enumerate([("hit1", "Hit@1"), ("hit5", "Hit@5"), ("mrr10", "MRR@10")]):
        axes[1].bar(x + (j - 1) * 0.27, [mq["retrievers"][k][m] for k in keys2], 0.27, color=PALETTE[j], label=lab)
    axes[1].set_xticks(x, lab2, rotation=30, ha="right", fontsize=8)
    axes[1].set(title=f"MedQuAD-NIDDK — {mq['n_qa']} câu hỏi tiếng Anh", ylim=(0, 1.05))
    fig.tight_layout()
    save_fig(fig, "ch5_retrieval")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()
    triples = load_triples()
    kg = plot_kg(triples)
    print("KG:", kg)
    print("(A) KB tiếng Việt")
    vi, runs, kb, qs, graph_info = eval_vi()
    for k, v in vi["results"].items():
        print(f"  {k:10s} {v}  {vi['latency_ms'].get(k, '')} ms")
    print("(B) MedQuAD")
    mq = eval_medquad(args.no_llm)
    for k, v in mq["retrievers"].items():
        print(f"  {k:10s} {v}")
    print("  chéo ngôn ngữ:", mq.get("cross_lingual_vi_en"))
    plot_kb_stats(kb, mq)
    plot_retrieval(vi, mq)
    payload = {"kg": kg, "kb_vi": vi, "medquad": mq,
               "kb_sample": [{k: c[k] for k in ["id", "document_id", "heading"]} | {"content": c["content"][:160]}
                             for c in kb[:3]],
               "questions_sample": qs[:6], "triples_sample": triples[:4]}
    if not args.no_llm:
        print("(C) Sinh câu trả lời")
        gen, rows = asyncio.run(generate_all(qs, runs, kb, graph_info, triples))
        print(gen)
        payload["generation"] = gen
        (RESULTS / "ch5_answers.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    save_results("ch5", payload)


if __name__ == "__main__":
    main()
