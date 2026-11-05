"""Web App của Tiểu luận 02: tác tử ReAct y tế và tài chính, khám phá đồ thị tri thức, so sánh bộ truy xuất RAG.

Chạy:  python app/app.py   rồi mở http://127.0.0.1:8092/
Mọi mô hình (MLP, BiLSTM, e5, bge-reranker, Qwen2.5) chạy cục bộ; không có lời gọi tới dịch vụ ngoài.
"""
import sys
import time
from pathlib import Path

import networkx as nx
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2.agent import ReActAgent  # noqa: E402
from tl2.retrieval import BM25Retriever, DenseRetriever, Embedder  # noqa: E402

app = Flask(__name__)
AGENT = ReActAgent(use_llm="--no-llm" not in sys.argv)
CH = AGENT.chunks
RETRIEVERS = {"BM25": BM25Retriever(CH), "Dense e5-base": DenseRetriever(CH, Embedder("e5-base", half=True)),
              "Hybrid + Rerank": AGENT.rag}
LAYOUT = {}


def kg_payload(name):
    """Đồ thị kèm toạ độ tính sẵn (Kamada-Kawai) để trình duyệt vẽ SVG mà không cần thư viện ngoài."""
    if name in LAYOUT:
        return LAYOUT[name]
    kg = AGENT.hkg if name == "health" else AGENT.fkg
    U = nx.Graph(kg.G.to_undirected())
    pos = nx.kamada_kawai_layout(U)
    xs, ys = [p[0] for p in pos.values()], [p[1] for p in pos.values()]
    sx = lambda v: 40 + 920 * (v - min(xs)) / (max(xs) - min(xs))  # noqa: E731
    sy = lambda v: 30 + 540 * (v - min(ys)) / (max(ys) - min(ys))  # noqa: E731
    LAYOUT[name] = {
        "nodes": [{"id": n, "label": d["label"], "type": d["type"], "x": sx(pos[n][0]), "y": sy(pos[n][1]),
                   "deg": U.degree(n)} for n, d in kg.G.nodes(data=True)],
        "edges": [{"s": s, "o": o, "p": p, "label": kg.onto.predicates[p].label, "source": d.get("source"),
                   "text": kg.verbalize((s, p, o, d))} for s, p, o, d in kg.facts()],
        "stats": {k: v for k, v in kg.stats().items() if not isinstance(v, dict)}}
    return LAYOUT[name]


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/agent")
def api_agent():
    body = request.get_json(force=True)
    profile = {k: body[k] for k in ("HbA1c_level", "blood_glucose_level", "bmi", "age", "gender") if body.get(k)}
    r = AGENT.run(body["question"], profile or None)
    return jsonify({k: r[k] for k in ("question", "domain", "trace", "evidence", "answer", "verification",
                                       "total_ms", "profile")} | {"observations": _jsonable(r["observations"])})


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o if isinstance(o, (int, float, str, bool)) or o is None else str(o)


@app.get("/api/kg/<name>")
def api_kg(name):
    return jsonify(kg_payload(name))


@app.post("/api/rag")
def api_rag():
    q = request.get_json(force=True)["q"]
    out = {}
    for name, r in RETRIEVERS.items():
        t0 = time.perf_counter()
        hits = r.search(q, 3)
        out[name] = {"ms": 1000 * (time.perf_counter() - t0),
                     "hits": [{"doc": c.doc_id, "heading": c.heading, "score": s, "text": c.text[:420]} for c, s in hits]}
    return jsonify(out)


@app.get("/api/series/<sym>")
def api_series(sym):
    df = AGENT.gold if sym == "GOLD_VND" else AGENT.market
    s = df[sym].iloc[-250:]
    return jsonify({"date": [d.strftime("%Y-%m-%d") for d in s.index], "v": s.round(3).tolist()})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8092, debug=False)
