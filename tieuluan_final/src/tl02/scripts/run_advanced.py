"""Advanced RAG, tối ưu tiền truy xuất bằng LLM cục bộ: HyDE và Multi-Query, so với truy xuất dày gốc.

HyDE: LLM viết một đoạn trả lời giả định, đem embedding của đoạn đó đi tìm (chuyển từ không gian "hỏi" sang "đáp").
Multi-Query: LLM viết 3 cách hỏi khác, truy xuất từng câu rồi hợp nhất bằng RRF.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_retrieval import load_qa  # noqa: E402
from tl2 import chunking as C  # noqa: E402
from tl2.common import save_json, seed_all  # noqa: E402
from tl2.corpus import load_documents  # noqa: E402
from tl2.llm import LocalLLM  # noqa: E402
from tl2.retrieval import DenseRetriever, Embedder, is_relevant, rank_metrics  # noqa: E402

SYS = "Bạn là chuyên gia y tế và tài chính Việt Nam, trả lời ngắn gọn bằng tiếng Việt."


def rrf(score_lists, k=60):
    fused = np.zeros(len(score_lists[0]))
    for s in score_lists:
        rank = np.empty(len(s), dtype=int)
        rank[np.argsort(-s)] = np.arange(1, len(s) + 1)
        fused += 1.0 / (k + rank)
    return fused


def main():
    seed_all()
    docs, qa = load_documents(), load_qa()
    chunks = C.chunk_corpus(docs, C.structure_aware, size=900)
    llm = LocalLLM()                         # nạp LLM trước embedder (xem tl2/agent.py)
    emb = Embedder("e5-base")
    dense = DenseRetriever(chunks, emb)
    n_rel = {q["id"]: sum(is_relevant(c, q["a"]) for c in chunks) for q in qa}
    res = {m: [] for m in ("dense", "hyde", "hyde_plus_q", "multiquery")}
    examples, gen_s = [], {"hyde": [], "multiquery": []}
    for q in qa:
        base = dense.scores(q["q"])
        hyp, _, s1 = llm.chat(f"Viết một đoạn 2-3 câu trả lời câu hỏi sau như trích từ tài liệu hướng dẫn: {q['q']}",
                              system=SYS, max_new_tokens=120)
        variants, _, s2 = llm.chat(f"Viết 3 cách diễn đạt khác của câu hỏi sau, mỗi cách một dòng, không đánh số: {q['q']}",
                                   system=SYS, max_new_tokens=120)
        gen_s["hyde"].append(s1)
        gen_s["multiquery"].append(s2)
        h = dense.E @ emb.encode([hyp], kind="passage")[0]          # đoạn giả định là "passage"
        vs = [v.strip(" -•\t") for v in variants.splitlines() if len(v.strip()) > 8][:3]
        mq = rrf([base] + [dense.scores(v) for v in vs])
        for name, s in (("dense", base), ("hyde", h), ("hyde_plus_q", rrf([base, h])), ("multiquery", mq)):
            ranked = [chunks[i] for i in np.argsort(-s)[:10]]
            res[name].append({"id": q["id"], **rank_metrics(ranked, q["a"], n_rel[q["id"]])})
        if len(examples) < 6 and q["id"] in ("h26", "h41", "f13", "f32", "h56", "f37"):
            examples.append({"id": q["id"], "q": q["q"], "hyde": hyp, "variants": vs})
        print(q["id"], {k: v[-1]["hit@1"] for k, v in res.items()}, flush=True)
    agg = {m: {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "id"} for m, rows in res.items()}
    for m, rows in res.items():
        agg[m]["wins_vs_dense"] = int(sum(r["mrr"] > d["mrr"] for r, d in zip(rows, res["dense"])))
        agg[m]["losses_vs_dense"] = int(sum(r["mrr"] < d["mrr"] for r, d in zip(rows, res["dense"])))
    save_json("advanced", {"agg": agg, "examples": examples, "gen_s": {k: float(np.median(v)) for k, v in gen_s.items()},
                           "per_q": res})
    print(agg)


if __name__ == "__main__":
    t0 = time.perf_counter()
    main()
    print("tổng", time.perf_counter() - t0)
