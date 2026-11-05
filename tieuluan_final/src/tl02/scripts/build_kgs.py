"""Dựng các đồ thị tri thức (CPU): KG y tế, KG tài chính, KG từ dữ liệu bảng; xuất RDF, chạy SPARQL và suy luận."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2 import kg_tabular as T  # noqa: E402
from tl2.common import DATA, OUT, save_json  # noqa: E402
from tl2.data import load_diabetes  # noqa: E402
from tl2.kg_base import rdfs_closure, short  # noqa: E402
from tl2.kg_finance import build_finance_kg, downstream, impact_paths  # noqa: E402
from tl2.kg_health import build_health_kg, drug_warnings, food_advice  # noqa: E402
from tl2.retrieval import TfidfRetriever  # noqa: E402
from tl2.chunking import Chunk  # noqa: E402

PREFIX = "PREFIX ex: <http://tl02.ptit.local/kg#>\nPREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>\n" \
         "PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>\n"
SPARQL = {
    "q1_tuong_tac_ruou": ("health", """
SELECT ?thuoc ?mucdo ?hauqua WHERE {
  ?thuoc ex:TUONG_TAC_NGUY_HIEM ex:Ruou_Bia .
  ?st rdf:subject ?thuoc ; rdf:predicate ex:TUONG_TAC_NGUY_HIEM ; rdf:object ex:Ruou_Bia ;
      ex:severity ?mucdo ; ex:effect ?hauqua .
}"""),
    "q2_thuc_pham_gi": ("health", """
SELECT ?mon ?lop ?gi WHERE {
  ?mon a ex:Food ; a ?lop ; ex:gi ?gi .
  ?lop rdfs:subClassOf ex:Food .
  FILTER(isNumeric(?gi))
} ORDER BY DESC(?gi)"""),
    "q3_tieu_chuan": ("health", """
SELECT ?chiso ?nguong ?donvi WHERE {
  ?chiso ex:TIEU_CHUAN_CHAN_DOAN ex:DTD_Tip2 ; ex:threshold ?nguong ; ex:unit ?donvi .
}"""),
    "q4_thuoc_theo_lop": ("health", """
SELECT ?lop (COUNT(?t) AS ?so) WHERE {
  ?t a ex:Drug ; a ?lop . ?lop rdfs:subClassOf ex:Drug .
} GROUP BY ?lop ORDER BY ?lop"""),
    "q5_lan_truyen_fed": ("finance", """
SELECT DISTINCT ?bien WHERE {
  ex:Lai_Suat_Fed (ex:TAC_DONG_CUNG_CHIEU|ex:TAC_DONG_NGUOC_CHIEU)+ ?bien .
}"""),
    "q6_co_phieu_hose": ("finance", """
SELECT ?cp ?nganh WHERE {
  ?cp a ex:Stock ; ex:THUOC_VE ex:HOSE ; ex:THUOC_NGANH ?n . ?n rdfs:label ?nganh .
}"""),
    "q7_bat_loi_ty_gia": ("finance", """
SELECT ?dn ?cochế WHERE {
  ex:USD_VND ex:TAC_DONG_NGUOC_CHIEU ?dn . ?dn a ex:Stock .
  ?st rdf:subject ex:USD_VND ; rdf:object ?dn ; ex:mechanism ?cochế .
}"""),
}


def run_sparql(g, q):
    t0 = time.perf_counter()
    rows = [[short(x) for x in r] for r in g.query(PREFIX + q)]
    return rows, 1000 * (time.perf_counter() - t0)


def kg_json(kg):
    """Định dạng nút/cạnh cho trang Khám phá KG của Web App."""
    return {"nodes": [{"id": n, "label": d["label"], "type": d["type"]} for n, d in kg.G.nodes(data=True)],
            "edges": [{"s": s, "p": p, "o": o, "label": kg.onto.predicates[p].label, "source": d.get("source")}
                      for s, p, o, d in kg.facts()]}


def main():
    out = {}
    exp = OUT / "kg"
    exp.mkdir(parents=True, exist_ok=True)
    kgs = {"health": build_health_kg(), "finance": build_finance_kg()}
    closed = {}
    for key, kg in kgs.items():
        g = kg.to_rdf()
        t0 = time.perf_counter()
        g2, n_inf = rdfs_closure(g)
        inf_ms = 1000 * (time.perf_counter() - t0)
        g.serialize(exp / f"{key}.ttl", format="turtle")
        (exp / f"{key}.json").write_text(json.dumps(kg_json(kg), ensure_ascii=False), encoding="utf-8")
        closed[key] = g2
        s = kg.stats()
        s.update(rdf_triples=len(g), inferred=n_inf, inference_ms=inf_ms, rdf_after=len(g2))
        out[key] = s
        print(key, {k: v for k, v in s.items() if not isinstance(v, dict)})

    # SPARQL trước và sau suy luận: truy vấn dựa vào lớp cha chỉ trả kết quả sau khi suy luận
    out["sparql"] = {}
    for name, (dom, q) in SPARQL.items():
        raw, ms_raw = run_sparql(kgs[dom].to_rdf(), q)
        rows, ms = run_sparql(closed[dom], q)
        out["sparql"][name] = {"query": q.strip(), "rows": rows, "rows_before_inference": len(raw), "ms": ms,
                               "domain": dom}
        print(name, len(raw), "->", len(rows), rows[:4])

    # ràng buộc domain/range chặn dữ liệu sai
    h = kgs["health"]
    bad = [("Rau_Xanh", "CHI_DINH_DAU_TAY", "Metformin"), ("Metformin", "TIEU_CHUAN_CHAN_DOAN", "DTD_Tip2"),
           ("Com_Trang", "TUONG_TAC_NGUY_HIEM", "Ruou_Bia"), ("DTD_Tip2", "KHUYEN_NGHI_AN", "Metformin")]
    out["violations"] = []
    for t in bad:
        try:
            h.validate(*t)
            out["violations"].append({"triple": t, "error": None})
        except ValueError as e:
            out["violations"].append({"triple": t, "error": str(e)})

    # truy vấn nghiệp vụ trên KG
    direct, shared = drug_warnings(h, "Metformin", "Ruou_Bia")
    out["demo_health"] = {"warning": [h.verbalize(f) for f in direct], "shared_risk": [h.label(x) for x in shared],
                          "chicken": [h.verbalize(f) for f in food_advice(h, "DTD_Tip2", "Thit_Ga")],
                          "rice": [h.verbalize(f) for f in food_advice(h, "DTD_Tip2", "Tinh_Bot")]}
    f = kgs["finance"]
    out["demo_finance"] = {}
    for a, b in (("USD_VND", "FPT"), ("USD_VND", "HPG"), ("Lai_Suat_Fed", "Vang_SJC"), ("Lai_Suat_Fed", "FPT"),
                 ("ND_232_2025", "Vang_SJC"), ("DXY", "VN30"), ("Lai_Suat_NHNN", "HPG")):
        r = impact_paths(f, a, b)
        out["demo_finance"][f"{a}->{b}"] = {"verdict": r["verdict"], "net": r["net"],
                                           "paths": [{"nodes": [f.label(n) for n in p["nodes"]], "sign": p["sign"]}
                                                     for p in r["paths"]]}
    out["shock_dxy"] = {f.label(k): v["sign"] for k, v in downstream(f, "DXY", 4).items()}

    # ------------------------------------------------ KB/KG từ dữ liệu bảng
    df = load_diabetes()
    t0 = time.perf_counter()
    docs = T.diabetes_documents(df)
    doc_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    triples = T.diabetes_triples(df)
    G = T.build_graph(triples)
    kg_s = time.perf_counter() - t0
    tab = {"diabetes": {"records": len(df), "documents": len(docs), "triples": len(triples),
                        "nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "doc_s": doc_s, "kg_s": kg_s,
                        "sample_docs": docs["document"].head(3).tolist(),
                        "sample_triples": [[str(x) for x in t] for t in triples[:10]]}}
    # truy vấn có điều kiện số: KG trả lời chính xác, TF-IDF chỉ xấp xỉ theo từ khóa
    t0 = time.perf_counter()
    hit = set(T.retrieve_graph(triples, "has_hba1c_level", lambda v: v >= 8.8))
    hit &= set(T.retrieve_graph(triples, "has_age", lambda v: v >= 70))
    hit &= set(T.retrieve_graph(triples, "has_hypertension", lambda v: v == 1))
    q_ms = 1000 * (time.perf_counter() - t0)
    truth = set(f"Person_{i}" for i in df.index[(df.HbA1c_level >= 8.8) & (df.age >= 70) & (df.hypertension == 1)])
    sample = docs.sample(20000, random_state=0)
    chunks = [Chunk(r.record_id, r.record_id, "tab", 0, 0, r.document) for r in sample.itertuples()]
    tf = TfidfRetriever(chunks)
    top = [c.chunk_id for c, _ in tf.search("HbA1c 8.8 tuổi 70 tăng huyết áp có", 20)]
    truth_s = truth & set(sample.record_id)
    tab["numeric_query"] = {"kg_hits": len(hit), "truth": len(truth), "kg_exact": hit == truth, "kg_ms": q_ms,
                            "tfidf_top20_correct": sum(t in truth_s for t in top), "truth_in_sample": len(truth_s)}
    st = []
    for sym in ("FPT", "HPG", "VCB", "VIC"):
        px = pd.read_csv(DATA / "market" / f"{sym}.csv", parse_dates=["Date"], index_col="Date").loc["2015-01-01":]
        tr = T.stock_triples(px, sym)
        Gs = T.build_graph(tr)
        st.append({"symbol": sym, "days": len(px), "triples": len(tr), "nodes": Gs.number_of_nodes(),
                   "edges": Gs.number_of_edges()})
        if sym == "FPT":
            top_day = max(((s, o) for s, p, o in tr if p == "has_close"), key=lambda x: x[1])
            facts = T.record_facts(tr, top_day[0])
            tab["fpt_max_close"] = {"subject": top_day[0], "date": facts["has_date"], "close": facts["has_close"],
                                    "volume": facts["has_volume"]}
    tab["stocks"] = st
    out["tabular"] = tab
    print(json.dumps(tab["numeric_query"]), tab.get("fpt_max_close"))
    save_json("kg", out)


if __name__ == "__main__":
    main()
