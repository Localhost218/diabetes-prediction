"""Tác tử ReAct đa miền: Thought -> Action(công cụ) -> Observation lặp lại, rồi tổng hợp câu trả lời có căn cứ.

Bộ lập kế hoạch là tất định (liên kết thực thể + ý định), vì LLM 1,5B tham số gọi công cụ không ổn định; LLM chỉ
viết câu trả lời cuối từ bằng chứng đã thu thập. Bộ kiểm chứng đối chiếu trích dẫn và con số với bằng chứng; câu
trả lời không qua kiểm chứng bị thay bằng bản tổng hợp theo khuôn và vết suy luận ghi rõ lý do.
"""
import re
import time
from dataclasses import dataclass

from .chunking import chunk_corpus, structure_aware
from .common import vn
from .corpus import load_documents
from .graphrag import EntityLinker, GraphRAG, strip_accents
from .kg_finance import build_finance_kg, impact_paths
from .kg_health import build_health_kg, drug_warnings, food_advice
from .llm import LocalLLM, grounded_prompt

DISCLAIMER = {"health": "Thông tin chỉ mang tính tham khảo, không thay thế chỉ định của bác sĩ.",
              "finance": "Thông tin chỉ mang tính tham khảo, không phải khuyến nghị đầu tư."}
FACTOR_TYPES = {"MacroFactor", "GlobalFactor", "PolicyTool", "Regulation", "Event", "MarketFlow"}
TARGET_TYPES = {"Stock", "GoldAsset", "Index", "Indicator"}
GOLD_NODES = {"Vang_SJC", "Vang_Quy_Doi", "Vang_Nhan", "XAU_USD"}


@dataclass
class Tool:
    name: str
    kind: str          # DL | KG | RAG | CALC
    description: str
    fn: callable


class ReActAgent:
    def __init__(self, use_llm=True, dev=None):
        from .common import device
        from .data import gold_frame, market_frame
        from .dl_models import DiabetesRiskModel, Forecaster
        from .retrieval import BM25Retriever, DenseRetriever, Embedder, HybridRRF, Reranker, RerankPipeline
        dev = dev or device()
        # LLM phải nạp TRƯỚC embedder/reranker: nạp sau thì đọc safetensors của Qwen bị segfault trên máy này
        self.llm = LocalLLM() if use_llm else None
        self.hkg, self.fkg = build_health_kg(), build_finance_kg()
        self.linker = EntityLinker([self.hkg, self.fkg])
        docs = load_documents()
        self.chunks = chunk_corpus(docs, structure_aware, size=900)
        emb = Embedder("e5-base", half=True)
        hybrid = HybridRRF([BM25Retriever(self.chunks), DenseRetriever(self.chunks, emb)])
        self.rag = RerankPipeline(hybrid, Reranker(), n_candidates=30)
        self.graphrag = GraphRAG([self.hkg, self.fkg], self.chunks)
        self.risk = DiabetesRiskModel(dev)
        self._forecast_cls, self._forecasters, self.dev = Forecaster, {}, dev
        self.market, self.gold = market_frame(), gold_frame()
        self.tools = {t.name: t for t in [
            Tool("chan_doan_nguy_co", "DL", "MLP PyTorch ước lượng P(đái tháo đường) từ hồ sơ lâm sàng", self._t_risk),
            Tool("kg_y_te", "KG", "Truy vấn VietnamDiabetesKG: tương tác thuốc, dinh dưỡng, chống chỉ định", self._t_hkg),
            Tool("kg_tai_chinh", "KG", "Đường nhân quả đa bước trên VietnamFinanceKG kèm dấu tác động", self._t_fkg),
            Tool("du_bao_bilstm", "DL", "BiLSTM dự báo giá phiên kế tiếp, khoảng dự báo MC Dropout + phần dư", self._t_forecast),
            Tool("quy_doi_vang", "CALC", "Giá vàng thế giới quy đổi VND/lượng từ XAU/USD và USD/VND", self._t_gold),
            Tool("du_lieu_thi_truong", "CALC", "Giá đóng cửa gần nhất và biến động 20 phiên", self._t_market),
            Tool("truy_xuat_rag", "RAG", "Hybrid BM25 + e5 (RRF) rồi rerank bge-m3 trên kho tài liệu", self._t_rag),
        ]}

    # ================================================================ công cụ
    def _t_risk(self, profile):
        r = self.risk.predict(profile)
        txt = (f"MLP ước lượng xác suất đái tháo đường {vn(100 * r['probability'], 1)}% (ngưỡng quyết định "
               f"{vn(100 * r['threshold'], 1)}%), mức nguy cơ {r['risk_level']}.")
        return r, [("DiabetesMLP", txt)]

    def _t_hkg(self, ents):
        ids = {e for e, kg, _ in ents if kg == self.hkg.name}
        drugs = [e for e in ids if self.hkg.onto.is_a(self.hkg.G.nodes[e]["type"], "Drug")]
        subs = [e for e in ids if self.hkg.G.nodes[e]["type"] == "RiskSubstance"]
        foods = [e for e in ids if self.hkg.onto.is_a(self.hkg.G.nodes[e]["type"], "Food")]
        conds = [e for e in ids if self.hkg.onto.is_a(self.hkg.G.nodes[e]["type"], "Condition")]
        if subs and not drugs:                      # hỏi rượu bia nhưng không nêu thuốc: xét thuốc đầu tay
            drugs = ["Metformin"]
        ev, obs = [], {"warnings": [], "diet": [], "contra": [], "shared_risk": []}
        for d in drugs:
            for s in subs:
                direct, shared = drug_warnings(self.hkg, d, s)
                for f in direct:
                    obs["warnings"].append(self.hkg.verbalize(f))
                    ev.append((f"KG:{f[3]['source']}", self.hkg.verbalize(f)))
                obs["shared_risk"] += [self.hkg.label(x) for x in shared]
            for _, _, o, dd in self.hkg.facts(d, "CHONG_CHI_DINH_KHI"):
                if not conds or o in conds:
                    obs["contra"].append(self.hkg.verbalize((d, "CHONG_CHI_DINH_KHI", o, dd)))
                    ev.append((f"KG:{dd['source']}", obs["contra"][-1]))
        for c in conds:
            for s, p, o, dd in self.hkg.facts(p="UU_TIEN_KHI", o=c):
                obs["contra"].append(self.hkg.verbalize((s, p, o, dd)))
                ev.append((f"KG:{dd['source']}", obs["contra"][-1]))
        for f in foods:
            for fact in food_advice(self.hkg, "DTD_Tip2", f):
                obs["diet"].append(self.hkg.verbalize(fact))
                ev.append((f"KG:{fact[3]['source']}", self.hkg.verbalize(fact)))
        return obs, ev

    def _t_fkg(self, ents):
        ids = [e for e, kg, _ in ents if kg == self.fkg.name]
        types = {e: self.fkg.G.nodes[e]["type"] for e in ids}
        factors = [e for e in ids if self.fkg.onto.is_a(types[e], "MacroFactor") or types[e] in FACTOR_TYPES]
        targets = [e for e in ids if types[e] in TARGET_TYPES] or (["FPT", "HPG", "VCB", "VIC", "Vang_SJC"]
                                                                  if factors else [])
        obs, ev = [], []
        for a in factors:
            for b in targets:
                if a == b:
                    continue
                r = impact_paths(self.fkg, a, b)
                if not r["paths"]:
                    continue
                best = r["paths"][0]
                chain = " → ".join(self.fkg.label(n) for n in best["nodes"])
                why = "; ".join(s["mechanism"] for s in best["steps"])
                obs.append({"factor": a, "target": b, "verdict": r["verdict"], "n_paths": len(r["paths"]),
                            "chain": chain})
                src = best["steps"][-1]["source"]
                ev.append((f"KG:{src}", f"{self.fkg.label(a)} tăng → {self.fkg.label(b)}: {r['verdict']} "
                                        f"({len(r['paths'])} đường). Kênh chính: {chain}. Cơ chế: {why}."))
        return obs, ev

    def _t_forecast(self, target):
        if target not in self._forecasters:
            self._forecasters[target] = self._forecast_cls(target, self.dev)
        r = self._forecasters[target].predict()
        unit = "triệu đồng/lượng" if target == "GOLD_VND" else "đồng/cổ phiếu"
        reliab = ("không vượt" if r["test_rmse_vs_naive"] >= 0.999 else "nhỉnh hơn")
        txt = (f"BiLSTM dự báo {target} phiên sau {r['last_date']}: {vn(r['pred_price'], 1)} {unit} "
               f"(giá gần nhất {vn(r['last_price'], 1)}; khoảng dự báo 90%: {vn(r['interval90'][0], 1)} - "
               f"{vn(r['interval90'][1], 1)}; xác suất tăng {vn(100 * r['prob_up'])}%). Trên tập kiểm thử 2025-2026 "
               f"mô hình đoán đúng chiều {vn(100 * r['test_dir_acc'], 1)}% và {reliab} dự báo giữ nguyên giá "
               f"(RMSE/RMSE ngây thơ = {vn(r['test_rmse_vs_naive'], 3)}).")
        return r, [(f"BiLSTM:{target}", txt)]

    def _t_gold(self, _=None):
        last = self.gold.iloc[-1]
        v = float(last["GOLD_VND"])
        txt = (f"Ngày {self.gold.index[-1]:%d/%m/%Y}: XAU/USD = {vn(last['XAU'], 1)} USD/ounce, USD/VND = "
               f"{vn(last['USDVND'])}; giá thế giới quy đổi = {vn(last['XAU'], 1)} × {vn(last['USDVND'])} × 1,2057 "
               f"≈ {vn(v, 1)} triệu đồng/lượng, chưa gồm thuế phí và phần bù vàng miếng trong nước.")
        return {"xau": float(last["XAU"]), "usdvnd": float(last["USDVND"]), "gold_vnd": v}, [("Tính toán", txt)]

    def _t_market(self, symbols):
        obs, ev = {}, []
        for s in symbols:
            p = self.market[s]
            chg = float(p.iloc[-1] / p.iloc[-21] - 1)
            obs[s] = {"date": p.index[-1].strftime("%Y-%m-%d"), "close": float(p.iloc[-1]), "chg20": chg}
            ev.append((f"Dữ liệu:{s}", f"{s} đóng cửa {vn(p.iloc[-1])} đồng ngày {p.index[-1]:%d/%m/%Y}, "
                                       f"thay đổi 20 phiên {vn(100 * chg, 1, sign=True)}%."))
        return obs, ev

    def _t_rag(self, query, k=3):
        hits = self.rag.search(query, k)
        obs = [{"chunk": c.chunk_id, "score": s, "heading": c.heading} for c, s in hits]
        ev = [(f"RAG:{c.doc_id}", re.sub(r"\s+", " ", c.text.split("\n", 1)[-1]).strip()) for c, _ in hits]
        return obs, ev

    # ================================================================ phân tích câu hỏi
    @staticmethod
    def parse_profile(text):
        t = text.lower().replace(",", ".")
        pats = {"HbA1c_level": r"hba1c\D{0,8}(\d+(?:\.\d+)?)", "bmi": r"bmi\D{0,8}(\d+(?:\.\d+)?)",
                "blood_glucose_level": r"(?:glucose|đường huyết)[^\d]{0,20}(\d{2,3}(?:\.\d+)?)",
                "age": r"(\d{2})\s*tuổi"}
        p = {k: float(m.group(1)) for k, rx in pats.items() if (m := re.search(rx, t))}
        if re.search(r"\b(nữ|phụ nữ|bà)\b", t):
            p["gender"] = "Female"
        elif re.search(r"\b(nam|đàn ông|ông)\b", t):
            p["gender"] = "Male"
        if "huyết áp" in t:
            p["hypertension"] = 1
        if "hút thuốc" in t:
            p["smoking_history"] = "current"
        return p

    def plan(self, question, profile=None):
        ents = self.linker.link(question)
        kgs = [kg for _, kg, _ in ents]
        n_h, n_f = kgs.count(self.hkg.name), kgs.count(self.fkg.name)
        domain = "finance" if n_f > n_h else "health"
        profile = {**self.parse_profile(question), **(profile or {})}
        steps = []
        q = strip_accents(question)
        if domain == "health":
            if any(k in profile for k in ("HbA1c_level", "blood_glucose_level", "bmi")):
                steps.append(("chan_doan_nguy_co", profile,
                              "Người dùng cung cấp chỉ số xét nghiệm; cần mô hình học sâu ước lượng nguy cơ bệnh."))
            if n_h:
                steps.append(("kg_y_te", ents, "Câu hỏi nhắc tới thuốc, chất hoặc món ăn; tra đồ thị tri thức để lấy "
                                               "ràng buộc tương tác và khuyến nghị dinh dưỡng có nguồn."))
        else:
            fin = [(e, self.fkg.G.nodes[e]["type"]) for e, kg, _ in ents if kg == self.fkg.name]
            if any(t in FACTOR_TYPES or self.fkg.onto.is_a(t, "MacroFactor") for _, t in fin):
                steps.append(("kg_tai_chinh", ents, "Câu hỏi về tác động của biến số vĩ mô; cần suy luận đa bước trên "
                                                    "đồ thị nhân quả."))
            stocks = [e for e, t in fin if t == "Stock"]
            if stocks:
                steps.append(("du_lieu_thi_truong", stocks, "Lấy giá gần nhất của cổ phiếu được hỏi."))
            golds = [e for e, _ in fin if e in GOLD_NODES]
            if golds and re.search(r"quy doi|bao nhieu|gia vang|hom nay|hien nay", q):
                steps.append(("quy_doi_vang", None, "Câu hỏi về mức giá vàng; tính giá thế giới quy đổi từ dữ liệu mới nhất."))
            if re.search(r"du bao|du doan|xu huong|phien toi|tuan toi|sap toi|tang hay giam|nen mua", q):
                for e in stocks[:2] + (["GOLD_VND"] if golds else []):
                    steps.append(("du_bao_bilstm", e, f"Người dùng hỏi về xu hướng; gọi BiLSTM dự báo {e}."))
        rag_q = question + " " + " ".join(self._label(e, kg) for e, kg, _ in ents[:4])
        steps.append(("truy_xuat_rag", rag_q, "Truy xuất văn bản gốc trong kho tri thức để trích dẫn căn cứ."))
        return domain, ents, profile, steps

    def _label(self, e, kgname):
        kg = self.hkg if kgname == self.hkg.name else self.fkg
        return kg.label(e)

    # ================================================================ vòng lặp ReAct
    def run(self, question, profile=None, synthesize=True):
        t_all = time.perf_counter()
        domain, ents, profile, plan = self.plan(question, profile)
        trace, evidence, obs_all = [], [], {}
        trace.append({"type": "THOUGHT", "text": f"Miền {domain}; thực thể liên kết: "
                      + (", ".join(self._label(e, kg) for e, kg, _ in ents) or "không có")
                      + f". Kế hoạch: {' → '.join(s[0] for s in plan)}."})
        for tool, arg, why in plan:
            trace.append({"type": "THOUGHT", "text": why})
            t0 = time.perf_counter()
            obs, ev = self.tools[tool].fn(arg)
            ms = 1000 * (time.perf_counter() - t0)
            obs_all[tool] = obs
            trace.append({"type": "ACTION", "tool": tool, "kind": self.tools[tool].kind, "ms": ms,
                          "args": arg if tool != "kg_y_te" and tool != "kg_tai_chinh" else [e for e, _, _ in arg]})
            trace.append({"type": "OBSERVATION", "tool": tool, "text": " | ".join(t for _, t in ev)[:600] or "rỗng"})
            evidence += ev
        result = {"question": question, "domain": domain, "entities": [(e, kg) for e, kg, _ in ents],
                  "profile": profile, "trace": trace, "evidence": evidence, "observations": obs_all}
        if synthesize:
            result.update(self.synthesize(question, domain, evidence, obs_all))
        result["total_ms"] = 1000 * (time.perf_counter() - t_all)
        return result

    def synthesize(self, question, domain, evidence, obs):
        template = self.template_answer(domain, evidence)
        if self.llm is None:
            return {"answer": template + f"\n\n{DISCLAIMER[domain]}",
                    "verification": {"used": "template", "reason": "không dùng LLM"}}
        text, n_tok, sec = self.llm.chat(grounded_prompt(question, evidence))
        ver = verify(text, evidence, question)
        ok = ver["citations_valid"] and ver["has_citation"] and ver["numeric_faithfulness"] >= 0.8
        ver.update(used="llm" if ok else "template", llm_tokens=n_tok, llm_s=sec, llm_text=text)
        if not ok:
            ver["reason"] = "câu trả lời LLM không qua kiểm chứng (trích dẫn hoặc con số không có trong bằng chứng)"
        answer = (text if ok else template) + f"\n\n{DISCLAIMER[domain]}"
        return {"answer": answer, "verification": ver}

    @staticmethod
    def template_answer(domain, evidence):
        lines = [f"- {txt} [{i}]" for i, (_, txt) in enumerate(evidence, 1)]
        return "Tổng hợp từ các công cụ:\n" + "\n".join(lines)


NUM = re.compile(r"\d+(?:[.,]\d+)*")


def _norm_num(s):
    return s.replace(".", "").replace(",", "")


def verify(answer, evidence, question=""):
    """Kiểm chứng: trích dẫn [n] hợp lệ; tỷ lệ con số trong câu trả lời có xuất hiện trong bằng chứng.

    Con số người dùng tự nêu trong câu hỏi (liều 500 mg, HbA1c 7,2) được coi là có căn cứ.
    """
    cites = [int(x) for x in re.findall(r"\[(\d+)\]", answer)]
    ev_text = " ".join(t for _, t in evidence) + " " + question
    ev_nums = {_norm_num(n) for n in NUM.findall(ev_text)}
    nums = [n for n in NUM.findall(re.sub(r"\[\d+\]", "", answer)) if len(_norm_num(n)) >= 2]
    found = [n for n in nums if _norm_num(n) in ev_nums]
    return {"has_citation": bool(cites), "citations_valid": all(1 <= c <= len(evidence) for c in cites),
            "n_citations": len(set(cites)), "numbers": len(nums),
            "numeric_faithfulness": 1.0 if not nums else len(found) / len(nums),
            "unsupported_numbers": [n for n in nums if n not in found]}
