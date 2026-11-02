"""
Tác tử Tri thức (nhánh PDF/Web → KB → RAG → Agent trên sơ đồ)
  - Đối chiếu xét nghiệm với tiêu chuẩn ADA 2024 / QĐ 5481/QĐ-BYT
  - Vector RAG: hybrid search trên pgvector (Chương 5)
  - GraphRAG: duyệt đồ thị tri thức Apache AGE từ các yếu tố nguy cơ của bệnh nhân (Chương 5)
"""

import asyncio
from typing import Any, Dict, List

from src.communication.message_bus import AgentMessage, BaseAgent
from src.knowledge.clinical_criteria import evaluate_glycemia, graph_entities
from src.knowledge.embedder import get_embedder
from src.schemas.patient_profile import PatientProfile
from src.storage.db import session_scope
from src.storage.graph_store import get_graph_store
from src.storage.vector_store import PgVectorStore

RELATION_VI = {"HAS_RISK_FACTOR": "có yếu tố nguy cơ", "DIAGNOSED_BY": "chẩn đoán bằng", "CAUSES": "gây/dẫn tới",
               "SCREENED_BY": "tầm soát bằng", "TREATED_BY": "điều trị/can thiệp bằng",
               "CONTRAINDICATED_IN": "chống chỉ định khi", "PREVENTED_BY": "dự phòng bằng"}


class KnowledgeAgent(BaseAgent):
    name = "KnowledgeAgent"
    handles = {"RETRIEVE_EVIDENCE", "ANSWER_QUESTION", "CHALLENGE_VERIFY_CRITERIA", "CHALLENGE_SCREENING_PLAN"}

    def __init__(self, bus):
        super().__init__(bus)
        self.embedder = get_embedder()

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        if msg.msg_type == "RETRIEVE_EVIDENCE":
            return self.reply(msg, "CLINICAL_EVIDENCE", await asyncio.to_thread(self._evidence, msg.payload))
        if msg.msg_type == "ANSWER_QUESTION":
            passages = await asyncio.to_thread(self._search, msg.payload["question"], 5)
            return self.reply(msg, "KNOWLEDGE_PASSAGES", {"passages": passages, "citations": _citations(passages)})
        if msg.msg_type == "CHALLENGE_SCREENING_PLAN":
            return self.reply(msg, "CHALLENGE_RESPONSE", await asyncio.to_thread(self._screening_plan, msg.payload),
                              "RESPONSE")
        return self.reply(msg, "CHALLENGE_RESPONSE", await asyncio.to_thread(self._verify_criteria, msg.payload),
                          "RESPONSE")

    # ------------------------------------------------------------------
    def _search(self, query: str, k: int) -> List[Dict[str, Any]]:
        vec = self.embedder.embed([query])[0]
        with session_scope() as s:
            rows = PgVectorStore(s).hybrid_search(query, vec, k=k)
        return [{"chunk_id": r["id"], "document_id": r["document_id"], "title": r["title"], "heading": r["heading"],
                 "source": r["source"], "url": r["url"], "content": r["content"],
                 "score": round(float(r["score"]), 4)} for r in rows]

    def _graph(self, entities: List[str]) -> Dict[str, Any]:
        with session_scope() as s:
            edges = get_graph_store(s).neighborhood(entities, max_hops=2, limit=60)
            chunk_ids = sorted({c for e in edges if e["hops"] == 1 for c in e["chunk_ids"]})[:6]
            chunks = PgVectorStore(s).chunks_by_ids(chunk_ids)
        recommendations = {
            "screenings": _tails(edges, "SCREENED_BY"),
            "interventions": _tails(edges, "TREATED_BY", {"Intervention"}) + _tails(edges, "PREVENTED_BY"),
            "drug_classes": _tails(edges, "TREATED_BY", {"Drug"}),
            "complications": _tails(edges, "CAUSES", {"Complication"}),
            "contraindications": [f"{e['head']} chống chỉ định khi {e['tail']}" for e in edges
                                  if e["relation"] == "CONTRAINDICATED_IN" and e["tail"] in entities],
        }
        return {
            "start_entities": entities,
            "paths": [f"{e['head']} —{RELATION_VI[e['relation']]}→ {e['tail']}" for e in edges[:25]],
            "edges": edges,
            "recommendations": recommendations,
            "linked_chunks": [{"chunk_id": c["id"], "document_id": c["document_id"], "title": c["title"],
                               "heading": c["heading"], "source": c["source"], "url": c["url"],
                               "content": c["content"]} for c in chunks],
        }

    def _evidence(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        profile = PatientProfile.model_validate(payload["profile"])
        glycemia = evaluate_glycemia(profile)
        entities = graph_entities(profile, glycemia["status"])
        graph = self._graph(entities)

        query_terms = [glycemia["status_vi"]] + [f["test"] for f in glycemia["findings"]] + entities[:5]
        if glycemia["status"] in ("unknown", "normal"):
            query_terms.append("sàng lọc yếu tố nguy cơ")
        passages = self._search(" ".join(query_terms), 4)

        # Gộp đoạn văn từ vector RAG và từ đồ thị, bỏ trùng
        seen, merged = set(), []
        for p in passages + graph["linked_chunks"]:
            if p["chunk_id"] not in seen:
                seen.add(p["chunk_id"])
                merged.append(p)

        return {
            "glycemia": glycemia,
            "graph_rag": {k: v for k, v in graph.items() if k != "linked_chunks"},
            "passages": merged[:6],
            "citations": _citations(merged[:6]),
        }

    def _verify_criteria(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        glycemia = payload["glycemia"]
        passages = self._search("cần xác nhận lại kết quả hai xét nghiệm bất thường chẩn đoán", 2)
        rule = next((p for p in passages if "xác nhận" in (p["heading"] or "").lower()), passages[0] if passages else None)
        abnormal = [f"{f['test']} {f['value']:g} {f['unit']}" for f in glycemia["findings"] if f["status"] == "diabetes_range"]
        return {
            "position": "confirmation_required" if glycemia["needs_confirmation"] else "criteria_met",
            "statement": (f"Theo ADA 2024, khi không có tăng đường huyết rõ ràng cần 2 kết quả bất thường để chẩn đoán xác định. "
                          f"Hiện có {len(abnormal)} kết quả vượt ngưỡng ({', '.join(abnormal)}): "
                          + ("cần xét nghiệm lại (FPG hoặc HbA1c) để xác nhận." if glycemia["needs_confirmation"]
                             else "đã đủ tiêu chí.")),
            "citation": _citations([rule])[0] if rule else None,
        }


    def _screening_plan(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        passages = self._search("sàng lọc lặp lại ít nhất mỗi 3 năm người có yếu tố nguy cơ thay đổi lối sống", 3)
        rule = next((p for p in passages if p["document_id"] == "sang-loc-nguoi-viet"), passages[0] if passages else None)
        factors = [f for f in payload["risk_factors"] if f not in ("Tuổi ≥ 35",)]
        return {
            "position": "screening_schedule",
            "statement": ("Xét nghiệm hiện chưa vượt ngưỡng nhưng có yếu tố nguy cơ "
                          f"({', '.join(factors) or 'theo mô hình ML'}): ADA khuyến cáo sàng lọc lại ít nhất mỗi 3 năm, "
                          "nên làm sớm hơn (mỗi năm) khi có nhiều yếu tố nguy cơ, kết hợp giảm cân và vận động 150 phút/tuần."),
            "citation": _citations([rule])[0] if rule else None,
        }


def _tails(edges: List[Dict[str, Any]], relation: str, labels: set = None) -> List[str]:
    out = []
    for e in edges:
        if e["relation"] == relation and (labels is None or e["tail_label"] in labels) and e["tail"] not in out:
            out.append(e["tail"])
    return out


def _citations(passages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen, out = set(), []
    for p in passages:
        key = (p["document_id"], p["heading"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"document_id": p["document_id"], "title": p["title"], "section": p["heading"],
                    "source": p["source"], "url": p["url"]})
    return out
