"""
Knowledge Graph store.

- AgeGraphStore: Apache AGE (openCypher chạy trong PostgreSQL) — mặc định.
- SqlGraphStore: bảng kg_nodes/kg_edges + WITH RECURSIVE — dự phòng cho Postgres không cài được AGE
  (ví dụ AWS RDS). Chọn bằng KG_BACKEND=age|sql.

Mỗi cạnh lưu `chunk_ids` trỏ về kb_chunks để GraphRAG trích dẫn được đoạn văn bản gốc.
"""

import json
import re
from typing import Any, Dict, Iterable, List, Protocol

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.config import settings

IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
LABELS = {"RiskFactor", "Disease", "Complication", "Test", "Drug", "Intervention"}
RELATIONS = {"HAS_RISK_FACTOR", "DIAGNOSED_BY", "CAUSES", "SCREENED_BY", "TREATED_BY",
             "CONTRAINDICATED_IN", "PREVENTED_BY"}


class GraphStore(Protocol):
    def upsert_triples(self, triples: Iterable[Dict[str, Any]]) -> int: ...
    def neighborhood(self, start_names: List[str], max_hops: int = 2, limit: int = 60) -> List[Dict[str, Any]]: ...
    def stats(self) -> Dict[str, int]: ...
    def clear(self) -> None: ...


def _validate(t: Dict[str, Any]) -> None:
    if t["head_label"] not in LABELS or t["tail_label"] not in LABELS or t["relation"] not in RELATIONS:
        raise ValueError(f"Bộ ba ngoài ontology: {t}")


def _cypher_str(s: str) -> str:
    s = str(s)
    if "$cy$" in s:
        raise ValueError("Chuỗi chứa ký tự không hợp lệ")
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


class AgeGraphStore:
    def __init__(self, session: Session, graph: str = settings.kg_graph_name):
        if not IDENT.match(graph):
            raise ValueError("Tên graph không hợp lệ")
        self.session = session
        self.graph = graph
        self.session.execute(text("LOAD 'age'"))
        self.session.execute(text('SET search_path = "$user", public, ag_catalog'))

    def _cypher(self, query: str, columns: str = "result agtype") -> List[Any]:
        sql = f"SELECT * FROM ag_catalog.cypher('{self.graph}', $cy$ {query} $cy$) AS ({columns})"
        return self.session.execute(text(sql)).all()

    def upsert_triples(self, triples: Iterable[Dict[str, Any]]) -> int:
        n = 0
        for t in triples:
            _validate(t)
            chunk_ids = "[" + ", ".join(str(int(c)) for c in t.get("chunk_ids", [])) + "]"
            self._cypher(
                f"MERGE (h:{t['head_label']} {{name: {_cypher_str(t['head'])}}}) "
                f"MERGE (x:{t['tail_label']} {{name: {_cypher_str(t['tail'])}}}) "
                f"MERGE (h)-[r:{t['relation']}]->(x) "
                f"SET r.evidence = {_cypher_str(t.get('evidence', ''))}, "
                f"r.source_doc = {_cypher_str(t.get('source_doc', ''))}, r.chunk_ids = {chunk_ids} "
                f"RETURN id(r)"
            )
            n += 1
        return n

    def neighborhood(self, start_names: List[str], max_hops: int = 2, limit: int = 60) -> List[Dict[str, Any]]:
        if not start_names:
            return []
        names = "[" + ", ".join(_cypher_str(n) for n in start_names) + "]"
        rows = self._cypher(
            f"MATCH p = (s)-[*1..{int(max_hops)}]-(t) WHERE s.name IN {names} "
            f"RETURN p LIMIT {int(limit) * 4}",
            "p agtype",
        )
        edges: Dict[Any, Dict[str, Any]] = {}
        for (path,) in rows:
            items = json.loads(re.sub(r"::(vertex|edge|path)", "", str(path)))
            vertices = {v["id"]: v for v in items if "start_id" not in v}
            hop = 0
            for item in items:
                if "start_id" not in item:
                    continue
                hop += 1
                key = item["id"]
                if key in edges:
                    edges[key]["hops"] = min(edges[key]["hops"], hop)
                    continue
                head, tail = vertices[item["start_id"]], vertices[item["end_id"]]
                props = item.get("properties", {})
                edges[key] = {
                    "head": head["properties"]["name"], "head_label": head["label"],
                    "relation": item["label"],
                    "tail": tail["properties"]["name"], "tail_label": tail["label"],
                    "evidence": props.get("evidence", ""), "source_doc": props.get("source_doc", ""),
                    "chunk_ids": props.get("chunk_ids", []), "hops": hop,
                }
        return sorted(edges.values(), key=lambda e: e["hops"])[:limit]

    def stats(self) -> Dict[str, int]:
        nodes = self._cypher("MATCH (n) RETURN count(n)")[0][0]
        edges = self._cypher("MATCH ()-[r]->() RETURN count(r)")[0][0]
        return {"nodes": int(str(nodes)), "edges": int(str(edges))}

    def clear(self) -> None:
        self._cypher("MATCH (n) DETACH DELETE n")


class SqlGraphStore:
    def __init__(self, session: Session):
        self.session = session

    def _node_id(self, label: str, name: str) -> int:
        return self.session.execute(text("""
            INSERT INTO kg_nodes (label, name) VALUES (:label, :name)
            ON CONFLICT (label, name) DO UPDATE SET name = EXCLUDED.name RETURNING id
        """), {"label": label, "name": name}).scalar_one()

    def upsert_triples(self, triples: Iterable[Dict[str, Any]]) -> int:
        n = 0
        for t in triples:
            _validate(t)
            h, x = self._node_id(t["head_label"], t["head"]), self._node_id(t["tail_label"], t["tail"])
            self.session.execute(text("""
                INSERT INTO kg_edges (head_id, relation, tail_id, evidence, source_doc, chunk_ids)
                VALUES (:h, :r, :x, :e, :s, CAST(:c AS jsonb))
                ON CONFLICT (head_id, relation, tail_id)
                DO UPDATE SET evidence = EXCLUDED.evidence, chunk_ids = EXCLUDED.chunk_ids
            """), {"h": h, "r": t["relation"], "x": x, "e": t.get("evidence", ""), "s": t.get("source_doc"),
                   "c": json.dumps([int(c) for c in t.get("chunk_ids", [])])})
            n += 1
        return n

    def neighborhood(self, start_names: List[str], max_hops: int = 2, limit: int = 60) -> List[Dict[str, Any]]:
        if not start_names:
            return []
        rows = self.session.execute(text("""
            WITH RECURSIVE walk(node_id, edge_id, hops, visited) AS (
                SELECT n.id, NULL::int, 0, ARRAY[n.id] FROM kg_nodes n WHERE n.name = ANY(:names)
                UNION ALL
                SELECT CASE WHEN e.head_id = w.node_id THEN e.tail_id ELSE e.head_id END,
                       e.id, w.hops + 1,
                       w.visited || CASE WHEN e.head_id = w.node_id THEN e.tail_id ELSE e.head_id END
                FROM walk w JOIN kg_edges e ON w.node_id IN (e.head_id, e.tail_id)
                WHERE w.hops < :max_hops
                  AND NOT (CASE WHEN e.head_id = w.node_id THEN e.tail_id ELSE e.head_id END) = ANY(w.visited)
            )
            SELECT h.name AS head, h.label AS head_label, e.relation, t.name AS tail, t.label AS tail_label,
                   e.evidence, e.source_doc, e.chunk_ids, min(w.hops) AS hops
            FROM walk w JOIN kg_edges e ON e.id = w.edge_id
            JOIN kg_nodes h ON h.id = e.head_id JOIN kg_nodes t ON t.id = e.tail_id
            GROUP BY e.id, h.name, h.label, t.name, t.label
            ORDER BY hops LIMIT :limit
        """), {"names": start_names, "max_hops": max_hops, "limit": limit}).mappings().all()
        return [dict(r) for r in rows]

    def stats(self) -> Dict[str, int]:
        return {"nodes": self.session.execute(text("SELECT count(*) FROM kg_nodes")).scalar_one(),
                "edges": self.session.execute(text("SELECT count(*) FROM kg_edges")).scalar_one()}

    def clear(self) -> None:
        self.session.execute(text("DELETE FROM kg_edges"))
        self.session.execute(text("DELETE FROM kg_nodes"))


def get_graph_store(session: Session) -> GraphStore:
    return SqlGraphStore(session) if settings.kg_backend == "sql" else AgeGraphStore(session)
