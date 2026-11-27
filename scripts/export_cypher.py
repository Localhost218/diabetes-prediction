"""
Xuất Knowledge Graph đang chạy trên Neo4j ra script Cypher (khôi phục được bằng cypher-shell).

    python scripts/export_cypher.py                      # → data/output/kb/db_diabet.cypher
    cat data/output/kb/db_diabet.cypher | cypher-shell -u neo4j -p <mật khẩu>
"""

import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from neo4j import GraphDatabase  # noqa: E402

from src.config import settings  # noqa: E402

OUT = settings.kb_sources_dir.parent / "db_diabet.cypher"


def lit(value) -> str:
    """Giá trị Python → literal Cypher (chuỗi JSON hợp lệ cũng là chuỗi Cypher hợp lệ)."""
    if isinstance(value, list):
        return "[" + ", ".join(lit(v) for v in value) + "]"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def main() -> None:
    driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
    with driver.session(database=settings.neo4j_database) as s:
        nodes = s.run("MATCH (n) RETURN labels(n)[0] AS label, n.name AS name ORDER BY label, name").data()
        edges = s.run("MATCH (h)-[r]->(t) RETURN labels(h)[0] AS hl, h.name AS h, type(r) AS rel, "
                      "labels(t)[0] AS tl, t.name AS t, properties(r) AS props ORDER BY rel, h, t").data()
    driver.close()

    labels = sorted({n["label"] for n in nodes})
    lines = [
        "// Knowledge Graph đái tháo đường — xuất trực tiếp từ Neo4j bằng scripts/export_cypher.py",
        f"// Thời điểm xuất: {datetime.now():%Y-%m-%d %H:%M}",
        f"// {len(nodes)} nút ({', '.join(f'{l}: {c}' for l, c in sorted(Counter(n['label'] for n in nodes).items()))})",
        f"// {len(edges)} cạnh ({', '.join(f'{r}: {c}' for r, c in sorted(Counter(e['rel'] for e in edges).items()))})",
        "// Mỗi cạnh có source_doc, evidence và chunk_ids trỏ về đoạn văn trong kb_chunks (RAG).",
        "",
        "// 1. Ràng buộc duy nhất",
        *(f"CREATE CONSTRAINT {l.lower()}_name IF NOT EXISTS FOR (n:{l}) REQUIRE n.name IS UNIQUE;" for l in labels),
        "",
        "// 2. Nút",
        *(f"MERGE (:{n['label']} {{name: {lit(n['name'])}}});" for n in nodes),
        "",
        "// 3. Cạnh",
    ]
    for e in edges:
        props = ", ".join(f"r.{k} = {lit(v)}" for k, v in sorted(e["props"].items()))
        lines.append(f"MATCH (h:{e['hl']} {{name: {lit(e['h'])}}}), (t:{e['tl']} {{name: {lit(e['t'])}}}) "
                     f"MERGE (h)-[r:{e['rel']}]->(t)" + (f" SET {props}" if props else "") + ";")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Đã xuất {len(nodes)} nút, {len(edges)} cạnh → {OUT}")


if __name__ == "__main__":
    main()
