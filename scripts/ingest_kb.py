"""
Nạp tri thức vào Knowledge Base (pgvector + Apache AGE).

    python scripts/ingest_kb.py                       # nạp data/kb_sources/
    python scripts/ingest_kb.py --url https://...     # nạp thêm trang web / PDF từ URL
    python scripts/ingest_kb.py --llm-triples         # trích thêm bộ ba tri thức bằng LLM
    python scripts/ingest_kb.py --force               # embed lại toàn bộ
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import settings  # noqa: E402
from src.knowledge.ingest import discover_sources, ingest_documents, load_url, rebuild_graph  # noqa: E402
from src.storage.db import init_db  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", action="append", default=[], help="URL trang web hoặc PDF cần nạp")
    parser.add_argument("--llm-triples", action="store_true", help="Trích thêm bộ ba tri thức bằng LLM")
    parser.add_argument("--force", action="store_true", help="Embed lại kể cả tài liệu không đổi")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    init_db()
    docs = discover_sources(settings.kb_sources_dir) + [load_url(u) for u in args.url]
    print("Vector store:", ingest_documents(docs, force=args.force))
    if args.llm_triples and not settings.llm_enabled:
        print("Bỏ qua --llm-triples: chưa cấu hình API key cho LLM")
    print("Knowledge graph:", rebuild_graph(llm_triples=args.llm_triples and settings.llm_enabled))


if __name__ == "__main__":
    main()
