"""
Cấu hình hệ thống, đọc từ biến môi trường (và file .env ở thư mục gốc nếu có).
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Nạp file .env đơn giản (KEY=VALUE), không ghi đè biến đã có trong môi trường."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(BASE_DIR / ".env")


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_bool(key: str, default: bool) -> bool:
    return _env(key, str(default)).lower() in ("1", "true", "yes", "on")


LLM_PROVIDER = _env("LLM_PROVIDER", "gemini").lower()  # gemini | anthropic
DEFAULT_LLM_MODELS = {"gemini": "gemini-flash-latest", "anthropic": "claude-opus-5-5"}


def _llm_enabled() -> bool:
    mode = _env("LLM_ENABLED", "auto").lower()
    if mode == "auto":
        if LLM_PROVIDER == "anthropic":
            return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))
        return bool(os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
    return mode in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    # Bộ lưu trữ mặc định: SQLite (DB quan hệ + full-text FTS5) · Qdrant (vector) · Neo4j (Knowledge Graph).
    # Bộ cũ vẫn dùng được: DATABASE_URL=postgresql+psycopg://... VECTOR_BACKEND=pgvector KG_BACKEND=age|sql
    database_url: str = _env("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'multiagent.db'}")
    vector_backend: str = _env("VECTOR_BACKEND", "qdrant")  # qdrant | pgvector
    kg_backend: str = _env("KG_BACKEND", "neo4j")  # neo4j | age | sql
    kg_graph_name: str = _env("KG_GRAPH_NAME", "diabetes_kg")

    # Qdrant: QDRANT_URL rỗng → chế độ nhúng (lưu trên đĩa tại QDRANT_PATH, không cần server)
    qdrant_url: str = _env("QDRANT_URL", "")
    qdrant_api_key: str = _env("QDRANT_API_KEY", "")
    qdrant_path: Path = Path(_env("QDRANT_PATH", str(BASE_DIR / "data" / "qdrant")))
    qdrant_collection: str = _env("QDRANT_COLLECTION", "kb_chunks")

    # Neo4j
    neo4j_uri: str = _env("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = _env("NEO4J_USER", "neo4j")
    neo4j_password: str = _env("NEO4J_PASSWORD", "multiagent-neo4j")
    neo4j_database: str = _env("NEO4J_DATABASE", "neo4j")

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    # Lưu trữ file (SeaweedFS qua S3 API)
    storage_backend: str = _env("STORAGE_BACKEND", "s3")  # s3 | local
    s3_endpoint_url: str = _env("S3_ENDPOINT_URL", "http://localhost:8333")
    s3_access_key: str = _env("S3_ACCESS_KEY", "astarcare")
    s3_secret_key: str = _env("S3_SECRET_KEY", "astarcare-dev-secret")
    s3_region: str = _env("S3_REGION", "us-east-1")
    s3_bucket_uploads: str = _env("S3_BUCKET_UPLOADS", "astarcare-uploads")
    s3_bucket_kb: str = _env("S3_BUCKET_KB", "astarcare-kb")
    local_storage_dir: Path = BASE_DIR / "data" / "local_storage"
    upload_ttl_days: int = int(_env("UPLOAD_TTL_DAYS", "30"))
    max_upload_mb: int = int(_env("MAX_UPLOAD_MB", "15"))

    # LLM (Gemini hoặc Claude) — không bắt buộc, mọi agent đều có nhánh dự phòng bằng luật
    llm_provider: str = LLM_PROVIDER
    llm_enabled: bool = field(default_factory=_llm_enabled)
    llm_model: str = _env("LLM_MODEL", DEFAULT_LLM_MODELS.get(LLM_PROVIDER, "gemini-flash-latest"))
    llm_timeout_s: float = float(_env("LLM_TIMEOUT_S", "25"))  # cho mỗi lần thử một model
    # Gemini: thử lần lượt khi model chính quá tải / không khả dụng
    llm_fallback_models: tuple = tuple(m.strip() for m in _env(
        "LLM_FALLBACK_MODELS", "gemini-3.5-flash,gemini-flash-lite-latest").split(",") if m.strip())

    # Embedding cho RAG
    embedding_backend: str = _env("EMBEDDING_BACKEND", "fastembed")  # fastembed | hashing
    embedding_model: str = _env("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    embedding_dim: int = int(_env("EMBEDDING_DIM", "384"))

    # Bộ dữ liệu của mô hình dự đoán: nhanes (dữ liệu thật CDC) | vn (vietnam_diabetes_dataset)
    prediction_dataset: str = _env("PREDICTION_DATASET", "nhanes")

    # Hội đồng tranh biện
    max_debate_rounds: int = int(_env("MAX_DEBATE_ROUNDS", "2"))

    # Đường dẫn dữ liệu
    dataset_csv: Path = BASE_DIR / "data" / "vietnam_diabetes_dataset.csv"
    model_dir: Path = BASE_DIR / "data" / "models"
    kb_sources_dir: Path = BASE_DIR / "data" / "kb_sources"
    frontend_dist: Path = BASE_DIR / "frontend" / "dist"


settings = Settings()
