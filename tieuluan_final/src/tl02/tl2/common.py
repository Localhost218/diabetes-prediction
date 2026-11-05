"""Đường dẫn, thiết bị, hạt giống và ghi kết quả dùng chung cho mọi kịch bản."""
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Nạp trọng số tuần tự: luồng song song của transformers đọc tệp safetensors ánh xạ bộ nhớ và sập (access
# violation) khi bộ nhớ commit của Windows gần cạn, mà máy này phải giữ LLM, embedder, reranker cùng lúc.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS = DATA / "corpus"
MODELS = ROOT / "models"
HF = MODELS / "hf"
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"
SEED = 42

EMBEDDERS = {
    "e5-small": "multilingual-e5-small",
    "e5-base": "multilingual-e5-base",
    "minilm": "paraphrase-multilingual-MiniLM-L12-v2",
}
RERANKER = "bge-reranker-v2-m3-bf16"     # bản bf16 do scripts/download_models.py ghi từ bản gốc fp32
LLM = "Qwen2.5-1.5B-Instruct"


def seed_all(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def device():
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def hf_path(name):
    p = HF / name
    if not p.exists():
        raise FileNotFoundError(f"Chưa tải mô hình {name} vào {p}; chạy scripts/download_models.py")
    return str(p)


def vn(x, d=0, sign=False):
    """Số kiểu Việt cho văn bản công cụ trả về: 4.125,4 và −2,0 (chấm ngăn nghìn, phẩy thập phân)."""
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".").replace("-", "−")


def save_json(name, obj):
    METRICS.mkdir(parents=True, exist_ok=True)
    path = METRICS / f"{name}.json"
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=_to_py), encoding="utf-8")
    return path


def load_json(name):
    return json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8"))


def _to_py(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Không tuần tự hóa được {type(o)}")


@contextmanager
def timer(store, key):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0
