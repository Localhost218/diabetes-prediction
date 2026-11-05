"""Đường dẫn, hạt giống, giới hạn luồng và ghi kết quả dùng chung cho mọi kịch bản của Tiểu luận 03."""
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

# Nạp trọng số tuần tự: luồng song song của transformers đọc safetensors ánh xạ bộ nhớ và sập (access violation)
# khi bộ nhớ commit của Windows gần cạn.
os.environ.setdefault("HF_DEACTIVATE_ASYNC_LOAD", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
OUT = ROOT / "outputs"
METRICS = OUT / "metrics"
LOGS = OUT / "logs"
REPORT = ROOT.parents[1] / "Report" / "TieuLuan 03"
GEN = REPORT / "generated"
FIGS = REPORT / "figures"
# Mô hình Hugging Face đã tải ở Tiểu luận 02 (Qwen2.5-1.5B-Instruct, e5, bge-reranker) — chỉ đọc, không chép.
HF = ROOT.parent / "tl02" / "models" / "hf"   # tiểu luận final: mã TL02 nằm ở src/tl02
LLM_NAME = "Qwen2.5-1.5B-Instruct"
SEED = 42

for _d in (DATA, MODELS, METRICS, LOGS, GEN, FIGS):
    _d.mkdir(parents=True, exist_ok=True)


def limit_threads(n):
    """Giới hạn số luồng BLAS/OpenMP/torch/TF của tiến trình để nhiều làn chạy song song trên 32 luồng."""
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = str(n)
    # TF đọc hai biến này lúc khởi tạo, nên nạp TF sau lời gọi này vẫn nhận giới hạn luồng
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(n)
    os.environ["TF_NUM_INTEROP_THREADS"] = str(max(1, n // 2))
    try:
        import torch
        torch.set_num_threads(n)
    except ImportError:
        pass
    # Chỉ chỉnh TF khi tiến trình đã nạp nó: import tensorflow tốn 1–2 GB bộ nhớ commit cho cả tiến trình chỉ dùng torch.
    tf = sys.modules.get("tensorflow")
    if tf is not None:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(n)
            tf.config.threading.set_inter_op_parallelism_threads(max(1, n // 2))
        except RuntimeError:
            pass


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


def data_dir(ch):
    p = DATA / ch
    p.mkdir(parents=True, exist_ok=True)
    return p


def model_dir(ch):
    p = MODELS / ch
    p.mkdir(parents=True, exist_ok=True)
    return p


def hf_path(name):
    p = HF / name
    if not p.exists():
        raise FileNotFoundError(f"Không thấy mô hình {name} ở {p}")
    return str(p)


def vn(x, d=0, sign=False):
    """Số kiểu Việt cho văn bản giao diện/LLM: 4.125,4 và −2,0."""
    s = f"{x:+,.{d}f}" if sign else f"{x:,.{d}f}"
    return s.replace(",", "\0").replace(".", ",").replace("\0", ".").replace("-", "−")


def save_json(name, obj):
    """Ghi outputs/metrics/<name>.json — nguồn duy nhất của mọi con số \\SL{name/...} trong báo cáo."""
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
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"Không tuần tự hóa được {type(o)}")


@contextmanager
def timer(store, key):
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0
