"""Tiện ích dùng chung cho mọi thí nghiệm của tiểu luận: đường dẫn, seed, đo thời gian, lưu kết quả."""
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
METRICS = ROOT / "outputs" / "metrics"
FIGURES = ROOT / "outputs" / "figures"
LOGS = ROOT / "outputs" / "logs"
SEED = 42

for _d in (MODELS, METRICS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)


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


@contextmanager
def timer(store, key):
    """with timer(d, 'train_s'): ... -> d['train_s'] = số giây chạy khối lệnh."""
    t0 = time.perf_counter()
    yield
    store[key] = time.perf_counter() - t0


def file_kb(path):
    return Path(path).stat().st_size / 1024


def _to_py(o):
    if isinstance(o, dict):
        return {str(k): _to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_to_py(v) for v in o]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def save_json(obj, name):
    """Ghi outputs/metrics/<name>.json; build.py của báo cáo đọc mọi số liệu từ đây."""
    path = METRICS / f"{name}.json"
    path.write_text(json.dumps(_to_py(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_json(name):
    return json.loads((METRICS / f"{name}.json").read_text(encoding="utf-8"))


def merge_json(name, key, value):
    """Gộp một mục vào tệp JSON có sẵn: các tiến trình CPU và GPU ghi vào tệp riêng nên không tranh chấp."""
    path = METRICS / f"{name}.json"
    obj = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    obj[key] = _to_py(value)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def log(msg, name="run"):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOGS / f"{name}.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
