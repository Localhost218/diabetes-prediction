"""Chạy bốn hệ (direct, cot, pal, multi) trên tập con GSM8K; ghi từng câu vào outputs/logs/ch7_rows_<hệ>.jsonl
(chạy lại thì tiếp tục từ câu chưa làm). Dùng: python scripts/ch7_run.py [--n 250] [--systems direct,cot,pal,multi]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch6_threads import torch_threads  # noqa: E402
from tl3.common import LOGS, seed_all  # noqa: E402
from tl3 import gpu  # noqa: E402
from tl3.ch7_agents import load_gsm8k, run_question  # noqa: E402


def rows_path(system):
    return LOGS / f"ch7_rows_{system}.jsonl"


def run_with_retry(llm, system, q, tries=30):
    """Máy dùng chung có lúc cạn bộ nhớ commit giữa chừng (RuntimeError 'bad allocation'); chờ rồi làm lại đúng câu đó
    với cùng hạt giống, kết quả không đổi so với chạy một mạch."""
    for k in range(tries):
        try:
            seed_all(q["id"])
            return run_question(llm, system, q)
        except (RuntimeError, MemoryError) as e:
            if "alloc" not in str(e) and not isinstance(e, MemoryError):
                raise
            print(f"[{system}] câu {q['id']}: thiếu bộ nhớ, thử lại ({k + 1}/{tries})", flush=True)
            wait_for_commit()
    raise RuntimeError("hết lượt thử")


def run_system(llm, system, questions):
    """Giữ khoá GPU nặng trong suốt một hệ, nhả giữa các hệ để làn khác xen vào."""
    path = rows_path(system)
    done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    todo = [q for q in questions if q["id"] not in done]
    print(f"[{system}] còn {len(todo)}/{len(questions)} câu", flush=True)
    with gpu.heavy(f"ch7 {system}"), path.open("a", encoding="utf-8") as f:
        for i, q in enumerate(todo, 1):
            r = run_with_retry(llm, system, q)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print(f"[{system}] {i}/{len(todo)}", flush=True)


def free_commit_gb():
    """Bộ nhớ RAM/swap còn trống (GB)."""
    import psutil
    vm = psutil.virtual_memory()
    sm = psutil.swap_memory()
    return (vm.available + sm.free) / 2 ** 30


def wait_for_commit(need_gb=1.5, wait=15):
    import time
    while (free := free_commit_gb()) < need_gb:
        print(f"RAM trống {free:.1f} GB < {need_gb} GB, chờ {wait}s", flush=True)
        time.sleep(wait)


def wait_for_vram(need_gb=3.1, wait=15):
    import time
    import torch
    while (free := torch.cuda.mem_get_info()[0] / 2 ** 30) < need_gb:
        print(f"VRAM trống {free:.1f} GB < {need_gb} GB, chờ {wait}s", flush=True)
        time.sleep(wait)


def load_llm_with_retry(cls, tries=40, wait=60):
    """Các làn khác chạy song song có lúc chiếm gần hết bộ nhớ commit của Windows (os error 1455 khi nạp
    safetensors); chờ rồi thử lại thay vì bỏ cả lượt chạy."""
    import gc
    import time
    for k in range(tries):
        wait_for_commit()
        wait_for_vram()
        try:
            with gpu.heavy("ch7 nạp LLM"):
                return cls()
        except (OSError, RuntimeError, MemoryError) as e:
            print(f"nạp LLM lỗi ({e.__class__.__name__}: {str(e)[:80]}), thử lại sau {wait}s [{k + 1}/{tries}]",
                  flush=True)
            gc.collect()
            time.sleep(wait)
    raise RuntimeError("không nạp được LLM")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--systems", default="direct,cot,pal,multi")
    ap.add_argument("--limit", type=int, default=None, help="chỉ làm k câu đầu của cùng tập mẫu n câu")
    a = ap.parse_args()
    torch_threads(3)
    from tl3.ch7_llm import load_lowmem
    limit = min(a.limit, 50) if a.limit else 50
    questions = load_gsm8k(a.n)[:limit]
    llm = load_llm_with_retry(load_lowmem)
    for s in a.systems.split(","):
        run_system(llm, s, questions)
    import ch7_eval
    ch7_eval.main()
    print("XONG", flush=True)


if __name__ == "__main__":
    main()
