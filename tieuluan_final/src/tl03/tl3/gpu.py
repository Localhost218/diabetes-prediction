"""Chia sẻ một GPU 8 GB giữa nhiều tiến trình chạy song song.

heavy(): khoá tệp toàn máy cho việc GPU nặng (CNN PyTorch, LLM, embedding hàng loạt) — chỉ một việc nặng một lúc.
cap(): giới hạn phần VRAM của tiến trình cho việc nhẹ (MLP, RNN, GNN) chạy song song không cần khoá.
"""
import os
import time
from contextlib import contextmanager

from .common import OUT

LOCK = OUT / "gpu_heavy.lock"


def cap(fraction=0.2):
    import torch
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(fraction)


def _holder_dead():
    """Tiến trình ghi trong khoá đã chết chưa. Tiến trình sập kiểu access violation không chạy khối finally nên để lại
    khoá; không gỡ thì mọi tiến trình sau chờ mãi."""
    import psutil
    try:
        pid = int(LOCK.read_text(encoding="utf-8").split()[0])
    except (FileNotFoundError, ValueError, IndexError):
        return False  # khoá vừa bị gỡ hoặc đang được ghi dở
    return not psutil.pid_exists(pid)


@contextmanager
def heavy(what, poll=5.0, stale_after=6 * 3600):
    """Chờ tới lượt rồi giữ khoá; khoá của tiến trình đã chết hoặc cũ hơn stale_after giây bị gỡ."""
    t0 = time.time()
    while True:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()} {what} {time.strftime('%H:%M:%S')}".encode("utf-8"))
            os.close(fd)
            break
        except FileExistsError:
            try:
                if _holder_dead() or time.time() - LOCK.stat().st_mtime > stale_after:
                    print(f"[gpu] gỡ khoá bỏ lại: {LOCK.read_text(encoding='utf-8')}", flush=True)
                    LOCK.unlink(missing_ok=True)
                    continue
            except FileNotFoundError:
                continue
            time.sleep(poll)
    waited = time.time() - t0
    if waited > 1:
        print(f"[gpu] chờ khoá {waited:.0f}s cho: {what}", flush=True)
    try:
        yield
    finally:
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        LOCK.unlink(missing_ok=True)
