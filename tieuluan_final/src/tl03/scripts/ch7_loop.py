"""Chạy lại ch7_run.py cho tới khi in XONG: tiến trình có thể segfault khi nạp LLM lúc máy thiếu bộ nhớ commit;
các câu đã làm nằm trong outputs/logs/ch7_rows_*.jsonl nên lần chạy sau tiếp tục từ chỗ dừng."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "logs" / "ch7_run.log"

for i in range(1, 31):
    with LOG.open("a", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "ch7_run.py"), *sys.argv[1:]], cwd=ROOT,
                           stdout=f, stderr=subprocess.STDOUT)
    if "\nXONG" in LOG.read_text(encoding="utf-8", errors="replace"):
        break
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"lần {i} kết thúc bất thường (mã {r.returncode}), chạy lại sau 60s\n")
    time.sleep(60)
