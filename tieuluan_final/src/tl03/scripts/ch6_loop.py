"""Chạy lại ch6_run.py cho tới khi in XONG: trên máy dùng chung tiến trình có lúc chết giữa chừng (thiếu bộ nhớ commit);
mỗi (bộ dữ liệu, mô hình) và mỗi phân tích đã xong được lưu nên lần chạy sau tiếp tục từ chỗ dừng."""
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "outputs" / "logs" / "ch6_run.log"

for i in range(1, 31):
    with LOG.open("a", encoding="utf-8") as f:
        r = subprocess.run([sys.executable, "-u", str(ROOT / "scripts" / "ch6_run.py")], cwd=ROOT,
                           stdout=f, stderr=subprocess.STDOUT)
    if "\nXONG" in LOG.read_text(encoding="utf-8", errors="replace"):
        break
    with LOG.open("a", encoding="utf-8") as f:
        f.write(f"lần {i} kết thúc bất thường (mã {r.returncode}), chạy lại sau 30s\n")
    time.sleep(30)
