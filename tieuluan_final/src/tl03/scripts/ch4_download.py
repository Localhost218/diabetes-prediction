"""Tải ba bộ dữ liệu của Chương 4 vào data/ch4/ (kagglehub ẩn danh)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl3.ch4_data import download_all  # noqa: E402

if __name__ == "__main__":
    download_all()
