#!/bin/sh
# Sau khi Chương 7A (trắc nghiệm tiểu đường) xong mới chạy đối chiếu GSM8K của nhóm (chung GPU 4 GB).
cd "$(dirname "$0")/.."
until grep -q XONG logs/ch7_medqa.log; do sleep 30; done
cd src/tl03 && exec ../../../.venv-gpu/bin/python -u scripts/ch7_run.py --n 250 --limit 100
