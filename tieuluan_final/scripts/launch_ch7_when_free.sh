#!/bin/sh
# Chờ VRAM trống >= 3.3 GB (các job GPU khác xong) rồi mới nạp Qwen2.5-1.5B cho Chương 7.
cd "$(dirname "$0")/.."
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 3300 ]; do sleep 20; done
echo "$(date +%T) VRAM đủ, bắt đầu Chương 7"
exec ../.venv-gpu/bin/python -u scripts/run_ch7_medqa.py --n 150 --systems direct,cot,rag,multi
