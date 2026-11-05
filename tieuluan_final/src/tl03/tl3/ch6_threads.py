"""Giới hạn luồng CPU cho làn Ch.6/Ch.7 mà không nạp TensorFlow vào cùng tiến trình với PyTorch CUDA
(tl3.common.limit_threads có import tensorflow; nạp cả hai làm tiến trình nạp LLM bị segfault)."""
import os


def torch_threads(n):
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[k] = str(n)
    import torch
    torch.set_num_threads(n)
