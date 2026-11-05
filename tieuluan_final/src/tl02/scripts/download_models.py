"""Tải các mô hình HuggingFace vào models/hf/ (thư mục thường, không dùng symlink của bộ đệm HF trên Windows).

Sau khi tải, ghi thêm bản bf16 của reranker (bge-reranker-v2-m3-bf16). Trên máy chạy GPU qua WDDM, mỗi byte trên GPU
cũng bị tính vào bộ nhớ commit của Windows; nạp bản fp32 2,3 GB để ép về bf16 tạo đỉnh commit đủ làm sập tiến trình
khi LLM đã nằm trên GPU. Bản bf16 trên đĩa cho kết quả y hệt (trước đây cũng ép bf16 lúc nạp) mà đỉnh chỉ còn một nửa.
"""
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tl2.common import HF, RERANKER  # noqa: E402

REPOS = ["intfloat/multilingual-e5-small", "intfloat/multilingual-e5-base",
         "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "BAAI/bge-reranker-v2-m3",
         "Qwen/Qwen2.5-1.5B-Instruct"]
PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "sentencepiece*", "1_Pooling/*"]


def bf16_copy(src_name, dst_name):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    dst = HF / dst_name
    if (dst / "model.safetensors").exists():
        return
    model = AutoModelForSequenceClassification.from_pretrained(HF / src_name, dtype=torch.bfloat16)
    model.save_pretrained(dst)
    AutoTokenizer.from_pretrained(HF / src_name).save_pretrained(dst)
    print("đã ghi", dst_name)


if __name__ == "__main__":
    for repo in REPOS:
        snapshot_download(repo, local_dir=HF / repo.split("/")[1], allow_patterns=PATTERNS)
        print("đã tải", repo)
    bf16_copy("bge-reranker-v2-m3", RERANKER)
