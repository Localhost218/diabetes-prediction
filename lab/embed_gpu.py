"""
Tính nhúng multilingual-E5-large cho MedQuAD trên GPU (sentence-transformers) và ghi vào cache của
lab/ch5_rag.py — cùng mô hình, cùng tiền tố "query: "/"passage: ", chuẩn hóa L2.

    .venv-gpu/bin/python lab/embed_gpu.py
"""

import numpy as np
from sentence_transformers import SentenceTransformer

from ch5_rag import CACHE, chunk, load_medquad


def main():
    qa = load_medquad()
    texts = [f"{x['focus']}. {c}" for x in qa for c in chunk(x["a"], 800)]
    model = SentenceTransformer("intfloat/multilingual-e5-large", device="cuda")
    model.half()
    for name, items, prefix in [("mq800_e5_p", texts, "passage: "), ("mqq_e5_q", [x["q"] for x in qa], "query: ")]:
        v = model.encode([prefix + t for t in items], batch_size=32, normalize_embeddings=True,
                         show_progress_bar=False).astype(np.float32)
        np.save(CACHE / f"{name}.npy", v)
        print(name, v.shape)


if __name__ == "__main__":
    main()
