"""Nạp Qwen2.5-1.5B lên GPU với ít bộ nhớ commit nhất (dùng cho Ch.7 khi nhiều làn chạy song song).

Cách nạp mặc định ánh xạ cả tệp safetensors 3,1 GB kiểu copy-on-write rồi dựng thêm một bản trên CPU: cần ~6,5 GB
commit, máy thiếu commit thì tiến trình segfault. Ở đây: dựng mô hình trên thiết bị 'meta' (không cấp phát), đọc
từng tensor bằng đọc tệp thường theo offset trong header rồi chuyển thẳng lên GPU — đỉnh commit chỉ ~1 tensor.
"""
import json
import struct
from pathlib import Path

import torch

from .common import LLM_NAME, hf_path
from .llm import LocalLLM

DTYPES = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32, "I64": torch.int64}


def read_safetensors(path, device="cuda"):
    """Đọc lần lượt từng tensor của một tệp .safetensors (8 byte độ dài header + JSON + dữ liệu thô)."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            f.seek(base + a)
            buf = bytearray(f.read(b - a))
            t = torch.frombuffer(buf, dtype=DTYPES[meta["dtype"]]).reshape(meta["shape"])
            out[name] = t.to(device, dtype=torch.bfloat16 if t.is_floating_point() else t.dtype)
            del buf, t
    return out


def load_lowmem(device="cuda"):
    """Trả LocalLLM dùng chung bộ nhớ đệm: mọi LocalLLM() sau đó dùng mô hình đã nạp theo cách này."""
    if LocalLLM._cache is None:
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
        path = Path(hf_path(LLM_NAME))
        cfg = AutoConfig.from_pretrained(path)
        with torch.device("meta"):
            model = AutoModelForCausalLM.from_config(cfg, dtype=torch.bfloat16)
        sd = {}
        for p in sorted(path.glob("*.safetensors")):
            sd.update(read_safetensors(p, device))
        missing, _ = model.load_state_dict(sd, strict=False, assign=True)
        if getattr(cfg, "tie_word_embeddings", False):
            model.lm_head.weight = model.model.embed_tokens.weight
            missing = [m for m in missing if m != "lm_head.weight"]
        if missing:
            raise RuntimeError(f"thiếu trọng số: {missing[:5]}")
        # bộ đệm không lưu trong tệp (tần số RoPE) vẫn ở 'meta': dựng lại trên GPU
        rot = model.model.rotary_emb
        try:
            model.model.rotary_emb = type(rot)(config=cfg, device=device)
        except TypeError:                     # transformers mới hơn: không còn tham số device
            with torch.device(device):
                model.model.rotary_emb = type(rot)(config=cfg)
        left = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
        if left:
            raise RuntimeError(f"còn tensor trên meta: {left[:5]}")
        LocalLLM._cache = (AutoTokenizer.from_pretrained(path), model.eval())
    return LocalLLM()
