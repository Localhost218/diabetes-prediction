"""Mô hình ngôn ngữ chạy cục bộ trên GPU (Qwen2.5-1.5B-Instruct, đã tải ở Tiểu luận 02).

Không gọi API bên ngoài: câu hỏi, bằng chứng và câu trả lời đều nằm trên máy. Dùng cho bước sinh của RAG (Ch.5)
và các tác tử (Ch.7). Lần sinh đầu tiên cần nằm trong `with tl3.gpu.heavy(...)` vì mô hình chiếm ~3,5 GB VRAM.
"""
import time

from .common import LLM_NAME, device, hf_path

SYSTEM_GROUNDED = (
    "Bạn là trợ lý tiếng Việt của một hệ thống RAG. Chỉ được dùng thông tin trong phần BẰNG CHỨNG. "
    "Mỗi ý quan trọng phải kèm số hiệu bằng chứng trong ngoặc vuông, ví dụ [1]. "
    "Nếu bằng chứng không đủ để trả lời, nói rõ 'không đủ thông tin'. Không bịa số liệu. Trả lời ngắn gọn, mạch lạc."
)


class LocalLLM:
    """Bọc tokenizer + mô hình; nạp một lần cho cả tiến trình."""
    _cache = None

    def __init__(self, max_new_tokens=320):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if LocalLLM._cache is None:
            path = hf_path(LLM_NAME)
            tok = AutoTokenizer.from_pretrained(path)
            # bf16 và nạp lên CPU rồi mới chuyển: trên máy này fp16 làm cuBLASLt báo CUBLAS_STATUS_NOT_SUPPORTED,
            # còn device_map="cuda" gây segfault khi đọc safetensors
            model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(device()).eval()
            LocalLLM._cache = (tok, model)
        self.tok, self.model = LocalLLM._cache
        self.max_new_tokens = max_new_tokens

    def chat(self, user, system=SYSTEM_GROUNDED, max_new_tokens=None, history=None, temperature=0.0):
        """Sinh một lượt; mặc định tham lam (greedy) để tái lập. Trả (văn bản, số token sinh, giây)."""
        import torch
        msgs = [{"role": "system", "content": system}, *(history or []), {"role": "user", "content": user}]
        ids = self.tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(self.model.device) for k, v in ids.items()}
        sampling = dict(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else dict(do_sample=False)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(**ids, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                      repetition_penalty=1.05, pad_token_id=self.tok.eos_token_id, **sampling)
        new = out[0, ids["input_ids"].shape[1]:]
        return self.tok.decode(new, skip_special_tokens=True).strip(), int(new.shape[0]), time.perf_counter() - t0


def grounded_prompt(question, evidence):
    """evidence: list[(nhãn nguồn, văn bản)] -> prompt đánh số [1]..[n]."""
    lines = [f"[{i}] ({src}) {txt}" for i, (src, txt) in enumerate(evidence, 1)]
    return "BẰNG CHỨNG:\n" + "\n".join(lines) + f"\n\nCÂU HỎI: {question}\n\nTRẢ LỜI (tiếng Việt, có trích dẫn [n]):"
