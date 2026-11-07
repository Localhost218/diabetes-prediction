"""
Lớp trừu tượng LLM: các agent chỉ gọi `llm.parse(...)` / `llm.generate(...)`,
không phụ thuộc nhà cung cấp. Hỗ trợ 2 chế độ nội bộ ổn định:
  - Ollama (`LLM_PROVIDER=ollama`, mặc định `qwen2.5:3b` / `qwen2.5:1.5b` qua HTTP API)
  - Local HuggingFace Qwen (`LLM_PROVIDER=qwen`, nạp trực tiếp `data/output/storage/hf/Qwen2.5-1.5B-Instruct`)

Nếu `LLM_PROVIDER=ollama` mà Ollama daemon chưa bật, client tự động fallback sang
bản `Qwen2.5-1.5B-Instruct` cục bộ trên đĩa (nếu có `transformers`), và nếu vẫn
không khả dụng thì trả về `None` để từng agent chuyển sang nhánh dự phòng bằng luật (deterministic fallback).
"""

import asyncio
import base64
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Type, TypeVar, Union

import httpx
from pydantic import BaseModel

from src.config import settings

log = logging.getLogger(__name__)
T = TypeVar("T", bound=BaseModel)


@dataclass
class Blob:
    """Dữ liệu nhị phân gửi kèm prompt (ảnh, PDF, audio)."""
    data: bytes
    mime_type: str


Part = Union[str, Blob]


def _extract_json_str(raw: str) -> str:
    """Tách khối JSON hợp lệ đầu tiên từ đầu ra văn bản của mô hình."""
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, flags=re.DOTALL)
    if fence:
        return fence.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


class LLMClient:
    _hf_cache = None

    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.enabled = settings.llm_enabled

    # ------------------------------------------------------------------
    async def parse(self, system: str, parts: List[Part], schema: Type[T]) -> Optional[T]:
        """Sinh JSON theo schema Pydantic. Trả về None nếu không dùng được LLM."""
        if not self.enabled:
            return None
        try:
            if self.provider in ("qwen", "local", "hf"):
                import torch
                if not torch.cuda.is_available():
                    return None
                return await asyncio.wait_for(
                    asyncio.to_thread(self._parse_qwen_local, system, parts, schema),
                    settings.llm_timeout_s,
                )
            return await self._parse_ollama_with_fallback(system, parts, schema)
        except Exception as exc:  # noqa: BLE001 — mọi lỗi LLM đều chuyển sang nhánh dự phòng
            log.warning("LLM parse thất bại (%s): %s", type(exc).__name__, exc)
            return None

    async def generate(self, system: str, parts: List[Part]) -> Optional[str]:
        """Sinh văn bản tự do. Trả về None nếu không dùng được LLM."""
        if not self.enabled:
            return None
        try:
            if self.provider in ("qwen", "local", "hf"):
                text = await asyncio.wait_for(
                    asyncio.to_thread(self._generate_qwen_local, system, parts),
                    settings.llm_timeout_s,
                )
            else:
                text = await self._generate_ollama_with_fallback(system, parts)
            return text.strip() if text else None
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM generate thất bại (%s): %s", type(exc).__name__, exc)
            return None

    # ---------------------------- Ollama -------------------------------
    @staticmethod
    def _ollama_user_message(parts: List[Part]) -> dict:
        texts: List[str] = []
        images: List[str] = []
        for p in parts:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, Blob) and p.mime_type.startswith("image/"):
                images.append(base64.standard_b64encode(p.data).decode("ascii"))
        msg = {"role": "user", "content": "\n\n".join(texts)}
        if images:
            msg["images"] = images
        return msg

    async def _ollama_chat(self, model: str, system: str, parts: List[Part], fmt=None) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                self._ollama_user_message(parts),
            ],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 1024},
        }
        if fmt is not None:
            payload["format"] = fmt
        url = f"{settings.ollama_base_url.rstrip('/')}/api/chat"
        async with httpx.AsyncClient(timeout=settings.llm_timeout_s) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return (data.get("message") or {}).get("content", "")

    async def _parse_ollama_with_fallback(self, system: str, parts: List[Part], schema: Type[T]) -> Optional[T]:
        # Nếu có dữ liệu nhị phân không phải ảnh (PDF/audio), chuyển cho nhánh dự phòng nếu model không hỗ trợ
        if any(isinstance(p, Blob) and not p.mime_type.startswith("image/") for p in parts):
            return None
        models = [self.model] + [m for m in settings.llm_fallback_models if m != self.model]
        json_schema = schema.model_json_schema()
        for model in models:
            try:
                raw = await self._ollama_chat(model, system, parts, fmt=json_schema)
                if raw:
                    return schema.model_validate_json(_extract_json_str(raw))
            except Exception as exc:  # noqa: BLE001
                log.debug("Ollama model %s không khả dụng: %s", model, exc)
        # Fallback sang bản Qwen2.5-1.5B-Instruct local trên đĩa khi có GPU (trên CPU ưu tiên regex nhanh cho parse,
        # dành toàn bộ tài nguyên Qwen local cho bước generate() lập luận câu trả lời cuối cùng)
        if self._can_use_qwen_local():
            import torch
            if torch.cuda.is_available():
                return await asyncio.to_thread(self._parse_qwen_local, system, parts, schema)
        return None

    async def _generate_ollama_with_fallback(self, system: str, parts: List[Part]) -> Optional[str]:
        if any(isinstance(p, Blob) and not p.mime_type.startswith("image/") for p in parts):
            return None
        models = [self.model] + [m for m in settings.llm_fallback_models if m != self.model]
        for model in models:
            try:
                text = await self._ollama_chat(model, system, parts)
                if text:
                    return text
            except Exception as exc:  # noqa: BLE001
                log.debug("Ollama model %s không khả dụng: %s", model, exc)
        if self._can_use_qwen_local():
            import torch
            if torch.cuda.is_available():
                return await asyncio.to_thread(self._generate_qwen_local, system, parts)
        return None

    # --------------------- Local HuggingFace Qwen ----------------------
    @staticmethod
    def _can_use_qwen_local() -> bool:
        if not settings.qwen_local_path.exists():
            return False
        try:
            import transformers  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_hf_model(self):
        if LLMClient._hf_cache is None:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            model_path = Path(self.model) if Path(self.model).exists() else settings.qwen_local_path
            tok = AutoTokenizer.from_pretrained(str(model_path))
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            dtype = torch.bfloat16 if dev == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(str(model_path), dtype=dtype).to(dev).eval()
            LLMClient._hf_cache = (tok, model)
            log.info("Đã nạp Qwen local từ %s trên %s", model_path, dev)
        return LLMClient._hf_cache

    def _generate_qwen_local(self, system: str, parts: List[Part], max_new_tokens: int = 450) -> Optional[str]:
        if any(isinstance(p, Blob) for p in parts):
            return None
        import torch

        tok, model = self._get_hf_model()
        user_text = "\n\n".join(p for p in parts if isinstance(p, str))
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user_text}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
        ids = {k: v.to(model.device) for k, v in ids.items()}
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = out[0, ids["input_ids"].shape[1] :]
        return tok.decode(new_tokens, skip_special_tokens=True).strip()

    def _parse_qwen_local(self, system: str, parts: List[Part], schema: Type[T]) -> Optional[T]:
        if any(isinstance(p, Blob) for p in parts):
            return None
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        sys_with_schema = (
            f"{system}\n\nChỉ trả về duy nhất một đối tượng JSON hợp lệ khớp với JSON Schema sau "
            f"(không giải thích thêm):\n{schema_json}"
        )
        raw = self._generate_qwen_local(sys_with_schema, parts, max_new_tokens=512)
        if not raw:
            return None
        return schema.model_validate_json(_extract_json_str(raw))


llm = LLMClient()
