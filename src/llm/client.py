"""
Lớp trừu tượng LLM: các agent chỉ gọi `llm.parse(...)` / `llm.generate(...)`,
không phụ thuộc nhà cung cấp. Hỗ trợ Gemini (google-genai) và Claude (anthropic).

Mọi lỗi (thiếu key, mạng, từ chối, sai schema) đều trả về None để agent chuyển
sang nhánh dự phòng bằng luật — LLM không bao giờ là điểm chết của hệ thống.
"""

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import List, Optional, Type, TypeVar, Union

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


class LLMClient:
    def __init__(self):
        self.provider = settings.llm_provider
        self.model = settings.llm_model
        self.enabled = settings.llm_enabled
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic
                self._client = anthropic.AsyncAnthropic(timeout=settings.llm_timeout_s)
            else:
                from google import genai
                from google.genai import types
                # Tắt retry nội bộ của SDK: khi quá tải thì chuyển ngay sang model dự phòng
                self._client = genai.Client(http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(attempts=1)))
        return self._client

    # ------------------------------------------------------------------
    async def parse(self, system: str, parts: List[Part], schema: Type[T]) -> Optional[T]:
        """Sinh JSON theo schema Pydantic. Trả về None nếu không dùng được LLM."""
        if not self.enabled:
            return None
        try:
            if self.provider == "anthropic":
                return await asyncio.wait_for(self._parse_anthropic(system, parts, schema), settings.llm_timeout_s)
            return await self._parse_gemini(system, parts, schema)
        except Exception as exc:  # noqa: BLE001 — mọi lỗi LLM đều chuyển sang nhánh dự phòng
            log.warning("LLM parse thất bại (%s): %s", type(exc).__name__, exc)
            return None

    async def generate(self, system: str, parts: List[Part]) -> Optional[str]:
        """Sinh văn bản tự do. Trả về None nếu không dùng được LLM."""
        if not self.enabled:
            return None
        try:
            if self.provider == "anthropic":
                text = await asyncio.wait_for(self._generate_anthropic(system, parts), settings.llm_timeout_s)
            else:
                text = await self._generate_gemini(system, parts)
            return text.strip() if text else None
        except Exception as exc:  # noqa: BLE001
            log.warning("LLM generate thất bại (%s): %s", type(exc).__name__, exc)
            return None

    # ---------------------------- Gemini -------------------------------
    def _gemini_contents(self, parts: List[Part]):
        from google.genai import types
        return [types.Part.from_bytes(data=p.data, mime_type=p.mime_type) if isinstance(p, Blob)
                else types.Part.from_text(text=p) for p in parts]

    async def _gemini_call(self, parts: List[Part], config):
        """Gọi model chính; nếu quá tải, hết hạn mức, không khả dụng hoặc quá thời gian thì thử model dự phòng."""
        from google.genai import errors
        models = [self.model] + [m for m in settings.llm_fallback_models if m != self.model]
        for i, model in enumerate(models):
            try:
                return await asyncio.wait_for(self.client.aio.models.generate_content(
                    model=model, contents=self._gemini_contents(parts), config=config), settings.llm_timeout_s)
            except (errors.APIError, asyncio.TimeoutError) as exc:
                code = getattr(exc, "code", "timeout")
                if code not in (404, 429, 500, 503, "timeout") or i == len(models) - 1:
                    raise
                log.info("Gemini %s lỗi %s, chuyển sang %s", model, code, models[i + 1])

    async def _parse_gemini(self, system: str, parts: List[Part], schema: Type[T]) -> Optional[T]:
        from google.genai import types
        response = await self._gemini_call(parts, types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=schema,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        ))
        parsed = response.parsed
        if parsed is None and response.text:
            parsed = schema.model_validate_json(response.text)
        return parsed

    async def _generate_gemini(self, system: str, parts: List[Part]) -> Optional[str]:
        from google.genai import types
        response = await self._gemini_call(parts, types.GenerateContentConfig(
            system_instruction=system, thinking_config=types.ThinkingConfig(thinking_level="LOW")))
        return response.text

    # ---------------------------- Claude -------------------------------
    def _anthropic_content(self, parts: List[Part]) -> list:
        blocks = []
        for p in parts:
            if isinstance(p, str):
                blocks.append({"type": "text", "text": p})
                continue
            data = base64.standard_b64encode(p.data).decode("ascii")
            block_type = "document" if p.mime_type == "application/pdf" else "image"
            blocks.append({"type": block_type,
                           "source": {"type": "base64", "media_type": p.mime_type, "data": data}})
        # Tài liệu/ảnh đặt trước phần văn bản
        return sorted(blocks, key=lambda b: b["type"] == "text")

    async def _parse_anthropic(self, system: str, parts: List[Part], schema: Type[T]) -> Optional[T]:
        response = await self.client.beta.messages.parse(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": self._anthropic_content(parts)}],
            output_format=schema,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return None
        return response.parsed_output

    async def _generate_anthropic(self, system: str, parts: List[Part]) -> Optional[str]:
        response = await self.client.beta.messages.create(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=[{"role": "user", "content": self._anthropic_content(parts)}],
            output_config={"effort": "medium"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )
        if response.stop_reason == "refusal":
            return None
        return "".join(b.text for b in response.content if b.type == "text")


llm = LLMClient()
