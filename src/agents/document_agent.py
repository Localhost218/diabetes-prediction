"""
Tác tử Đọc Bệnh án (Document / Medical Record Intake Agent)
Đọc file người dùng tải lên từ SeaweedFS (S3): PDF, PDF scan, ảnh chụp phiếu xét nghiệm, DOCX, CSV chuỗi glucose
→ trích chỉ số (kết hợp LLM Ollama/Qwen + Regex Rules bù khuyết) → chuẩn hóa đơn vị → PatientProfile có nguồn "file:<tên file>".
"""

import asyncio

from src.communication.message_bus import AgentMessage, BaseAgent
from src.document_processing import loaders
from src.document_processing.lab_extractor import extract_profile
from src.llm.client import Blob, llm
from src.llm.prompts import EXTRACTION_SYSTEM, ExtractedClinicalData
from src.storage.object_storage import get_storage

DOC_INSTRUCTION = (
    "Đây là tài liệu y tế người dùng tải lên ({filename}). Trích các chỉ số xét nghiệm và yếu tố nguy cơ "
    "của bệnh nhân, tóm tắt ngắn nội dung tài liệu. Nếu tài liệu không phải tài liệu y tế, đặt is_medical_document=false."
)


class DocumentAgent(BaseAgent):
    name = "DocumentAgent"
    handles = {"PARSE_DOCUMENT"}

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        p = msg.payload
        filename, mime = p["filename"], p["mime_type"]
        source = f"file:{filename}"
        data = await asyncio.to_thread(get_storage().get, p["bucket"], p["object_key"])
        doc = await asyncio.to_thread(loaders.load, data, filename, mime)

        text = doc.text or ""
        ocr_used = False
        if doc.images and not text.strip() and loaders.tesseract_available():
            text = await asyncio.to_thread(loaders.ocr_images, doc.images)
            ocr_used = True

        profile, method, summary, is_medical = None, None, None, True
        llm_parts = self._llm_parts(doc, text, filename)
        if llm_parts is not None:
            extracted = await llm.parse(EXTRACTION_SYSTEM, llm_parts, ExtractedClinicalData)
            if extracted is not None:
                profile, method = extracted.to_profile(source), "llm"
                summary, is_medical = extracted.summary, extracted.is_medical_document

        rules_profile = extract_profile(text, source) if text.strip() else None
        if profile is None:
            if rules_profile is None:
                return self.reply(msg, "DOCUMENT_UNREADABLE", {
                    "error": "Ảnh/PDF scan cần LLM vision hoặc cài đặt Tesseract OCR để đọc.",
                    "kind": doc.kind,
                }, performative="FAILURE")
            profile = rules_profile
            method = "ocr+rules" if ocr_used else "rules"
        elif rules_profile is not None:
            filled = [n for n in rules_profile.fields if n not in profile.fields]
            for name in filled:
                profile.fields[name] = rules_profile.fields[name]
            if filled:
                method = "llm+rules"
                profile.derive()

        if doc.glucose_series and not profile.glucose_history:
            profile.glucose_history, profile.glucose_history_source = doc.glucose_series, source

        return self.reply(msg, "DOCUMENT_PARSED", {
            "document_id": p["document_id"],
            "filename": filename,
            "kind": doc.kind,
            "pages": doc.pages,
            "method": method,
            "is_medical_document": is_medical,
            "summary": summary,
            "notes": doc.notes,
            "fields": sorted(profile.fields),
            "profile_delta": profile.model_dump(),
        })

    @staticmethod
    def _llm_parts(doc: loaders.LoadedDocument, extracted_text: str, filename: str):
        if not llm.enabled:
            return None
        instruction = DOC_INSTRUCTION.format(filename=filename)
        if extracted_text.strip():
            return [f"{instruction}\n\n--- Nội dung tài liệu ---\n{extracted_text[:30000]}"]
        if doc.kind in ("image", "pdf_scanned") and doc.images:
            return [Blob(doc.images[0], doc.image_mime or "image/png"), instruction]
        return None
