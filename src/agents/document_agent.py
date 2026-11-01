"""
Tác tử Đọc Bệnh án (Document / Medical Record Intake Agent)
Đọc file người dùng tải lên từ SeaweedFS (S3): PDF, PDF scan, ảnh chụp phiếu xét nghiệm, DOCX, CSV chuỗi glucose
→ trích chỉ số → chuẩn hóa đơn vị → PatientProfile có nguồn "file:<tên file>".
"""

import asyncio

from src.communication.message_bus import AgentMessage, BaseAgent
from src.document_processing import loaders
from src.document_processing.lab_extractor import extract_profile
from src.llm.client import Blob, llm
from src.llm.prompts import EXTRACTION_SYSTEM, ExtractedClinicalData
from src.schemas.patient_profile import PatientProfile
from src.storage.object_storage import get_storage

DOC_INSTRUCTION = ("Đây là tài liệu y tế người dùng tải lên ({filename}). Trích các chỉ số xét nghiệm và yếu tố nguy cơ "
                   "của bệnh nhân, tóm tắt ngắn nội dung tài liệu. Nếu tài liệu không phải tài liệu y tế, đặt is_medical_document=false.")


class DocumentAgent(BaseAgent):
    name = "DocumentAgent"
    handles = {"PARSE_DOCUMENT"}

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        p = msg.payload
        filename, mime = p["filename"], p["mime_type"]
        source = f"file:{filename}"
        data = await asyncio.to_thread(get_storage().get, p["bucket"], p["object_key"])
        doc = await asyncio.to_thread(loaders.load, data, filename, mime)

        profile, method, summary, is_medical = None, None, None, True
        llm_parts = self._llm_parts(doc, data, filename)
        if llm_parts is not None:
            extracted = await llm.parse(EXTRACTION_SYSTEM, llm_parts, ExtractedClinicalData)
            if extracted is not None:
                profile, method = extracted.to_profile(source), "llm"
                summary, is_medical = extracted.summary, extracted.is_medical_document

        if profile is None:
            text = doc.text
            if doc.images and not text.strip():
                if not loaders.tesseract_available():
                    return self.reply(msg, "DOCUMENT_UNREADABLE", {
                        "error": "Ảnh/PDF scan cần LLM vision (cấu hình API key) hoặc Tesseract OCR để đọc.",
                        "kind": doc.kind,
                    }, performative="FAILURE")
                text = await asyncio.to_thread(loaders.ocr_images, doc.images)
                method = "ocr+rules"
            profile, method = extract_profile(text, source), method or "rules"

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
    def _llm_parts(doc: loaders.LoadedDocument, data: bytes, filename: str):
        if not llm.enabled:
            return None
        instruction = DOC_INSTRUCTION.format(filename=filename)
        if doc.kind in ("pdf_text", "pdf_scanned"):
            return [Blob(data, "application/pdf"), instruction]  # LLM đọc được cả bố cục bảng xét nghiệm
        if doc.kind == "image":
            return [Blob(doc.images[0], doc.image_mime), instruction]
        return [f"{instruction}\n\n--- Nội dung tài liệu ---\n{doc.text[:30000]}"]
