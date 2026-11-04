"""
Prompt và schema đầu ra có cấu trúc cho các agent dùng LLM.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from src.document_processing.unit_normalizer import normalize, unit_flags
from src.schemas.patient_profile import PatientProfile

NumericField = Literal[
    "bmi", "height_cm", "weight_kg", "waist_cm",
    "glucose_fasting", "glucose_random", "ogtt_2h", "hba1c",
    "systolic_bp", "diastolic_bp",
    "total_cholesterol", "ldl", "hdl", "triglycerides", "creatinine", "egfr",
]
BooleanField = Literal[
    "high_bp", "high_chol", "family_history", "smoker", "phys_activity",
    "heart_disease", "stroke", "heavy_alcohol",
    "pregnant", "diagnosed_diabetes", "glucose_lowering_meds", "steroid_use", "acute_illness",
    "hba1c_interference", "fasting_confirmed",
]

NUMERIC_FIELD_MAP = {
    "glucose_fasting": "glucose_fasting_mgdl", "glucose_random": "glucose_random_mgdl",
    "ogtt_2h": "ogtt_2h_mgdl", "hba1c": "hba1c_percent",
    "total_cholesterol": "total_cholesterol_mgdl", "ldl": "ldl_mgdl", "hdl": "hdl_mgdl",
    "triglycerides": "triglycerides_mgdl", "creatinine": "creatinine_mgdl",
}


class Measurement(BaseModel):
    field: NumericField
    value: float
    unit: Optional[str] = Field(None, description="Đơn vị đúng như văn bản ghi, ví dụ mmol/L, mg/dL, %, mmol/mol, umol/L")
    evidence: str = Field(description="Trích nguyên văn đoạn chứa giá trị")
    timing_unclear: bool = Field(False, description="True nếu là glucose trong tin nhắn mà không nói rõ đo lúc đói hay sau ăn")


class RiskFactor(BaseModel):
    field: BooleanField
    present: bool
    evidence: str


class ExtractedClinicalData(BaseModel):
    measurements: List[Measurement] = Field(default_factory=list)
    risk_factors: List[RiskFactor] = Field(default_factory=list)
    age_years: Optional[int] = None
    birth_year: Optional[int] = None
    sex: Optional[Literal["male", "female"]] = None
    glucose_series: List[float] = Field(default_factory=list, description="Chuỗi các lần đo glucose theo thời gian, nếu có")
    glucose_series_unit: Optional[str] = None
    is_medical_document: bool = Field(True, description="False nếu tài liệu/ảnh không liên quan đến y tế")
    summary: Optional[str] = Field(None, description="Tóm tắt 1-2 câu nội dung tài liệu (chỉ với tài liệu)")

    def to_profile(self, source: str, base_confidence: float = 0.95) -> PatientProfile:
        from datetime import date
        profile = PatientProfile()
        for m in self.measurements:
            field = NUMERIC_FIELD_MAP.get(m.field, m.field)
            value, conf = normalize(field, m.value, m.unit)
            flags = unit_flags(field, m.unit) + (["fasting_unclear"] if m.timing_unclear and field == "glucose_fasting_mgdl" else [])
            profile.set(field, value, source, base_confidence * conf, m.evidence, flags)
        for r in self.risk_factors:
            profile.set(r.field, r.present, source, base_confidence * 0.95, r.evidence)
        if self.age_years:
            profile.set("age_years", self.age_years, source, base_confidence)
        elif self.birth_year:
            profile.set("age_years", date.today().year - self.birth_year, source, base_confidence * 0.95)
        if self.sex:
            profile.set("sex", self.sex, source, base_confidence)
        if len(self.glucose_series) >= 5:
            profile.glucose_history = [normalize("glucose_fasting_mgdl", v, self.glucose_series_unit)[0]
                                       for v in self.glucose_series]
            profile.glucose_history_source = source
        profile.derive()
        return profile


EXTRACTION_SYSTEM = """Bạn là bộ trích xuất dữ liệu lâm sàng cho hệ thống sàng lọc đái tháo đường.
Nhiệm vụ: đọc tin nhắn hoặc bệnh án tiếng Việt và trích ra các chỉ số của CHÍNH người bệnh.

Quy tắc:
- Chỉ trích những gì được ghi rõ. Không suy đoán, không điền giá trị mặc định. Trường nào không có thì bỏ qua.
- Giữ nguyên giá trị và đơn vị như văn bản ghi (không tự quy đổi). Nếu không ghi đơn vị thì để unit trống.
- "Đường huyết/đường máu/glucose" không nói rõ thời điểm: nếu là phiếu xét nghiệm thì coi là lúc đói; nếu là "sau ăn" hoặc "bất kỳ" thì là glucose_random; "2 giờ sau uống 75g glucose"/OGTT là ogtt_2h.
- Bỏ qua giá trị tham chiếu/khoảng bình thường in trên phiếu xét nghiệm, chỉ lấy kết quả của bệnh nhân.
- Yếu tố nguy cơ: present=false khi người dùng phủ định rõ ràng ("không hút thuốc", "nhà không ai bị tiểu đường").
- Tiền sử gia đình chỉ tính bố mẹ, anh chị em ruột hoặc con bị đái tháo đường.
- Người dùng hỏi về người khác (ví dụ "mẹ tôi") thì vẫn trích chỉ số của người được hỏi.
- Tin nhắn ghi đường huyết mà không nói lúc đói hay sau ăn: trích là glucose_fasting và đặt timing_unclear=true.
- Bối cảnh lâm sàng (trích như yếu tố nguy cơ, chỉ khi nói về CHÍNH người được đánh giá, không phải câu hỏi kiến thức):
  pregnant (đang mang thai), diagnosed_diabetes (đã được chẩn đoán đái tháo đường), glucose_lowering_meds (đang dùng
  thuốc hạ đường huyết/insulin), steroid_use (đang dùng corticoid), acute_illness (đang ốm cấp, sốt, nhiễm trùng, vừa
  phẫu thuật lúc xét nghiệm), hba1c_interference (thiếu máu, bệnh thận nặng/chạy thận, vừa truyền máu, thalassemia),
  fasting_confirmed (xác nhận lần đo glucose đã nhịn ăn ≥ 8 giờ; false nếu đo sau ăn).
- Nếu tin nhắn là câu trả lời ngắn ("có", "không", "rồi", "chưa") cho câu chatbot vừa hỏi, trích trường mà câu hỏi đó nhắm tới."""


class ScopeIssue(BaseModel):
    reason: Literal["PREGNANCY", "ALREADY_DIAGNOSED", "ON_DIABETES_MEDS", "PEDIATRIC", "HBA1C_UNRELIABLE",
                    "ACUTE_ILLNESS_OR_DRUG", "OTHER"]
    evidence: str = Field(description="Trích NGUYÊN VĂN đoạn trong tin nhắn của người dùng gây ra vấn đề")
    question: str = Field(description="Một câu hỏi ngắn, lịch sự để hỏi lại người dùng (tiếng Việt)")
    blocking: bool = Field(description="True nếu không thể đưa kết luận sàng lọc có ý nghĩa trước khi có câu trả lời")


class ScopeCheck(BaseModel):
    issues: List[ScopeIssue] = Field(default_factory=list)


SCOPE_SYSTEM = """Bạn là bộ kiểm tra phạm vi cho một chatbot SÀNG LỌC nguy cơ đái tháo đường típ 2.
Hệ thống chỉ đúng khi người được đánh giá: là người lớn, không mang thai, chưa được chẩn đoán đái tháo đường, không
dùng thuốc ảnh hưởng đường huyết, xét nghiệm làm lúc sức khỏe ổn định, và HbA1c phản ánh đúng đường huyết.
Hệ thống đã tự hỏi các chỉ số còn thiếu (tuổi, BMI, đường huyết, đơn vị...) — KHÔNG hỏi lại những chỉ số đó.

Nhiệm vụ: đọc tin nhắn MỚI của người dùng và tìm các tình huống MƠ HỒ hoặc NẰM NGOÀI các giả định trên mà cần hỏi lại,
ví dụ: nhắc tới thai nghén nhưng không rõ, nói "đang điều trị" mà không rõ bệnh gì, đánh giá cho trẻ em, xét nghiệm
làm khi đang nằm viện, triệu chứng cấp cứu, người dùng hỏi cho nhiều người khác nhau cùng lúc, dữ liệu tự mâu thuẫn.
- Dùng OTHER cho tình huống không thuộc nhóm nào nhưng vẫn khiến kết quả sàng lọc có thể sai.
- evidence phải là đoạn trích nguyên văn có trong tin nhắn. Không có bằng chứng thì không báo.
- Thông tin đã nói RÕ RÀNG (ví dụ "tôi không mang thai", "tôi đang tiêm insulin") thì không cần hỏi.
- Phần lớn tin nhắn KHÔNG có vấn đề gì: khi đó trả về danh sách rỗng. Tối đa 2 vấn đề."""


class IntentDecision(BaseModel):
    intent: Literal["RISK_ASSESSMENT", "FOLLOW_UP", "KNOWLEDGE_QUESTION", "SMALL_TALK"]
    rationale: str


INTENT_SYSTEM = """Phân loại ý định tin nhắn MỚI NHẤT của người dùng một chatbot sàng lọc đái tháo đường, dựa vào cả lịch sử hội thoại:
- RISK_ASSESSMENT: người dùng cung cấp chỉ số/triệu chứng/tiền sử MỚI của bản thân hoặc người thân, muốn được đánh giá nguy cơ, hoặc đang trả lời câu hỏi bổ sung thông tin của chatbot.
- FOLLOW_UP: câu hỏi tiếp nối cuộc trò chuyện mà không đưa thêm dữ liệu mới — hỏi về thông tin của chính mình đã khai ("tôi cao bao nhiêu?", "tôi đã cho bạn những chỉ số nào?"), hỏi lại / nhờ giải thích kết quả vừa đánh giá ("tại sao nguy cơ cao?", "vậy tôi cần làm gì?"), hoặc tham chiếu tới nội dung trước đó.
- KNOWLEDGE_QUESTION: câu hỏi kiến thức chung về đái tháo đường, xét nghiệm, thuốc, chế độ ăn... không gắn với hồ sơ cá nhân.
- SMALL_TALK: chào hỏi, cảm ơn, hỏi chatbot là ai, nội dung không liên quan y tế."""


CONTEXT_SYSTEM = """Bạn là Diabetes Prediction, trợ lý sàng lọc nguy cơ đái tháo đường, trả lời bằng tiếng Việt.
Người dùng đang hỏi tiếp trong cùng cuộc trò chuyện. Bạn được cung cấp: lịch sử hội thoại, hồ sơ sức khỏe đã ghi nhận
(kèm nguồn), danh sách chỉ số còn thiếu, và kết quả hội chẩn gần nhất (nếu có).
- Trả lời đúng câu hỏi, ngắn gọn, CHỈ dựa trên dữ liệu được cung cấp. Không bịa chỉ số.
- Nếu người dùng hỏi một chỉ số chưa từng khai (ví dụ chiều cao khi mới chỉ có BMI), nói rõ là chưa được cung cấp và mời họ bổ sung.
- Nếu hỏi về kết quả trước, giải thích dựa trên các bằng chứng (+)/(-) và khuyến cáo trong kết quả hội chẩn.
- Không lặp lại toàn bộ báo cáo đánh giá trừ khi người dùng yêu cầu."""


def format_history(history: List[dict], max_chars: int = 500) -> str:
    """Rút gọn lịch sử chat để đưa vào prompt (bộ nhớ hội thoại ngắn hạn)."""
    lines = []
    for h in history:
        who = "Người dùng" if h["sender"] == "user" else "Diabetes Prediction"
        text = h["text"].replace("\n", " ")
        lines.append(f"{who}: {text[:max_chars]}{'…' if len(text) > max_chars else ''}")
    return "\n".join(lines)


RESPONSE_SYSTEM = """Bạn là Diabetes Prediction, trợ lý sàng lọc nguy cơ đái tháo đường, trả lời bằng tiếng Việt.
Bạn nhận kết quả JSON từ hội đồng các tác tử chuyên môn. Hãy viết câu trả lời cho người dùng:
- CHỈ dùng số liệu và kết luận có trong JSON. Không tự thêm chẩn đoán, số liệu hay tên thuốc mới.
- Mở đầu bằng kết luận phân tầng nguy cơ; giải thích ngắn gọn lý do (các bằng chứng [+] và điểm cần lưu ý [-]).
- Nêu khuyến cáo kèm nguồn trích dẫn dạng [Nguồn: tên tài liệu] lấy từ trường citations.
- Nếu có gợi ý dùng thuốc, nói rõ đây là định hướng để trao đổi với bác sĩ, không tự ý dùng thuốc.
- Nếu còn thiếu dữ liệu (missing_recommended), gợi ý người dùng cung cấp thêm.
- Dùng Markdown gọn: tiêu đề nhỏ, gạch đầu dòng. Tối đa khoảng 250 từ.
- Kết thúc bằng lời nhắc: kết quả chỉ mang tính tham khảo, cần khám bác sĩ để chẩn đoán."""


KNOWLEDGE_SYSTEM = """Bạn là Diabetes Prediction, trợ lý kiến thức về đái tháo đường, trả lời bằng tiếng Việt.
Trả lời câu hỏi CHỈ dựa trên các đoạn tài liệu được cung cấp. Nếu tài liệu không đủ để trả lời, nói rõ là
cơ sở tri thức chưa có thông tin đó và khuyên người dùng hỏi bác sĩ. Trích dẫn nguồn dạng [Nguồn: tiêu đề].
Trả lời ngắn gọn bằng Markdown, tối đa khoảng 200 từ."""


SMALL_TALK_SYSTEM = """Bạn là Diabetes Prediction, chatbot sàng lọc nguy cơ đái tháo đường. Trả lời thân thiện, ngắn gọn bằng tiếng Việt.
Nếu phù hợp, gợi ý người dùng: nhập tuổi, chiều cao, cân nặng, đường huyết lúc đói hoặc HbA1c để đánh giá nguy cơ,
hoặc tải lên phiếu xét nghiệm/bệnh án (PDF, ảnh, DOCX)."""


TRIPLE_SYSTEM = """Trích các bộ ba tri thức (head, relation, tail) về đái tháo đường từ đoạn văn bản hướng dẫn lâm sàng.
Chỉ dùng các nhãn thực thể: RiskFactor, Disease, Complication, Test, Drug, Intervention.
Chỉ dùng các quan hệ: HAS_RISK_FACTOR, DIAGNOSED_BY, CAUSES, SCREENED_BY, TREATED_BY, CONTRAINDICATED_IN, PREVENTED_BY.
Tên thực thể ngắn gọn bằng tiếng Việt, viết hoa chữ cái đầu. evidence là câu trích nguyên văn."""


class Triple(BaseModel):
    head: str
    head_label: Literal["RiskFactor", "Disease", "Complication", "Test", "Drug", "Intervention"]
    relation: Literal["HAS_RISK_FACTOR", "DIAGNOSED_BY", "CAUSES", "SCREENED_BY", "TREATED_BY",
                      "CONTRAINDICATED_IN", "PREVENTED_BY"]
    tail: str
    tail_label: Literal["RiskFactor", "Disease", "Complication", "Test", "Drug", "Intervention"]
    evidence: str


class TripleList(BaseModel):
    triples: List[Triple] = Field(default_factory=list)
