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


CONTEXT_SYSTEM = """Bạn là Diabetes Prediction, trợ lý y khoa chuyên sâu về sàng lọc và tư vấn đái tháo đường, trả lời bằng tiếng Việt.
Người dùng đang hỏi tiếp trong cùng cuộc trò chuyện. Bạn được cung cấp: lịch sử hội thoại, hồ sơ sức khỏe đã ghi nhận
(kèm nguồn), danh sách chỉ số còn thiếu, và kết quả hội chẩn gần nhất (nếu có).
- Phân tích và suy luận kỹ lưỡng dựa trên dữ liệu đã cung cấp trong hồ sơ bệnh nhân và kết quả hội chẩn trước đó. Không bịa đặt số liệu ngoài JSON.
- Nếu người dùng hỏi về một chỉ số chưa từng khai, nêu rõ chỉ số đó chưa có trong hồ sơ, giải thích vì sao chỉ số đó quan trọng trong đánh giá nguy cơ đái tháo đường và mời họ bổ sung.
- Nếu người dùng hỏi giải thích thêm về kết quả hội chẩn, cơ chế bệnh sinh, ý nghĩa xét nghiệm hoặc thuốc điều trị, hãy phân tích chi tiết từng luận điểm (+)/(-), mối liên hệ nhân quả lâm sàng và hướng xử trí cụ thể."""


def format_history(history: List[dict], max_chars: int = 500) -> str:
    """Rút gọn lịch sử chat để đưa vào prompt (bộ nhớ hội thoại ngắn hạn)."""
    lines = []
    for h in history:
        who = "Người dùng" if h["sender"] == "user" else "Diabetes Prediction"
        text = h["text"].replace("\n", " ")
        lines.append(f"{who}: {text[:max_chars]}{'…' if len(text) > max_chars else ''}")
    return "\n".join(lines)


RESPONSE_SYSTEM = """Bạn là Bác sĩ Nội tiết & Hội đồng AI Chuyên gia của hệ thống Diabetes Prediction, trả lời bằng tiếng Việt.
Bạn được cung cấp Bản báo cáo hội chẩn 4 phần đã được xác thực từ 8 tác tử chuyên môn.
Nhiệm vụ của bạn là viết tiếp phần **Biện luận lâm sàng chuyên sâu (khoảng 180–280 từ)** cho bệnh nhân này:
- Giải thích cơ chế bệnh sinh liên kết các yếu tố của chính người bệnh (ví dụ: vì sao thừa cân/béo phì theo chuẩn châu Á BMI ≥ 23 kg/m² gây đề kháng insulin tại mô cơ và gan; vì sao tiền sử gia đình kết hợp tuổi tác làm suy giảm chức năng tế bào beta đảo tụy; ý nghĩa của chỉ số đường huyết lúc đói / HbA1c hiện tại).
- Giải thích vì sao phác đồ can thiệp lối sống (giảm 5–7% cân nặng, vận động ≥ 150 phút/tuần) và nhóm thuốc tham chiếu trong báo cáo lại tác động trúng đích vào cơ chế bệnh sinh đó.
- Dặn dò cụ thể những câu hỏi người bệnh nên trao đổi trực tiếp với bác sĩ chuyên khoa Nội tiết khi đi khám và các dấu hiệu cảnh báo cần đi khám sớm.

Quy tắc bắt buộc:
- CHỈ phân tích dựa trên đúng các chỉ số có trong báo cáo được cung cấp. Tuyệt đối KHÔNG bịa thêm chỉ số xét nghiệm mà bệnh nhân chưa làm (như HbA1c, huyết áp nếu trong báo cáo không ghi).
- Viết thành các đoạn văn hoặc gạch đầu dòng mạch lạc, sâu sắc, dễ hiểu với người bệnh; KHÔNG lặp lại nguyên văn các tiêu đề mục 1, 2, 3, 4 đã có ở trên.
- Không dùng cú pháp LaTeX hay ký hiệu `$`; viết trực tiếp `≥, ≤, ×, →, %`."""


KNOWLEDGE_SYSTEM = """Bạn là Diabetes Prediction — Chuyên gia Tri thức Y khoa về Đái tháo đường (ADA 2024, Bộ Y tế QĐ 5481/QĐ-BYT, Vinmec, Dược điển FPT Long Châu), trả lời bằng tiếng Việt.
Hãy suy luận kỹ lưỡng và trả lời chi tiết, đầy đủ câu hỏi của người dùng dựa trên các đoạn tài liệu (Vector RAG) và các đường đi trên Đồ thị Tri thức (Neo4j GraphRAG) được cung cấp:
- Giải thích rõ bản chất y khoa, cơ chế tác dụng, ngưỡng tiêu chuẩn chẩn đoán, chỉ định điều trị, cách dùng thực tế và các chống chỉ định / tác dụng phụ cần lưu ý.
- Kết hợp thông tin từ cả đoạn văn bản và chuỗi quan hệ đồ thị (`Quan hệ từ Đồ thị Tri thức`) để câu trả lời có tính hệ thống và lập luận chặt chẽ.
- Trình bày bằng Markdown rõ ràng (tiêu đề mục, gạch đầu dòng, in đậm các ý quan trọng).
- Trích dẫn nguồn rõ ràng dạng `[Nguồn: tiêu đề tài liệu]` ở các luận điểm chính.
- Không dùng LaTeX hay ký hiệu `$`; viết thẳng `≥, ≤, ×, →`."""


SMALL_TALK_SYSTEM = """Bạn là Diabetes Prediction, trợ lý AI chuyên sâu sàng lọc nguy cơ và tư vấn đái tháo đường. Trả lời thân thiện, rõ ràng bằng tiếng Việt.
Hướng dẫn cụ thể cho người dùng các cách sử dụng hệ thống:
- Nhập các chỉ số sức khỏe (tuổi, giới tính, chiều cao, cân nặng, vòng eo, huyết áp, đường huyết lúc đói, HbA1c, tiền sử gia đình...) để Hội đồng 8 tác tử AI phân tích chuyên sâu.
- Tải lên phiếu xét nghiệm máu hoặc sổ khám bệnh (PDF, ảnh chụp, DOCX, CSV).
- Đặt câu hỏi chuyên sâu về tiêu chuẩn chẩn đoán ADA 2024 / Bộ Y tế, phác đồ điều trị Vinmec hoặc thông tin thuốc đái tháo đường tại Nhà thuốc Long Châu."""


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
