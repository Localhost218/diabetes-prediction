"""
Tác tử Phản hồi (Response Agent)
Viết câu trả lời tiếng Việt từ kết quả của các tác tử. Với LLM: chỉ được dùng dữ liệu trong JSON đầu vào
(grounded generation). Không có LLM: dùng mẫu câu có cấu trúc.

VẤN ĐÁP — VỊ TRÍ TRONG LUỒNG:
  Nhận: COMPOSE_ASSESSMENT | COMPOSE_KNOWLEDGE_ANSWER | COMPOSE_CONTEXTUAL_ANSWER | COMPOSE_FOLLOW_UP |
        COMPOSE_CLARIFICATION | COMPOSE_SMALL_TALK — luôn là bước cuối của lượt, do Orchestrator gửi
  Trả:  RESPONSE_TEXT {text} → routes_chat lưu tin nhắn và đẩy sự kiện "final" lên giao diện
  Đánh giá nguy cơ (_assessment_sections): 1) khả năng mắc (%), 2) vì sao (lời thường), 3) nên làm gì;
  LLM chỉ viết đoạn "Lời bác sĩ AI" ngắn, phần còn lại sinh từ mẫu để không bịa số liệu.
  Tác tử loại: phản xạ (reflex) — ánh xạ kết quả các tác tử thành câu trả lời.
"""

import re
import json
from typing import Any, Dict, List

from src.communication.message_bus import AgentMessage, BaseAgent
from src.llm.client import llm
from src.llm.prompts import CONTEXT_SYSTEM, KNOWLEDGE_SYSTEM, RESPONSE_SYSTEM, SMALL_TALK_SYSTEM, format_history
from src.schemas.patient_profile import FIELD_SPECS

# Từ khóa trong câu hỏi → trường hồ sơ (dùng cho mẫu trả lời khi không có LLM)
FIELD_KEYWORDS = [
    (r"chiều cao|cao bao nhiêu|cao mấy", "Chiều cao"), (r"cân nặng|nặng bao nhiêu|nặng mấy", "Cân nặng"),
    (r"tuổi", "Tuổi"),
    (r"bmi", "BMI"), (r"hba1c|a1c", "HbA1c"), (r"đường huyết|glucose|đường máu", "Đường huyết"),
    (r"huyết áp", "Huyết áp"), (r"vòng eo|eo", "Vòng eo"), (r"mỡ máu|cholesterol", "mỡ"),
]

DISCLAIMER = "_Kết quả chỉ mang tính tham khảo, không thay thế chẩn đoán của bác sĩ._"


class ResponseAgent(BaseAgent):
    name = "ResponseAgent"
    handles = {"COMPOSE_ASSESSMENT", "COMPOSE_FOLLOW_UP", "COMPOSE_KNOWLEDGE_ANSWER", "COMPOSE_SMALL_TALK",
               "COMPOSE_CONTEXTUAL_ANSWER", "COMPOSE_CLARIFICATION"}

    async def handle(self, msg: AgentMessage) -> AgentMessage:
        # ĐIỂM VÀO: COMPOSE_* — bước cuối của mọi lượt
        p = msg.payload
        history = _history_block(p.get("history"))
        if msg.msg_type == "COMPOSE_ASSESSMENT":
            head, advice = _assessment_sections(p)
            # GPU laptop ~19 token/s: lời bác sĩ ngắn (~300 token, ~15 s); phần còn lại đã có sẵn từ mẫu
            # Chỉ đưa phần tỷ lệ + lý do để lời bác sĩ ngắn, không chép lại danh sách lời khuyên
            note = await llm.generate(RESPONSE_SYSTEM, [history + head], max_tokens=220)
            note = _whole_sentences(_strip_cjk(note)) if note else ""
            text = head + (f"\n\n**👩‍⚕️ Lời bác sĩ AI:** {note}" if note else "") + "\n\n" + advice
        elif msg.msg_type == "COMPOSE_FOLLOW_UP":
            text = _follow_up_template(p)
        elif msg.msg_type == "COMPOSE_CLARIFICATION":
            text = _clarification_template(p)
        elif msg.msg_type == "COMPOSE_CONTEXTUAL_ANSWER":
            text = await llm.generate(CONTEXT_SYSTEM, [history + _contextual_brief(p)]) or _contextual_template(p)
        elif msg.msg_type == "COMPOSE_KNOWLEDGE_ANSWER":
            llm_ans = await llm.generate(KNOWLEDGE_SYSTEM, [history + _knowledge_brief(p)])
            text = (_strip_cjk(llm_ans).strip() if llm_ans else None) or _knowledge_template(p)
        else:
            text = await llm.generate(SMALL_TALK_SYSTEM, [history + p["text"]]) or _small_talk_template()
        if "tham khảo" not in text[-300:] and msg.msg_type in ("COMPOSE_ASSESSMENT", "COMPOSE_KNOWLEDGE_ANSWER"):
            text = f"{text}\n\n{DISCLAIMER}"
        if msg.msg_type == "COMPOSE_ASSESSMENT" and p.get("questions"):
            # Câu hỏi làm rõ được gắn nguyên văn (không qua LLM viết lại) để chắc chắn người dùng nhìn thấy
            text = f"{text}\n\n---\n**Để kết quả chính xác hơn, mình cần hỏi thêm:**\n{_questions_block(p['questions'])}"
        return self.reply(msg, "RESPONSE_TEXT", {"text": _plain_math(text),
                                                 "generated_by": "llm" if llm.enabled else "template"})


def _strip_cjk(text: str) -> str:
    """Loại bỏ ký tự chữ Hán (CJK) đôi khi rò rỉ từ bộ từ vựng song ngữ của Qwen nhỏ."""
    return re.sub(r"[\u4e00-\u9fff\u3400-\u4dbf]+", "", text)


# ------------------------------------------------------------------ dữ liệu cho LLM
_MATH = {r"\geq": "≥", r"\ge": "≥", r"\leq": "≤", r"\le": "≤", r"\times": "×", r"\%": "%", r"\approx": "≈",
         r"\pm": "±", r"\to": "→", r"\rightarrow": "→"}


def _plain_math(text: str) -> str:
    """LLM đôi khi viết ký hiệu LaTeX ($\\ge$ 126) mà khung chat không hiển thị: đổi về ký tự Unicode."""
    def repl(m: "re.Match") -> str:
        inner = m.group(1)
        if "\\" not in inner:  # không phải lệnh LaTeX (ví dụ "$5 và $10") → giữ nguyên
            return m.group(0)
        for k in sorted(_MATH, key=len, reverse=True):
            inner = inner.replace(k, _MATH[k])
        return inner.strip() if "\\" not in inner else m.group(0)
    return re.sub(r"\$([^$\n]{1,40})\$", repl, text)


def _history_block(history) -> str:
    return f"Lịch sử hội thoại gần đây:\n{format_history(history)}\n\n" if history else ""


def _contextual_brief(p: Dict[str, Any]) -> str:
    last = p.get("last_assessment") or {}
    brief = {
        "cau_hoi": p["question"],
        "ho_so_da_ghi_nhan": p["profile_summary"],
        "chi_so_bat_buoc_con_thieu": [FIELD_SPECS[f].label_vi for f in p.get("missing_required", [])],
        "chi_so_nen_bo_sung": [FIELD_SPECS[f].label_vi for f in p.get("missing_recommended", [])],
        "ket_qua_hoi_chan_gan_nhat": {k: last.get(k) for k in ("risk_label", "positive_evidence_plus",
                                                               "negative_contradictions_minus", "safety_guardrails",
                                                               "treatment", "citations") if last.get(k)} or None,
    }
    return "Dữ liệu (JSON):\n" + json.dumps(brief, ensure_ascii=False, indent=1)


def _assessment_brief(p: Dict[str, Any]) -> str:
    c, k, t, pred = p["consensus"], p["knowledge"], p["treatment"], p["prediction"]["risk_model"]
    rec = k["graph_rag"]["recommendations"]
    brief = {
        "ket_luan": c["risk_label"],
        "diem_nguy_co_hoi_dong": c["consensus_risk_score"],
        "xac_suat_du_doan_theo_yeu_to_nguy_co": pred["risk_probability"],
        "xac_suat_tung_mo_hinh": {k2: v["risk_probability"] for k2, v in pred.get("models", {}).items()},
        "gap_bao_nhieu_lan_trung_binh": pred["relative_risk"],
        "xet_nghiem": [{k2: f[k2] for k2 in ("test", "value", "unit", "status_vi", "threshold")}
                       for f in k["glycemia"]["findings"]],
        "can_xet_nghiem_lai_xac_nhan": c["confirmation_required"],
        "bang_chung_cong": c["positive_evidence_plus"],
        "diem_luu_y_tru": c["negative_contradictions_minus"],
        "canh_bao_an_toan": c["safety_guardrails"],
        "chuoi_suy_luan_do_thi": k["graph_rag"].get("paths", [])[:12],
        "khuyen_cao_tam_soat_tu_do_thi": rec.get("screenings", []),
        "can_thiep_tu_do_thi": rec.get("interventions", []),
        "nhom_thuoc_tu_do_thi": rec.get("drug_classes", []),
        "bien_chung_can_luu_y": rec.get("complications", []),
        "chong_chi_dinh_tu_do_thi": rec.get("contraindications", []),
        "goi_y_can_thiep_RL": {
            "hanh_dong": t["recommended_action"],
            "chi_tiet": t["intervention_details"],
            "thuoc_tham_chieu_tai_vn": t.get("reference_drugs_vn", []),
            "ke_hoach_theo_doi": t.get("monitoring_plan", ""),
            "ghi_chu": t["notes"],
        },
        "citations": [f"{x['title']} — {x['section']}" for x in k["citations"]],
        "doan_tai_lieu": [f"[{x['title']} — {x['heading']}] {x['content'][:700]}" for x in k["passages"][:4]],
        "missing_recommended": [FIELD_SPECS[f].label_vi for f in p.get("missing_recommended", [])],
        "du_lieu_moi_cap_nhat": p.get("updated_fields", []),
    }
    return "Kết quả hội đồng (JSON):\n" + json.dumps(brief, ensure_ascii=False, indent=1)


def _knowledge_brief(p: Dict[str, Any]) -> str:
    passages = "\n\n".join(f"[Nguồn: {x['title']} — {x['heading']}]\n{x['content']}" for x in p["passages"])
    graph_paths = (p.get("graph_rag") or {}).get("paths") or []
    graph_block = ("\n\nQuan hệ từ Đồ thị Tri thức (GraphRAG):\n- " + "\n- ".join(graph_paths[:12])) if graph_paths else ""
    return f"Câu hỏi: {p['question']}\n\nTài liệu:\n{passages}{graph_block}"


# Lời khuyên ngắn theo từng thói quen ăn uống (ADA 2024 Section 5, khuyến cáo dinh dưỡng của Bộ Y tế)
DIET_ADVICE_VEG = "Ăn rau xanh trong mọi bữa (khoảng nửa đĩa ăn), chất xơ giúp đường huyết sau ăn tăng chậm hơn."
DIET_ADVICE_FRUIT = "Ăn 1–2 phần trái cây nguyên quả mỗi ngày (ổi, táo, cam, bưởi), không thay bằng nước ép đóng chai."
SLEEP_ADVICE = "Ngủ đều 7–8 giờ mỗi đêm, đi ngủ và thức dậy cùng giờ; thiếu ngủ lâu ngày làm cơ thể xử lý đường kém đi."


# ------------------------------------------------------------------ mẫu câu dự phòng
def _whole_sentences(text: str) -> str:
    """Bỏ câu bị cắt dở khi LLM chạm giới hạn token."""
    text = text.strip()
    end = max(text.rfind(c) for c in ".!?")
    return text[:end + 1] if end > 0 else text


LEVEL_WORD = {"low": "thấp", "moderate": "trung bình", "high": "cao", "very_high": "rất cao"}
GLYCEMIC_PLAIN = {"normal": "trong mức bình thường", "prediabetes_range": "cao hơn bình thường (mức tiền tiểu đường)",
                  "diabetes_range": "đạt mức bác sĩ dùng để chẩn đoán tiểu đường"}
WARNING_SIGNS = ("Đi khám ngay nếu bạn thấy: khát và đi tiểu nhiều bất thường, sụt cân không rõ lý do, mờ mắt, "
                 "mệt lả, vết thương lâu lành, hoặc tê bì chân tay.")


def _factor_plain(f: Dict[str, Any]) -> str:
    """Một yếu tố làm TĂNG nguy cơ → câu dễ hiểu kèm lý do, cho người không có chuyên môn."""
    v, name = f.get("value"), f["feature"]
    plain = {
        "bmi": lambda: f"**Cân nặng hơi cao so với chiều cao** (BMI {v:.1f}; người châu Á từ 23 trở lên là thừa cân) — "
                       "mỡ thừa, nhất là mỡ bụng, khiến cơ thể khó đưa đường từ máu vào tế bào.",
        "age_years": lambda: f"**Tuổi {v:.0f}** — từ khoảng 35–40 tuổi, cơ bắp giảm dần và cơ thể xử lý đường kém đi.",
        "sex": lambda: "**Nam giới** — nam thường tích mỡ ở bụng nhiều hơn nữ nên nguy cơ cao hơn một chút.",
        "waist_cm": lambda: f"**Vòng eo {v:.0f} cm** (nam từ 90 cm, nữ từ 80 cm là béo bụng) — mỡ bụng là loại mỡ ảnh hưởng mạnh nhất tới đường huyết.",
        "high_bp": lambda: "**Tăng huyết áp** — huyết áp cao và tiểu đường thường đi cùng nhau vì chung nguyên nhân (cân nặng, ít vận động).",
        "systolic_bp": lambda: f"**Huyết áp hơi cao** (số trên {v:.0f} mmHg) — thường đi kèm rối loạn đường huyết.",
        "high_chol": lambda: "**Mỡ máu cao** — dấu hiệu cơ thể đang rối loạn chuyển hóa, hay đi cùng tiểu đường.",
        "family_history": lambda: "**Có bố mẹ hoặc anh chị em ruột bị tiểu đường** — gen di truyền làm bạn dễ mắc hơn người khác 2–3 lần.",
        "smoker": lambda: "**Hút thuốc** — khói thuốc làm cơ thể dùng đường kém hơn và hại mạch máu.",
        "phys_activity": lambda: "**Ít vận động** — cơ bắp khi vận động sẽ \"đốt\" đường trong máu; ít vận động thì đường dễ tồn đọng.",
        "heart_disease": lambda: "**Có bệnh tim mạch** — tim mạch và tiểu đường có chung nhiều yếu tố nguy cơ.",
        "stroke": lambda: "**Từng bị đột quỵ** — cho thấy mạch máu đã bị ảnh hưởng, cần kiểm soát đường huyết chặt hơn.",
        "sleep_hours": lambda: f"**Ngủ {v:g} giờ mỗi đêm** (nên 7–8 giờ) — ngủ quá ít hoặc quá nhiều làm rối loạn hormone điều hòa đường và cảm giác đói.",
    }.get(name)
    return plain() if plain else f["feature_vi"]


# Nút yếu tố nguy cơ trên đồ thị tri thức → (đặc trưng mô hình tương ứng, câu dễ hiểu)
ADA_FACTORS = {
    "Thừa cân/béo phì": ("bmi", "**Thừa cân** (người châu Á có BMI từ 23 trở lên) — mỡ thừa khiến cơ thể khó đưa đường từ máu vào tế bào."),
    "Béo bụng": ("waist_cm", "**Béo bụng** (vòng eo nam từ 90 cm, nữ từ 80 cm) — mỡ bụng là loại mỡ ảnh hưởng mạnh nhất tới đường huyết."),
    "Tăng huyết áp": ("high_bp", "**Tăng huyết áp** — thường đi cùng tiểu đường vì chung nguyên nhân (cân nặng, ít vận động)."),
    "Rối loạn lipid máu": ("high_chol", "**Mỡ máu cao** — dấu hiệu rối loạn chuyển hóa, hay đi cùng tiểu đường."),
    "Tiền sử gia đình ĐTĐ": ("family_history", "**Có người thân ruột thịt bị tiểu đường** — gen di truyền làm bạn dễ mắc hơn 2–3 lần."),
    "Ít vận động": ("phys_activity", "**Ít vận động** — cơ bắp khi vận động sẽ \"đốt\" đường trong máu; ít vận động thì đường dễ tồn đọng."),
    "Hút thuốc lá": ("smoker", "**Hút thuốc** — khói thuốc làm cơ thể dùng đường kém hơn và hại mạch máu."),
}

GLYCEMIC_SCALE = {
    "Glucose huyết tương lúc đói": "Mức tham khảo: dưới 5,6 mmol/L là bình thường, 5,6–6,9 là tiền tiểu đường, từ 7,0 là mức tiểu đường.",
    "HbA1c": "HbA1c cho biết đường huyết trung bình 2–3 tháng qua: dưới 5,7% là bình thường, 5,7–6,4% là tiền tiểu đường, từ 6,5% là mức tiểu đường.",
}


def _assessment_sections(p: Dict[str, Any]):
    """Câu trả lời cho người dùng phổ thông: (tỷ lệ + vì sao, lời khuyên). Chi tiết mô hình/đồ thị nằm ở mục thu gọn."""
    c, k, t = p["consensus"], p["knowledge"], p["treatment"]
    ml = p["prediction"]["risk_model"]
    level, status = c["risk_level"], c["glycemic_status"]
    lines: List[str] = [
        f"### 🩺 Khả năng mắc tiểu đường của bạn: khoảng {c['consensus_risk_score'] * 100:.0f}% — mức {LEVEL_WORD[level]}",
        f"> ℹ️ {c.get('relative_note') or 'Kết quả chỉ mang tính tương đối, không thay thế chẩn đoán của bác sĩ.'}",
    ]
    lines += [f"> ⚠️ **{a['message']}**" for a in c["safety_guardrails"]]

    reasons: List[str] = []
    for f in k["glycemia"]["findings"]:
        value = f"{f['value']:g} {f['unit']}"
        if f["unit"] == "mg/dL" and "lucose" in f["test"]:
            value = f"{f['value'] / 18.016:.1f} mmol/L ({f['value']:g} mg/dL)"
        scale = GLYCEMIC_SCALE.get(f["test"], "")
        reasons.append(f"{f['test']} của bạn là **{value}**, {GLYCEMIC_PLAIN.get(f['status'], f['status_vi'].lower())}. {scale}".strip())
    if status == "unknown":
        reasons.append("Bạn **chưa có kết quả xét nghiệm đường huyết**, nên con số trên chỉ được ước tính từ tuổi, "
                       "cân nặng, tiền sử và thói quen sống.")
    if c["confirmation_required"]:
        reasons.append("Một lần xét nghiệm chưa đủ để kết luận chắc chắn — cần đo lại lần nữa.")
    up = [f for f in ml["top_risk_factors"] if f["direction"] == "RISK_UP"][:5]
    reasons += [_factor_plain(f) for f in up]
    # Yếu tố nguy cơ theo ADA mà đồ thị tri thức nhận ra nhưng mô hình không xếp vào top (ví dụ BMI, vòng eo)
    shown = {f["feature"] for f in up}
    entities = set(k["graph_rag"].get("start_entities") or [])
    reasons += [text for ent, (feat, text) in ADA_FACTORS.items() if ent in entities and feat not in shown]
    risks = c.get("diet_risks") or []
    LIFE_WHY = {"Ít ăn rau xanh": "**Ít ăn rau xanh** — thiếu chất xơ làm đường vào máu nhanh sau mỗi bữa ăn.",
                "Ít ăn trái cây": "**Ít ăn trái cây** — thiếu chất xơ và vitamin giúp kiểm soát đường huyết."}
    reasons += [LIFE_WHY.get(x, f"**{x}** — ngủ không đủ hoặc quá nhiều làm cơ thể xử lý đường kém đi.") for x in risks
                if not x.startswith("Ngủ") or not any(f["feature"] == "sleep_hours" for f in ml["top_risk_factors"])]
    lines += ["", "### 💡 Vì sao có kết quả này?"] + [f"- {x}" for x in reasons]

    advice: List[str] = ["### ✅ Bạn nên làm gì?"]
    step = {
        "diabetes_range": (["Đi khám **bác sĩ Nội tiết** trong 1–2 tuần và **làm lại xét nghiệm** để xác nhận: đường huyết lúc đói và HbA1c.",
                            "Nhịn ăn ít nhất 8 giờ trước khi lấy máu (chỉ uống nước lọc); mang theo kết quả xét nghiệm cũ.",
                            "Nên làm thêm mỡ máu, chức năng thận và nước tiểu để bác sĩ đánh giá toàn diện."]
                           if c["confirmation_required"] else
                           ["Đi khám **bác sĩ Nội tiết** sớm để được tư vấn điều trị và đặt mục tiêu đường huyết.",
                            "Nên kiểm tra thêm mỡ máu, chức năng thận, khám mắt và bàn chân để phát hiện sớm biến chứng."]),
        "prediabetes_range": ["Đây là giai đoạn **còn đảo ngược được**: thay đổi lối sống ngay từ bây giờ.",
                              "Kiểm tra lại đường huyết lúc đói hoặc HbA1c sau 6–12 tháng để xem tiến triển."],
        "unknown": ["**Đi xét nghiệm đường huyết lúc đói hoặc HbA1c** — có ở hầu hết bệnh viện, phòng khám, chi phí thấp.",
                    "Nhịn ăn ít nhất 8 giờ trước khi lấy máu (chỉ uống nước lọc), nên đi vào buổi sáng.",
                    "Có kết quả, bạn nhập vào đây để hội đồng đánh giá chính xác hơn."],
        "normal": ["Đường huyết hiện ổn: duy trì lối sống lành mạnh và kiểm tra lại mỗi 1–3 năm (sớm hơn nếu có triệu chứng)."],
    }.get(status, [])
    if step:
        advice += ["**1. Bước tiếp theo**"] + [f"- {x}" for x in step]
    advice += ["", "**2. Thay đổi lối sống** (quan trọng nhất, nên làm ngay)"]
    if entities & {"Thừa cân/béo phì", "Béo bụng"} or any(f["feature"] in ("bmi", "waist_cm") for f in up):
        advice.append("- **Cân nặng:** giảm 5–7% cân nặng trong 6 tháng (ví dụ 70 kg → giảm 4–5 kg, khoảng 0,5 kg mỗi tuần). "
                      "Giảm chừng đó đã đủ để nguy cơ giảm rõ rệt.")
    advice += [
        "- **Ăn uống:**",
        "  - Chia đĩa ăn: một nửa rau, một phần tư thịt cá/trứng/đậu, một phần tư cơm hoặc tinh bột.",
        "  - Ăn rau trước, cơm sau; đổi một phần cơm trắng sang gạo lứt, khoai, yến mạch.",
        "  - Uống nước lọc, trà không đường thay cho nước ngọt, trà sữa, nước ép đóng chai.",
        "  - Hạn chế bánh kẹo, chè, đồ chiên rán; ưu tiên luộc, hấp, nướng.",
        "- **Vận động:**",
        "  - Ít nhất 150 phút mỗi tuần, ví dụ đi bộ nhanh 30 phút × 5 ngày (nhanh đến mức hơi khó nói chuyện).",
        "  - Thêm 2 buổi tập sức mạnh mỗi tuần (squat, chống đẩy, dây kháng lực) để tăng cơ.",
        "  - Đi bộ 10–15 phút sau bữa ăn giúp đường huyết sau ăn không tăng vọt.",
    ]
    extra = [DIET_ADVICE_VEG] if "Ít ăn rau xanh" in risks else []
    extra += [DIET_ADVICE_FRUIT] if "Ít ăn trái cây" in risks else []
    extra += [SLEEP_ADVICE] if any(x.startswith("Ngủ") for x in risks) else []
    if "Hút thuốc lá" in entities or any(f["feature"] == "smoker" for f in up):
        extra.append("Bỏ thuốc lá: hút thuốc làm tăng cả nguy cơ tiểu đường lẫn biến chứng tim mạch.")
    if extra:
        advice += ["- **Riêng với thói quen của bạn:**"] + [f"  - {x}" for x in extra]
    advice += ["", "**3. Hướng điều trị & thuốc**",
               f"- Hướng gợi ý: **{re.sub(r'\s*\(.*?\)\s*$', '', t['recommended_action'])}**."]
    if t.get("reference_drugs_vn"):
        advice.append("- Thuốc bác sĩ có thể cân nhắc (**chỉ dùng khi được bác sĩ khám và kê đơn**, không tự mua, không tự ngừng):")
        advice += [f"  - {d}" for d in t["reference_drugs_vn"]]
        advice.append("- Khi đi khám, hãy nói với bác sĩ các thuốc bạn đang dùng và bệnh thận, gan (nếu có) để chọn thuốc an toàn.")
    else:
        advice.append("- Hiện **chưa cần dùng thuốc**. Nghiên cứu lớn cho thấy thay đổi lối sống đúng cách giảm khoảng một nửa "
                      "nguy cơ tiến triển thành tiểu đường — hiệu quả hơn cả dùng thuốc ở giai đoạn này.")
    if t.get("monitoring_plan"):
        advice.append(f"- **Theo dõi:** {t['monitoring_plan']}")
    advice.append("- Tự theo dõi tại nhà: cân nặng và vòng eo mỗi tuần, ghi lại để mang theo khi đi khám.")
    advice += ["", f"🚑 {WARNING_SIGNS}"]
    return "\n".join(lines), "\n".join(advice)


def _follow_up_template(p: Dict[str, Any]) -> str:
    lines = []
    if p.get("understood"):
        lines += ["Mình đã ghi nhận:"] + [f"- {u}" for u in p["understood"]] + [""]
    for d in p.get("documents", []):
        if d.get("error"):
            lines.append(f"⚠️ Không đọc được **{d['filename']}**: {d['error']}")
        elif not d.get("is_medical_document", True):
            lines.append(f"⚠️ **{d['filename']}** có vẻ không phải tài liệu y tế.")
    if lines and lines[-1] != "":
        lines.append("")
    questions = [FIELD_SPECS[f].question_vi for f in p["missing_required"] if FIELD_SPECS[f].question_vi]
    questions += [q["question"] for q in p.get("questions", [])]
    lines.append("Để hội đồng các tác tử đánh giá nguy cơ đái tháo đường, mình cần thêm:")
    lines += [f"{i}. {q}" for i, q in enumerate(questions, 1)]
    lines += ["", "Bạn có thể trả lời bằng tin nhắn hoặc tải lên phiếu xét nghiệm / bệnh án (PDF, ảnh, DOCX)."]
    return "\n".join(lines)


def _clarification_template(p: Dict[str, Any]) -> str:
    lines = []
    if p.get("understood"):
        lines += ["Mình đã ghi nhận:"] + [f"- {u}" for u in p["understood"]] + [""]
    lines += ["Trước khi đánh giá, mình cần hỏi thêm vì câu trả lời sẽ thay đổi cách hiểu kết quả:",
              _questions_block(p["questions"])]
    return "\n".join(lines)


def _questions_block(questions: List[Dict[str, Any]]) -> str:
    return "\n".join(f"{i}. {q['question']}  \n   _Lý do: {q['reason']}_" for i, q in enumerate(questions, 1))


def _contextual_template(p: Dict[str, Any]) -> str:
    import re
    question = p["question"].lower()
    asked = [label for pattern, label in FIELD_KEYWORDS if re.search(pattern, question)]
    lines: List[str] = []
    if asked:
        for label in asked:
            found = [s for s in p["profile_summary"] if label.lower() in s.lower()]
            lines += found or [f"- {label}: bạn **chưa cung cấp** trong cuộc trò chuyện này."]
        lines = [l if l.startswith("-") else f"- {l}" for l in lines]
        return "Theo những gì bạn đã cung cấp:\n" + "\n".join(lines)
    last = p.get("last_assessment")
    lines = ["**Hồ sơ đã ghi nhận:**"] + ([f"- {s}" for s in p["profile_summary"]] or ["- Chưa có dữ liệu."])
    if last:
        lines += ["", f"**Kết quả lần đánh giá gần nhất:** {last['risk_label']}"]
        lines += [f"- {x}" for x in last.get("positive_evidence_plus", []) + last.get("negative_contradictions_minus", [])]
    return "\n".join(lines)


def _knowledge_template(p: Dict[str, Any]) -> str:
    if not p["passages"]:
        return "Cơ sở tri thức hiện chưa có thông tin cho câu hỏi này. Bạn nên hỏi trực tiếp bác sĩ."
    lines = [f"### Giải đáp Chuyên sâu từ Cơ sở Tri thức Y khoa & Đồ thị Tri thức (GraphRAG)", ""]
    for i, x in enumerate(p["passages"][:4], 1):
        lines += [f"#### {i}. {x['heading']} _(Nguồn: {x['title']})_", x["content"], ""]
    graph = p.get("graph_rag") or {}
    paths = graph.get("paths") or []
    if paths:
        lines += ["#### Chuỗi suy luận liên kết trên Đồ thị Tri thức (Neo4j GraphRAG):"]
        for path_str in paths[:8]:
            lines.append(f"- `{path_str}`")
        lines.append("")
    citations = [f"[{x['title']} — {x['heading']}]" + (f"({x['url']})" if x.get("url") else "") for x in p["passages"][:4]]
    if citations:
        lines.append("**Nguồn trích dẫn:** " + "; ".join(citations))
    return "\n".join(lines).strip()


def _small_talk_template() -> str:
    return ("Xin chào, mình là **Diabetes Prediction** — trợ lý sàng lọc nguy cơ đái tháo đường.\n\n"
            "Bạn có thể:\n- Nhập tuổi, chiều cao, cân nặng, đường huyết lúc đói hoặc HbA1c để mình đánh giá nguy cơ\n"
            "- Tải lên phiếu xét nghiệm / bệnh án (PDF, ảnh, DOCX, CSV)\n- Hỏi kiến thức về đái tháo đường")
