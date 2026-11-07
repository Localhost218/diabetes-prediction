"""
Tác tử Phản hồi (Response Agent)
Viết câu trả lời tiếng Việt từ kết quả của các tác tử. Với LLM: chỉ được dùng dữ liệu trong JSON đầu vào
(grounded generation). Không có LLM: dùng mẫu câu có cấu trúc.
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
        p = msg.payload
        history = _history_block(p.get("history"))
        if msg.msg_type == "COMPOSE_ASSESSMENT":
            base_report = _assessment_template(p)
            reasoning = await llm.generate(RESPONSE_SYSTEM, [history + base_report])
            if reasoning:
                clean_reasoning = _strip_cjk(reasoning).strip()
                text = f"{base_report}\n\n### 6. Nhận định Tổng hợp từ Mô hình Ngôn ngữ (LLM Synthesis)\n{clean_reasoning}"
            else:
                text = base_report
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


# ------------------------------------------------------------------ mẫu câu dự phòng
def _assessment_template(p: Dict[str, Any]) -> str:
    c, k, t = p["consensus"], p["knowledge"], p["treatment"]
    ml = p["prediction"]["risk_model"]
    lines: List[str] = [
        f"### 1. Tổng quan & Phân tầng nguy cơ: {c['risk_label']} (Điểm hội đồng: {c['consensus_risk_score'] * 100:.1f}%)",
        "",
    ]
    for a in c["safety_guardrails"]:
        lines.append(f"> ⚠️ **{a['message']}**")
    if c["safety_guardrails"]:
        lines.append("")
    if k["glycemia"]["findings"]:
        lines.append("**Đối chiếu tiêu chuẩn lâm sàng (ADA 2024 & QĐ 5481/QĐ-BYT):**")
        lines += [f"- **{f['test']}**: {f['value']:g} {f['unit']} → **{f['status_vi']}** (Ngưỡng chuẩn: {f['threshold']})"
                  for f in k["glycemia"]["findings"]]
        lines.append("")
    models = ml.get("models", {})
    detail = (f" (GBDT: {models['ml']['risk_probability'] * 100:.1f}%, Deep MLP: {models['dl']['risk_probability'] * 100:.1f}%)"
              if models else "")
    lines += [
        "### 2. Suy luận từ Mô hình Học máy / Học sâu & Hội đồng Phản biện",
        f"- **Xác suất nguy cơ dự báo (NHANES + BRFSS):** **{ml['risk_probability'] * 100:.1f}%**{detail} — cao gấp **{ml['relative_risk']} lần** mức nền của quần thể.",
    ]
    for x in c["positive_evidence_plus"]:
        lines.append(f"- **[+] Bằng chứng tăng nguy cơ:** {x}")
    for x in c["negative_contradictions_minus"]:
        lines.append(f"- **[-] Điểm cần lưu ý / đối chứng:** {x}")

    rec = k["graph_rag"]["recommendations"]
    paths = k["graph_rag"].get("paths", [])
    lines += ["", "### 3. Suy luận Đồ thị Tri thức (Neo4j GraphRAG) & Tầm soát biến chứng"]
    if paths:
        lines.append("- **Chuỗi liên kết bệnh sinh – điều trị trên Đồ thị Tri thức:**")
        for path_str in paths[:6]:
            lines.append(f"  - `{path_str}`")
    if rec.get("complications"):
        lines.append(f"- **Các biến chứng đích cần cảnh giác:** {', '.join(rec['complications'])}.")
    if rec.get("screenings"):
        lines.append(f"- **Xét nghiệm tầm soát khuyến cáo:** {', '.join(rec['screenings'])}.")

    lines += [
        "",
        "### 4. Định hướng Can thiệp Lối sống & Phác đồ Tham chiếu (RL + ADA / Vinmec / Long Châu)",
        f"- **Chiến lược ưu tiên (Q-Learning RL):** **{t['recommended_action']}** — {t['intervention_details']}",
    ]
    if t.get("reference_drugs_vn"):
        lines.append("- **Nhóm thuốc & biệt dược tham chiếu để trao đổi với bác sĩ:**")
        for drug_info in t["reference_drugs_vn"]:
            lines.append(f"  - {drug_info}")
    if rec.get("contraindications"):
        lines.append(f"- **Lưu ý chống chỉ định trên đồ thị:** {'; '.join(rec['contraindications'])}.")
    if t.get("monitoring_plan"):
        lines.append(f"- **Kế hoạch theo dõi định kỳ:** {t['monitoring_plan']}")
    lines += [f"- {n}" for n in t["notes"][1:]]
    missing = [FIELD_SPECS[f].label_vi for f in p.get("missing_recommended", [])]
    if missing:
        lines += ["", f"**Gợi ý bổ sung thông tin:** Để tăng độ chính xác cho lần đánh giá sau, bạn nên cung cấp thêm: **{', '.join(missing)}**."]

    lines += ["", "### 5. Biện luận Cơ chế Bệnh sinh & Lời khuyên Lâm sàng Chuyên sâu", _pathophysiology_reasoning(p)]

    if k["citations"]:
        lines += ["", "**Nguồn tài liệu y khoa đối chiếu:** " + "; ".join(f"[{x['title']} — {x['section']}]" for x in k["citations"][:5])]
    return "\n".join(lines)


def _pathophysiology_reasoning(p: Dict[str, Any]) -> str:
    """Biện luận cơ chế bệnh sinh cá thể hóa dựa trên tổ hợp yếu tố nguy cơ, xét nghiệm và phác đồ của bệnh nhân."""
    c, k, t = p["consensus"], p["knowledge"], p["treatment"]
    ml = p["prediction"]["risk_model"]
    entities = set(k["graph_rag"].get("start_entities") or [])
    findings = k["glycemia"]["findings"]
    bullets: List[str] = []

    if findings:
        f_strs = [f"{f['test']} đạt **{f['value']:g} {f['unit']}** ({f['status_vi']})" for f in findings]
        if c["glycemic_status"] == "diabetes_range":
            bullets.append(
                f"- **Ý nghĩa sinh lý bệnh của xét nghiệm đường huyết ({', '.join(f_strs)}):** "
                "Mức đường huyết vượt ngưỡng chẩn đoán phản ánh sự suy giảm chức năng tiết insulin pha sớm của tế bào beta đảo tụy "
                "kết hợp với tình trạng đề kháng insulin tại gan (tăng tân tạo glucose ban đêm và sáng sớm) và tại mô cơ xương. "
                + ("Theo ADA 2024 và Bộ Y tế (QĐ 5481/QĐ-BYT), nếu chưa có triệu chứng tăng đường huyết kinh điển (uống nhiều, tiểu nhiều, sụt cân nhanh), "
                   "cần làm lại **xét nghiệm lần 2** (ưu tiên kết hợp cả FPG và HbA1c) để phân biệt tăng đường huyết mạn tính thực sự với tăng đường huyết nhất thời."
                   if c["confirmation_required"] else
                   "Khi đã có từ 2 chỉ số vượt ngưỡng trở lên, chẩn đoán đái tháo đường có độ đặc hiệu rất cao, cần khám chuyên khoa Nội tiết sớm để khởi trị.")
            )
        elif c["glycemic_status"] == "prediabetes":
            bullets.append(
                f"- **Giai đoạn cửa sổ vàng của Tiền đái tháo đường ({', '.join(f_strs)}):** "
                "Ở giai đoạn tiền đái tháo đường, tình trạng đề kháng insulin đã xuất hiện làm tăng gánh nặng tiết bù cho tế bào beta tụy, "
                "nhưng khoảng 50–70% chức năng đảo tụy vẫn có thể bảo tồn và phục hồi hoàn toàn nếu can thiệp giảm cân và tập luyện đúng phác đồ DPP (giảm tới 58% nguy cơ tiến triển thành ĐTĐ típ 2)."
            )

    mech_parts = []
    if "Thừa cân/béo phì" in entities or "Béo bụng" in entities:
        mech_parts.append(
            "**thừa cân/béo phì theo ngưỡng châu Á (BMI ≥ 23 kg/m²)** làm tăng mô mỡ nội tạng, giải phóng acid béo tự do (FFA) "
            "và các cytokine tiền viêm (TNF-α, IL-6) gây ức chế tín hiệu thụ thể insulin (IRS-1/PI3K) tại cơ và gan"
        )
    if "Tiền sử gia đình ĐTĐ" in entities:
        mech_parts.append(
            "**tiền sử gia đình thế hệ thứ nhất mắc đái tháo đường** mang yếu tố nhạy cảm di truyền làm giảm dự trữ tiết insulin của tế bào beta khi gặp gánh nặng chuyển hóa"
        )
    if "Tăng huyết áp" in entities or "Rối loạn lipid máu" in entities:
        mech_parts.append(
            "**tăng huyết áp / rối loạn lipid máu đồng mắc** tạo thành hội chứng chuyển hóa, thúc đẩy tổn thương nội mô mạch máu sớm"
        )
    if "Tuổi ≥ 35" in entities:
        mech_parts.append("**yếu tố tuổi tác (≥ 35 tuổi)** gắn liền với sự suy giảm tự nhiên khối lượng cơ xương và độ nhạy insulin")
    if mech_parts:
        bullets.append("- **Cơ chế tương tác cộng hưởng trên ca bệnh này:** Sự kết hợp giữa " + "; cùng với ".join(mech_parts) + f" giải thích vì sao mô hình học máy/học sâu đánh giá nguy cơ cao gấp **{ml['relative_risk']} lần** mức nền quần thể.")

    bullets.append(
        f"- **Cơ sở khoa học của phác đồ can thiệp ({t['recommended_action']}):** "
        "Vận động thể lực cường độ trung bình ≥ 150 phút/tuần kích hoạt trực tiếp kênh vận chuyển **GLUT4** đưa glucose vào tế bào cơ độc lập với insulin; "
        "đồng thời giảm 5–7% cân nặng giúp giảm mỡ gan và phục hồi độ nhạy insulin ngoại vi. "
        "Khi trao đổi với bác sĩ Nội tiết về các thuốc tham chiếu (như **Metformin / Glucophage** làm giảm sản xuất glucose tại gan, "
        "hoặc **ức chế SGLT2 — Forxiga / Jardiance** thải đường qua thận và bảo vệ tim – thận, hoặc **DPP-4i — Januvia / Trajenta** an toàn khi suy thận), "
        "người bệnh nên thông báo rõ chức năng thận (**eGFR, Creatinin**) và các thuốc đang dùng để bác sĩ cá thể hóa liều lượng an toàn nhất."
    )
    return "\n".join(bullets)


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
