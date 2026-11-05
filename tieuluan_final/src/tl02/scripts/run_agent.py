"""Đánh giá tác tử ReAct trên 16 kịch bản (8 y tế, 8 tài chính) và so với LLM trả lời một mình, không công cụ.

Chỉ số: độ chính xác chọn công cụ (precision/recall so với tập công cụ kỳ vọng), độ phủ dữ kiện bắt buộc trong
câu trả lời, độ trung thực số học (con số trong câu trả lời có trong bằng chứng), tỷ lệ qua kiểm chứng, độ trễ.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_graphrag import coverage  # noqa: E402
from tl2.agent import ReActAgent, verify  # noqa: E402
from tl2.common import save_json, seed_all  # noqa: E402

SC = [
    ("health", "Tôi bị tiểu đường típ 2 đang uống Metformin 500mg, tối nay đi tiệc có uống bia được không?",
     {"kg_y_te", "truy_xuat_rag"}, [["toan lactic"], ["tranh", "khong nen", "khong uong"]]),
    ("health", "Bị tiểu đường có ăn thịt gà được không?", {"kg_y_te", "truy_xuat_rag"},
     [["uc ga"], ["ga ran", "chien"]]),
    ("health", "Tôi nam 52 tuổi, HbA1c 7.3, đường huyết lúc đói 138, BMI 26.5. Tôi có bị tiểu đường không?",
     {"chan_doan_nguy_co", "kg_y_te", "truy_xuat_rag"}, [["126"], ["6,5", "6.5"]]),
    ("health", "Tôi nữ 45 tuổi, HbA1c 5.9, đường huyết 110, BMI 24. Tôi có nguy cơ gì và nên làm gì?",
     {"chan_doan_nguy_co", "kg_y_te", "truy_xuat_rag"}, [["tien dai thao duong"], ["van dong", "150"]]),
    ("health", "Người tiểu đường ăn cơm trắng thế nào cho đúng?", {"kg_y_te", "truy_xuat_rag"},
     [["73"], ["bat con", "dinh luong"]]),
    ("health", "Bệnh nhân suy thận nặng có dùng metformin được không?", {"kg_y_te", "truy_xuat_rag"},
     [["30"], ["chong chi dinh"]]),
    ("health", "Đang tiêm insulin mà bị hạ đường huyết thì xử trí thế nào?", {"kg_y_te", "truy_xuat_rag"},
     [["15"], ["70"]]),
    ("health", "Tiểu đường kèm suy tim nên ưu tiên thuốc gì?", {"kg_y_te", "truy_xuat_rag"},
     [["sglt2", "empagliflozin"]]),
    ("finance", "Tỷ giá USD/VND tăng mạnh thì cổ phiếu FPT và giá vàng SJC ảnh hưởng thế nào?",
     {"kg_tai_chinh", "du_lieu_thi_truong", "truy_xuat_rag"}, [["xuat khau", "ngoai te"], ["quy doi", "1,2057"]]),
    ("finance", "Giá vàng thế giới quy đổi hôm nay khoảng bao nhiêu triệu đồng một lượng?",
     {"quy_doi_vang", "truy_xuat_rag"}, [["trieu dong"], ["1,2057"]]),
    ("finance", "Dự báo giá cổ phiếu FPT phiên tới?", {"du_lieu_thi_truong", "du_bao_bilstm", "truy_xuat_rag"},
     [["bilstm"], ["dung chieu", "ngay tho", "tham khao"]]),
    ("finance", "Tỷ giá tăng thì Hòa Phát bị ảnh hưởng ra sao?", {"kg_tai_chinh", "du_lieu_thi_truong", "truy_xuat_rag"},
     [["quang sat", "than"], ["ty gia"]]),
    ("finance", "Lãi suất điều hành NHNN giảm thì VCB và VIC hưởng lợi thế nào?",
     {"kg_tai_chinh", "du_lieu_thi_truong", "truy_xuat_rag"}, [["casa", "nim", "tin dung"], ["bat dong san", "vay"]]),
    ("finance", "Nghị định 232/2025 thay đổi gì với vàng miếng SJC?", {"kg_tai_chinh", "truy_xuat_rag"},
     [["doc quyen"], ["chenh lech"]]),
    ("finance", "Fed tăng lãi suất thì giá vàng thế giới và tỷ giá biến động ra sao?", {"kg_tai_chinh", "truy_xuat_rag"},
     [["chi phi co hoi", "dxy"], ["ty gia"]]),
    ("finance", "Xu hướng giá vàng phiên tới thế nào, có nên mua vàng không?",
     {"quy_doi_vang", "du_bao_bilstm", "truy_xuat_rag"}, [["bilstm"], ["tham khao", "khong phai khuyen nghi"]]),
]
SHOWCASE = (0, 8)


def main():
    seed_all()
    agent = ReActAgent(use_llm=True)
    rows, traces = [], {}
    for i, (dom, q, tools, facts) in enumerate(SC):
        r = agent.run(q)
        used = {s["tool"] for s in r["trace"] if s["type"] == "ACTION"}
        tp = len(used & tools)
        cov, hits = coverage(r["answer"], facts)
        only, _, only_s = agent.llm.chat(q, system="Bạn là trợ lý tiếng Việt. Trả lời ngắn gọn, chính xác.")
        cov0, _ = coverage(only, facts)
        ver0 = verify(only, r["evidence"], q)
        tool_ms = {s["tool"]: s["ms"] for s in r["trace"] if s["type"] == "ACTION"}
        rows.append({"id": i, "domain": dom, "q": q, "domain_pred": r["domain"], "tools": sorted(used),
                     "expected": sorted(tools), "tool_precision": tp / max(1, len(used)),
                     "tool_recall": tp / len(tools), "coverage": cov, "hits": hits, "coverage_llm_only": cov0,
                     "faith": r["verification"].get("numeric_faithfulness"), "faith_llm_only": ver0["numeric_faithfulness"],
                     "used": r["verification"]["used"], "n_evidence": len(r["evidence"]),
                     "n_citations": r["verification"].get("n_citations"), "total_ms": r["total_ms"],
                     "llm_s": r["verification"].get("llm_s"), "tool_ms": tool_ms, "answer": r["answer"],
                     "llm_only": only, "llm_only_s": only_s, "entities": [e for e, _ in r["entities"]]})
        if i in SHOWCASE:
            traces[dom] = {k: r[k] for k in ("question", "domain", "entities", "profile", "trace", "evidence", "answer",
                                             "verification", "total_ms")}
        print(i, dom, r["domain"], sorted(used), round(cov, 2), round(cov0, 2), r["verification"]["used"], flush=True)
    agg = {}
    for dom in ("health", "finance", "all"):
        rs = [r for r in rows if dom == "all" or r["domain"] == dom]
        agg[dom] = {k: float(np.mean([r[k] for r in rs])) for k in
                    ("tool_precision", "tool_recall", "coverage", "coverage_llm_only", "faith", "faith_llm_only",
                     "total_ms", "llm_s", "llm_only_s")}
        agg[dom]["domain_acc"] = float(np.mean([r["domain"] == r["domain_pred"] for r in rs]))
        agg[dom]["llm_pass"] = float(np.mean([r["used"] == "llm" for r in rs]))
    tools = sorted({t for r in rows for t in r["tool_ms"]})
    agg["tool_ms"] = {t: float(np.median([r["tool_ms"][t] for r in rows if t in r["tool_ms"]])) for t in tools}
    agg["tool_calls"] = {t: int(sum(t in r["tool_ms"] for r in rows)) for t in tools}
    save_json("agent", {"rows": rows, "agg": agg, "traces": traces})
    print(agg)


if __name__ == "__main__":
    main()
