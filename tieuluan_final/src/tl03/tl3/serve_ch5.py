"""Chương 5 — hàm suy luận cho Web App: chatbot hỏi đáp trên ba kho tri thức (tiểu đường, cổ phiếu Google, quỹ vàng).

predict(exp_id, {"question": str, "mode": "auto"|"graph"|"rag"}) -> {answer, sources, tools, trace, latency_ms, model}
Miền, chỉ mục và LLM được nạp lười ở lần hỏi đầu tiên rồi giữ trong bộ nhớ tiến trình.
"""
import time

SPECS = {
    "ch5_diabetes": {"title": "Chatbot tri thức bệnh tiểu đường (KG + RAG)", "kind": "chat",
                     "inputs": [{"name": "question", "label": "Câu hỏi", "type": "text",
                                 "default": "Tỷ lệ mắc tiểu đường của bệnh nhân nữ nhóm tuổi 60–79 có tăng huyết áp là bao nhiêu?"},
                                {"name": "mode", "label": "Chế độ", "type": "select", "options": ["auto", "graph", "rag"],
                                 "default": "auto"}],
                     "models": ["agent"],
                     "example": {"question": "Tỷ lệ mắc tiểu đường của bệnh nhân nữ nhóm tuổi 60–79 có tăng huyết áp "
                                             "là bao nhiêu?", "mode": "auto"}},
    "ch5_stock": {"title": "Chatbot tri thức cổ phiếu Google 2015–2024", "kind": "chat",
                  "inputs": [{"name": "question", "label": "Câu hỏi", "type": "text",
                              "default": "Giá đóng cửa của cổ phiếu Google ngày 16/03/2020 là bao nhiêu?"},
                             {"name": "mode", "label": "Chế độ", "type": "select", "options": ["auto", "graph", "rag"],
                              "default": "auto"}],
                  "models": ["agent"],
                  "example": {"question": "Giá đóng cửa của cổ phiếu Google ngày 16/03/2020 là bao nhiêu?", "mode": "auto"}},
    "ch5_gold": {"title": "Chatbot tri thức quỹ vàng GLD và thị trường 2011–2018", "kind": "chat",
                 "inputs": [{"name": "question", "label": "Câu hỏi", "type": "text",
                             "default": "Trong năm 2013, quỹ vàng GLD thay đổi bao nhiêu phần trăm?"},
                            {"name": "mode", "label": "Chế độ", "type": "select", "options": ["auto", "graph", "rag"],
                             "default": "auto"}],
                 "models": ["agent"],
                 "example": {"question": "Trong năm 2013, quỹ vàng GLD thay đổi bao nhiêu phần trăm?", "mode": "auto"}},
}
STEP = {"question": "QUESTION", "thought": "THOUGHT", "action": "ACTION", "obs": "OBSERVATION", "answer": "ANSWER"}
_AGENTS = {}


def _agent(domain):
    if domain not in _AGENTS:
        from .ch5_agent import Ch5Agent, Retrievers, load_domain
        D = load_domain(domain)
        _AGENTS[domain] = Ch5Agent(D, Retrievers(D))
    return _AGENTS[domain]


def predict(exp_id, payload, model_id=None):
    """Trả lời một câu hỏi; nguồn kèm điểm (rerank/RRF) và vết suy luận THOUGHT → ACTION → OBSERVATION → ANSWER."""
    if exp_id not in SPECS:
        raise KeyError(exp_id)
    t0 = time.perf_counter()
    agent = _agent(exp_id.split("_", 1)[1])
    q = (payload.get("question") or SPECS[exp_id]["example"]["question"]).strip()
    out = agent.run(q, mode=payload.get("mode", "auto"))
    D = agent.D
    # nguồn của RAG mang điểm rerank; nguồn của công cụ chính xác là nút khớp tuyệt đối (điểm 1)
    scores = out.get("scores") or [1.0] * len(out["sources"])
    srcs = [{"id": sid, "text": D.meta[sid]["text"], "score": float(sc)}
            for sid, sc in zip(out["sources"], scores) if sid in D.meta]
    trace = [{"step": STEP.get(s[0], s[0].upper()), "text": s[1] + (f" ({s[2]} ms)" if len(s) > 2 else "")}
             for s in out["trace"]]
    return {"answer": out["answer"], "sources": srcs, "tools": out["tools"], "trace": trace,
            "model": model_id or "agent", "latency_ms": 1000 * (time.perf_counter() - t0)}
