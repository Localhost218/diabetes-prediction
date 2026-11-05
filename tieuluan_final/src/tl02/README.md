# Tiểu luận 02 — Knowledge Graph, RAG và AI Agent

Mã nguồn của báo cáo [`Report/TieuLuan 02/tieuluan02_CT_07_nghiand.pdf`](../../Report/TieuLuan%2002/tieuluan02_CT_07_nghiand.pdf).

| Chương | Nội dung | Mã chính |
|---|---|---|
| 1. Mở đầu | KG, RAG, LLM và ảo giác | — |
| 2. Kiến trúc KG | Ontology, suy diễn RDFS, xuất RDF và SPARQL, KG từ dữ liệu bảng | `tl2/kg_base.py`, `tl2/kg_tabular.py`, `scripts/build_kgs.py` |
| 3. RAG | 5 cách chunking × 5 bộ truy xuất, HyDE, Multi-Query, rerank, GraphRAG | `tl2/chunking.py`, `tl2/retrieval.py`, `tl2/graphrag.py`, `scripts/run_*.py` |
| 4. Y tế | KG đái tháo đường típ 2, MLP chẩn đoán nguy cơ (100.000 hồ sơ) | `tl2/kg_health.py`, `scripts/train_diabetes.py` |
| 5. Tài chính | KG nhân quả có dấu, suy luận đa bước, kiểm chứng cạnh bằng dữ liệu, BiLSTM dự báo VN30 và vàng SJC | `tl2/kg_finance.py`, `scripts/market_analysis.py`, `scripts/train_forecast.py` |
| 6. AI Agent | Tác tử ReAct 7 công cụ, bộ kiểm chứng trích dẫn và số liệu, Web App | `tl2/agent.py`, `tl2/llm.py`, `scripts/run_agent.py`, `app/` |

## Bố cục

```
tl2/             gói dùng chung
  kg_base.py  kg_health.py  kg_finance.py  kg_tabular.py     đồ thị tri thức, ontology, suy diễn, RDF
  corpus.py   chunking.py   retrieval.py   graphrag.py       kho văn bản, phân mảnh, truy xuất, GraphRAG
  llm.py      agent.py                                       LLM cục bộ, tác tử ReAct
  data.py     dl_models.py  common.py                        dữ liệu thị trường, MLP/BiLSTM, đường dẫn, thiết bị
scripts/         các thí nghiệm (mỗi tệp một tiến trình, ghi outputs/metrics/*.json)
app/             Web App Flask (app.py, templates/, static/) + screenshots.py
data/            corpus/ (tài liệu y tế, tài chính), market/ (giá CSV), qa/ (bộ câu hỏi), tabular/
models/          MLP, BiLSTM đã huấn luyện; models/hf/ chứa mô hình HuggingFace (tải bằng lệnh bên dưới)
outputs/         metrics/*.json, logs/
```

## Dựng lại từ đầu

```bash
pip install -r requirements.txt
python scripts/download_models.py     # e5-small/base, MiniLM, bge-reranker-v2-m3, Qwen2.5-1.5B-Instruct (~8 GB)
python scripts/download_data.py       # tùy chọn: CSV trong data/market/ là bản đã dùng cho báo cáo

# CPU: KG, RDF, SPARQL, suy diễn; phân tích thị trường
python scripts/build_kgs.py
python scripts/market_analysis.py
# GPU: huấn luyện MLP và BiLSTM (Logistic Regression, Ridge đối chứng chạy CPU)
python scripts/train_diabetes.py
python scripts/train_forecast.py
python scripts/forecast_live.py
# GPU: embedding, rerank, LLM — chạy lần lượt, mỗi tiến trình một LLM trên GPU
python scripts/run_retrieval.py
python scripts/run_advanced.py
python scripts/run_graphrag.py
python scripts/run_agent.py
# CPU: hình dữ liệu cho báo cáo
python scripts/make_figures.py

# Web App và ảnh chụp giao diện
python app/app.py                     # http://127.0.0.1:8092/
python app/screenshots.py
# Báo cáo
cd "../../Report/TieuLuan 02" && python build.py      # --kiem-tra: một lượt LaTeX, chỉ in lỗi
```

Phân bổ phần cứng: GPU (RTX 4060 Laptop) chạy mọi thứ có mạng nơ-ron, gồm Qwen2.5-1.5B bf16, các mô hình
embedding, reranker bf16, MLP và BiLSTM. CPU chạy đồ thị tri thức (NetworkX, rdflib), BM25, phân tích tương quan,
các mô hình đối chứng của scikit-learn, vẽ hình và biên dịch LaTeX, nên các bước CPU có thể chạy song song với
một tiến trình GPU.

Trên Windows, tiến trình dùng CUDA đôi khi thoát với mã `0xC0000409` **sau khi** đã ghi xong kết quả. Hãy kiểm
tra tệp trong `outputs/metrics/` thay vì mã thoát. `tl2/common.py` đặt `HF_DEACTIVATE_ASYNC_LOAD=1` vì khi bộ
nhớ commit thấp, việc nạp trọng số đa luồng của transformers gây lỗi truy cập bộ nhớ (`0xC0000005`).
