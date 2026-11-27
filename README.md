# Hệ đa tác tử sàng lọc & hội chẩn đái tháo đường (`MultiAgent`)

Project 01 — môn Phát triển Hệ thống thông minh · Nguyễn Tự Kiên (B23DCCN465) · Nhóm 07.

Chatbot tiếng Việt giúp **ước tính nguy cơ đái tháo đường típ 2**, giải thích lý do và đưa lời khuyên. Bên trong là **9 tác tử**
trao đổi qua `MessageBus` (FIPA-ACL), kết hợp:

- **Mô hình dự đoán**: ensemble HistGradientBoosting (ML) + MLP PyTorch (DL) trên **CDC NHANES 2011–2018** và **CDC BRFSS 2021–2023**,
  kèm ResNet-18 (ảnh đáy mắt), LSTM (chuỗi đường huyết CGM) và Q-learning (gợi ý can thiệp).
- **Cơ sở tri thức (KB)**: PDF **QĐ 5481/QĐ-BYT** + 3 link (**Vinmec**, **Long Châu**, **ADA 2024**) → Qdrant (vector) + SQLite FTS5 (từ khóa)
  + **Neo4j** (đồ thị tri thức, xuất ra `db_diabet.cypher`).
- **Hội đồng tranh biện (+/−)**: khi mô hình và xét nghiệm lệch nhau, tác tử phản biện chất vấn ngược các tác tử khác.
- **LLM nội bộ**: Ollama `qwen2.5:3b` chạy GPU trong container; mọi tác tử có nhánh dự phòng bằng luật khi LLM tắt.

> ⚠️ Kết quả chỉ mang tính tương đối, không thay thế chẩn đoán của bác sĩ.

---

## 1. Chạy nhanh

```bash
cp .env.example .env                      # điền NEO4J_PASSWORD, CLOUDFLARE_TUNNEL_TOKEN (nếu dùng tunnel)
docker compose up -d                      # neo4j, qdrant, seaweedfs, ollama (GPU), backend :8000, cloudflared
docker compose exec ollama ollama pull qwen2.5:3b
docker compose exec backend python scripts/ingest_kb.py     # nạp KB: PDF + link → Qdrant + Neo4j
```

- Giao diện: <http://localhost:8000> · API docs: <http://localhost:8000/docs> · Neo4j Browser: <http://localhost:7474>
- GPU cho Ollama cần `nvidia-container-toolkit` trên máy host.
- Phát triển giao diện: `cd web && npm install && npm run dev` (Vite :5173, proxy `/api` → :8000). `npm run build` ghi vào `web/dist` mà backend phục vụ.
- Kiểm thử: `.venv/bin/python -m pytest test -q` (64 test) · `cd web && npm test` (15 test).

## 2. Cấu trúc thư mục

```text
app.py                     FastAPI: REST + stream NDJSON + phục vụ web/dist
docker-compose.yml         neo4j, qdrant, seaweedfs, ollama (GPU), backend, cloudflared
src/agents/                9 tác tử (orchestrator.py + 8 tác tử chuyên trách)
src/communication/         AgentMessage, BaseAgent, MessageBus (FIPA-ACL)
src/models/                GBDT, MLP, LSTM, ResNet-18, Q-learning, mô tả bộ dữ liệu (datasets.py)
src/knowledge/             nạp KB (ingest.py), embedding, liên kết thực thể, tiêu chuẩn ADA, câu hỏi làm rõ
src/storage/               SQLite/Postgres, Qdrant, Neo4j/AGE, SeaweedFS S3
src/api/                   routes_chat.py (chat + stream), routes_session.py, routes_system.py
scripts/                   fetch_nhanes/brfss, train_model, crawl_longchau_vinmec, ingest_kb, export_cypher
data/output/kb/            kb_sources/ (PDF, HTML, Markdown, sources.json, curated_triples.json) + db_diabet.cypher
notebook/                  10 notebook thực nghiệm + model/ (11 artifact, evaluation_leaderboard.csv)
web/                       React + Vite (giao diện "sổ tay kỹ thuật")
docs/                      Báo cáo LaTeX Project 01 (main.tex → project01_CT_07_KienNT_v3.pdf)
test/                      Pytest
```

Dữ liệu thô (`data/download`), dữ liệu huấn luyện (`data/output/datasets`) và trọng số mô hình (`notebook/model`) không đưa lên Git
vì dung lượng lớn; tái tạo bằng `scripts/fetch_*.py` và `scripts/train_model.py`.

---

## 3. Hỏi đáp vấn đáp (8 câu thầy hay hỏi)

### Câu 1. Có mấy loại agent? Em dùng agent gì, ở đâu trong code? Cảm nhận khi code agent?

**Lý thuyết** (Russell & Norvig) có 5 loại tác tử: (1) phản xạ đơn giản, (2) phản xạ có mô hình, (3) hướng mục tiêu, (4) hướng tiện ích,
(5) tác tử học. Theo kiến trúc còn chia: phản ứng (reactive), suy diễn (deliberative/BDI) và lai (hybrid).

**Hệ của em: 1 Orchestrator + 8 tác tử chuyên trách = 9 tác tử**, khởi tạo trong `OrchestratorAgent.__init__`
([src/agents/orchestrator.py:41](src/agents/orchestrator.py#L41)). Mỗi tác tử khai báo các loại thông điệp nó xử lý trong thuộc tính `handles`:

| # | Tác tử | File | Việc chính | Loại tác tử |
|---|---|---|---|---|
| 1 | OrchestratorAgent | [orchestrator.py:47](src/agents/orchestrator.py#L47) `run_turn` | Phân loại ý định, lập kế hoạch động, điều phối | Hướng mục tiêu |
| 2 | ConversationAgent | [conversation_agent.py:52](src/agents/conversation_agent.py#L52) | Bóc chỉ số từ tin nhắn (LLM + regex) | Phản xạ có mô hình |
| 3 | DocumentAgent | [document_agent.py:24](src/agents/document_agent.py#L24) | Đọc PDF / ảnh / DOCX / CSV xét nghiệm | Phản xạ có mô hình |
| 4 | ClarificationAgent | [clarification_agent.py:29](src/agents/clarification_agent.py#L29) | Kiểm tra phạm vi (thai phụ, trẻ em…), đặt câu hỏi làm rõ | Hướng mục tiêu |
| 5 | PredictionAgent | [prediction_agent.py:32](src/agents/prediction_agent.py#L32) | Chạy mô hình ML/DL, giải trình khi bị chất vấn | Dùng mô hình đã học |
| 6 | KnowledgeAgent | [knowledge_agent.py:27](src/agents/knowledge_agent.py#L27) | Hybrid RAG + GraphRAG, đối chiếu ngưỡng ADA/BYT | Suy diễn (deliberative) |
| 7 | ConsensusCriticAgent | [consensus_critic_agent.py:53](src/agents/consensus_critic_agent.py#L53) | Hội đồng tranh biện +/−, chất vấn ngược | Hướng tiện ích |
| 8 | TreatmentRLAgent | [treatment_rl_agent.py:17](src/agents/treatment_rl_agent.py#L17) | Gợi ý can thiệp bằng Q-learning | Tác tử học |
| 9 | ResponseAgent | [response_agent.py:29](src/agents/response_agent.py#L29) | Viết câu trả lời cho người dùng | Phản xạ |

**Cách các tác tử nói chuyện**: không gọi hàm của nhau, mọi tương tác là một `AgentMessage` có *performative* kiểu FIPA-ACL
(`REQUEST, INFORM, CHALLENGE, RESPONSE, FAILURE` — [message_bus.py:21](src/communication/message_bus.py#L21)).
`BaseAgent.ask()` ([message_bus.py:76](src/communication/message_bus.py#L76)) gửi thư, `MessageBus.send()`
([message_bus.py:103](src/communication/message_bus.py#L103)) định tuyến theo `receiver`, kiểm tra `handles`, gọi `handle()` của tác tử nhận
và ghi toàn bộ hội thoại (audit trail) theo `conversation_id`.

**Cảm nhận**: Code agent **khá là hay**. Khác với viết một chương trình tuần tự, em phải nghĩ mỗi tác tử như một "người" có vai trò riêng,
chỉ giao tiếp bằng thông điệp — giống một hội đồng bác sĩ thật. Thú vị nhất là lúc tác tử phản biện **tự chất vấn** tác tử dự đoán khi mô hình
nói "nguy cơ thấp" mà xét nghiệm lại vượt ngưỡng, rồi cả hệ thống đi đến kết luận chung. Khó nhất là phần điều phối: khi nào chạy song song,
khi nào phải hỏi lại người dùng, và chống việc LLM nhỏ "bịa" số liệu.

### Câu 2. Có mấy tập dữ liệu? Gắn dữ liệu ở đâu trong code (CSV, link, PDF)?

**A. Dữ liệu huấn luyện mô hình (CSV)** — khai báo trong `DATASETS` ([src/models/datasets.py:127](src/models/datasets.py#L127)), trường `csv_path`;
`load_dataframe()` nạp CSV vào bảng SQLite rồi đọc ra để train.

| Tập | File | Nguồn / script | Quy mô | Dùng cho |
|---|---|---|---|---|
| NHANES 2011–2018 (CDC) | `data/output/datasets/nhanes_diabetes.csv` | [scripts/fetch_nhanes.py](scripts/fetch_nhanes.py) (tải file XPT của CDC) | 20.646 người ≥ 20 tuổi, 14 đặc trưng, nhãn bằng xét nghiệm | Mô hình chính (GBDT + MLP) |
| BRFSS 2021–2023 (CDC) | `data/output/datasets/brfss_diabetes.csv.gz` | [scripts/fetch_brfss.py](scripts/fetch_brfss.py) | 120.000 mẫu, 13 đặc trưng | Ý kiến thứ hai (GBDT + MLP) |
| RetinaMNIST | (lab tiểu luận) | MedMNIST v2 | 1.600 ảnh đáy mắt | ResNet-18 |
| ShanghaiT1DM + T2DM | (lab tiểu luận) | CGM 15 phút | 112 bệnh nhân | LSTM dự báo đường huyết |

**B. Cơ sở tri thức (KB): 1 PDF + 3 link** — khai báo trong `data/output/kb/kb_sources/sources.json`:

| Nguồn | File trong `kb_sources/` | Link |
|---|---|---|
| **PDF** QĐ 5481/QĐ-BYT (77 trang) | `qd-5481-byt-2020.pdf` | https://syt.hue.gov.vn/gia-dich-vu-ky-thuat/quyet-dinh-so-5481-qd-byt-ngay-30-12-2020-cua-bo-y-te-ve-huong-dan-chan-doan-dieu-tri-dai-thao-duong-tip-2.html |
| **Link 1** Vinmec – phác đồ ĐTĐ típ 2 | `vinmec-phac-do-dtd-tip-2.html` | https://www.vinmec.com/vie/bai-viet/phac-do-dieu-tri-tieu-duong-type-2-vi |
| **Link 2** Long Châu – các nhóm thuốc | `longchau-cac-nhom-thuoc-tieu-duong.html` | https://nhathuoclongchau.com.vn/bai-viet/cac-nhom-thuoc-tieu-duong-va-uu-nhuoc-diem-cua-thuoc-dieu-tri-tieu-duong.html |
| **Link 3** ADA Standards of Care 2024, Section 2 | `ada-2024-sec02-diagnosis.txt` | https://pmc.ncbi.nlm.nih.gov/articles/PMC10725812/ |

Danh sách link crawl thêm (thuốc Long Châu, bài Vinmec) nằm trong [scripts/crawl_longchau_vinmec.py:32](scripts/crawl_longchau_vinmec.py#L32).
Ngoài ra có các bản tóm tắt Markdown (`ada-2024-*.md`, `byt-5481-tom-tat.md`, …) — tổng **16 tài liệu, 384 đoạn**.

**Xem KB khi demo**:
- Đồ thị nút: giao diện → nút **"Xem KB"** ở sidebar phải; hoặc Neo4j Browser <http://localhost:7474>, chạy `MATCH (n)-[r]->(m) RETURN n,r,m`.
- File Cypher: [`data/output/kb/db_diabet.cypher`](data/output/kb/db_diabet.cypher) — **43 nút, 55 cạnh**, 6 nhãn
  (Disease, RiskFactor, Test, Drug, Intervention, Complication), 7 quan hệ. Sinh bằng [scripts/export_cypher.py](scripts/export_cypher.py).
- Nguồn để xây file Cypher: các bộ ba trong `kb_sources/curated_triples.json`; **mỗi bộ ba có `quotes` trích nguyên văn từ PDF / trang web**,
  nên mỗi cạnh trỏ được về đoạn văn gốc (`chunk_ids`).

### Câu 3. Knowledge graph được xây dựng như thế nào?

Toàn bộ nằm trong [src/knowledge/ingest.py](src/knowledge/ingest.py), chạy bằng `python scripts/ingest_kb.py`:

1. **Đọc nguồn**: `load_pdf()` (L107), `load_html()` (L127), `load_markdown()` (L90), `load_url()` (L135); `discover_sources()` (L146) quét `kb_sources/`.
2. **Chia đoạn & nhúng**: `ingest_documents()` (L175) chia theo mục (≤ 1.200 ký tự) → bảng `kb_chunks` (có chỉ mục FTS5)
   → embedding `paraphrase-multilingual-MiniLM-L12-v2` → **Qdrant**. Bản gốc lưu lên SeaweedFS.
3. **Tạo bộ ba (head, relation, tail)**: `load_curated_triples()` (L225) đọc `curated_triples.json`; trường `quotes` được tìm nguyên văn trong
   các đoạn đã nạp để gán `chunk_ids`. Có thể trích thêm bằng LLM: `extract_llm_triples()` (L250).
4. **Ghi vào Neo4j**: `rebuild_graph()` (L269) → `Neo4jGraphStore.upsert_triples()`
   ([graph_store.py:69](src/storage/graph_store.py#L69)) chạy `MERGE (h)-[r:REL]->(x)`.
5. **Xuất file**: `scripts/export_cypher.py` → `db_diabet.cypher` (nạp lại được bằng `cat db_diabet.cypher | cypher-shell`).

### Câu 4. Điều tâm đắc nhất & kịch bản demo

**Tâm đắc nhất: hội đồng tranh biện chạy song song hai nhánh.**
- Nhánh **dữ liệu → ML/DL** (PredictionAgent) và nhánh **KB → RAG** (KnowledgeAgent) chạy đồng thời bằng `asyncio.gather`
  ([orchestrator.py:141](src/agents/orchestrator.py#L141)).
- `ConsensusCriticAgent` phát hiện mâu thuẫn (`_detect_issues`), ví dụ *mô hình báo nguy cơ thấp nhưng đường huyết đã vượt ngưỡng*, rồi gửi
  **CHALLENGE** ngược lại; tác tử bị chất vấn trả lời **RESPONSE**, hội đồng cập nhật kết luận — xét nghiệm được ưu tiên hơn mô hình dân số.
- Giao diện thấy được toàn bộ quá trình: sidebar **Luồng xử lý** vẽ thông điệp chạy qua 9 tác tử và Qdrant/Neo4j theo thời gian thực.

**Demo (≈ 3 phút)**:
1. Mở trang → form hồ sơ hiện sẵn → nhập tuổi, chiều cao, cân nặng, chọn "Chưa có xét nghiệm" → **Hoàn thành**.
2. Chat: *"Tôi có dấu hiệu bị tiểu đường không?"* → kết quả **ước tính tương đối** + hỏi thêm đường huyết.
3. Trả lời: *"chắc khoảng 7,2 mmol/L"* → sổ tay cập nhật ngay, hội đồng chạy, sidebar hiện CHALLENGE → kết luận "Rất cao — cần xét nghiệm lại".
4. Mở **Chi tiết hội chẩn** (các vòng tranh biện), **Xem KB** (đồ thị + nguồn), **9 tác tử** (vai trò + 11 mô hình).

### Câu 5. Deep learning ở chỗ nào? Hàm nào?

| Mô hình | Code | Hàm chính |
|---|---|---|
| **MLP** dữ liệu bảng (PyTorch) | [src/models/deep_mlp.py](src/models/deep_mlp.py) | `MLPNet` (L22): 128 → 64 → 32, BatchNorm, ReLU, Dropout 0,2; `MLPRisk.train` (L61): AdamW + BCEWithLogitsLoss, dừng sớm theo AUC; `predict_proba` (L57) |
| **LSTM** dự báo đường huyết | [src/models/rnn_temporal.py](src/models/rnn_temporal.py) | `SeqModel` (L23): LSTM 2 lớp một chiều, hidden 64; `GlucoseForecaster` (L38): nhìn 12 điểm → dự báo 15–60 phút |
| **CNN ResNet-18** ảnh đáy mắt | [src/models/cnn_vision.py](src/models/cnn_vision.py) | `build_resnet18` (L39): 5 lớp đầu ra; `RetinaGrader.grad_cam` (L57): bản đồ nhiệt Grad-CAM |

Đầu vào MLP gồm giá trị chuẩn hóa z-score **và** bit "thiếu dữ liệu" cho từng đặc trưng (2d chiều), nên mô hình dự đoán được khi người dùng
chỉ khai một phần thông tin. Lúc huấn luyện dùng *masked-feature augmentation* (`mask_augment`, [datasets.py](src/models/datasets.py)).

### Câu 6. Mô hình nào được train? Gọi ở đâu? Chia việc giữa các tác tử thế nào?

Train bằng `python scripts/train_model.py --dataset nhanes|brfss` ([scripts/train_model.py:43](scripts/train_model.py#L43)).
Kết quả trên tập kiểm thử:

| Mô hình | Dữ liệu | ROC-AUC | Dùng trong hệ |
|---|---|---|---|
| HistGradientBoosting (ML) | NHANES | 0,816 | PredictionAgent – ensemble chính |
| MLP PyTorch (DL) | NHANES | 0,824 | PredictionAgent – ensemble chính |
| Ensemble GBDT + MLP | NHANES | 0,824 | PredictionAgent |
| HistGradientBoosting / MLP | BRFSS | 0,816 / 0,816 | PredictionAgent – ý kiến thứ hai |
| LSTM 2 lớp | Shanghai CGM | RMSE(+30 phút) 13,7 mg/dL | PredictionAgent khi có ≥ 12 lần đo |
| ResNet-18 | RetinaMNIST | QWK 0,70 | PredictionAgent khi có ảnh đáy mắt |
| Q-learning (24 trạng thái, γ = 0,9) | mô phỏng lâm sàng | — | TreatmentRLAgent |
| Logistic Regression, Random Forest, Extra Trees, TabNet | NHANES | 0,81–0,82 | Chỉ thử nghiệm để so sánh (notebook) |

**Nơi gọi mô hình**:
- `PredictionAgent.__init__` ([prediction_agent.py:35](src/agents/prediction_agent.py#L35)) nạp `GradientBoostingRisk.load` + `MLPRisk.load` (L39) và cặp BRFSS (L47).
- `_ensemble()` (L78) gọi `risk_report()` ([tabular_ml.py:58](src/models/tabular_ml.py#L58)) cho từng mô hình, lấy **trung bình hai xác suất**,
  tính nguy cơ tương đối và độ lệch ML–DL; `_large_scale()` (L107) chạy cặp BRFSS.
- Q-learning: `DiabetesQLearningEngine` ([qlearning_treatment.py:77](src/models/qlearning_treatment.py#L77)), gọi từ `TreatmentRLAgent`.

**Lưu ý khi bị hỏi "sao mô hình không dùng đường huyết?"**: nhãn NHANES được gán bằng chính glucose/HbA1c
(`diabetes = đã chẩn đoán | HbA1c ≥ 6,5 | FPG ≥ 126`), đưa vào sẽ **rò rỉ nhãn**. Vì vậy mô hình chấm điểm *yếu tố nguy cơ*, còn
KnowledgeAgent đối chiếu xét nghiệm với ngưỡng ADA; hai nhánh lệch nhau thì hội đồng chất vấn.

**Chia việc**: xem bảng ở Câu 1 — mỗi tác tử chỉ xử lý các `msg_type` trong `handles` của nó; `MessageBus` từ chối thông điệp sai loại
(trả `FAILURE`). Một lượt đầy đủ: tiếp nhận song song → làm rõ → dự đoán ‖ tri thức → tranh biện → RL → trả lời.

### Câu 7. Tác tử kết nối RAG và file Cypher như thế nào?

Tất cả ở [src/agents/knowledge_agent.py](src/agents/knowledge_agent.py):

```text
KnowledgeAgent._search()  (L60) ── embed câu hỏi ──► VectorStore.hybrid_search()  (vector_store.py:146)
                                                     ├─ Qdrant (vector, cosine)
                                                     ├─ SQLite FTS5 (từ khóa, BM25)
                                                     └─ gộp bằng Reciprocal Rank Fusion _rrf() (L66, k = 60)
KnowledgeAgent._graph()   (L68) ── link_entities() ──► GraphStore.neighborhood()   (graph_store.py:88)
                                                     └─ Cypher: MATCH p = (s)-[*1..2]-(t) WHERE s.name IN $names
                                  ── chunk_ids của cạnh ──► VectorStore.chunks_by_ids() (L72)  ← nối đồ thị về đoạn văn gốc
KnowledgeAgent._evidence() (L91) / _answer_question() (L44): gộp đoạn văn từ vector search + từ đồ thị (bỏ trùng) → trích dẫn
```

Đồ thị Neo4j mà `neighborhood()` duyệt chính là nội dung của `db_diabet.cypher` (file này là bản xuất của đồ thị). Điểm nối giữa **RAG**
và **đồ thị** là trường `chunk_ids` trên mỗi cạnh: khi duyệt tới cạnh "Đái tháo đường típ 2 —có yếu tố nguy cơ→ Thừa cân", tác tử lấy luôn
đoạn văn gốc trong PDF/web làm bằng chứng.

### Câu 8. Từ giao diện, thông tin xuống tác tử ở đâu? Hàm nào?

```text
web/src/App.jsx:114                 handleSendMessage → sendChatStream(sessionId, text, files, onEvent)
web/src/services/llmService.js:42   sendChatStream → fetch('/api/chat/stream') (hoặc /api/chat/upload/stream khi có file)
src/api/routes_chat.py:50           POST /api/chat/stream → _stream()  (L64, trả NDJSON)
src/api/routes_chat.py:139          _run_turn(): nạp hồ sơ + lịch sử phiên, lưu file lên S3
src/api/routes_chat.py:173          get_orchestrator().run_turn(profile, text, documents, history, last_assessment)
src/agents/orchestrator.py:47       run_turn(): self.ask("ConversationAgent", "EXTRACT_FROM_TEXT", …) …
src/communication/message_bus.py:76 BaseAgent.ask → MessageBus.send (L103) → agent.handle(msg)
```

Trong lúc chạy, `stream_sink` ([message_bus.py:25](src/communication/message_bus.py#L25)) đẩy từng thông điệp lên giao diện dưới dạng sự kiện
NDJSON: `message` (vẽ luồng tác tử), `graph` (đồ thị vừa duyệt), `profile` (hồ sơ cập nhật ngay sau bước tiếp nhận), `final` (câu trả lời).
Form hồ sơ lưu thẳng qua `PUT /api/sessions/{id}/profile` ([routes_session.py](src/api/routes_session.py)).

---

Tài liệu thêm: [data/README.md](data/README.md) (bộ dữ liệu) · [src/README.md](src/README.md) (kiến trúc mã nguồn) ·
[docs/errata_bao_cao_v2.md](docs/errata_bao_cao_v2.md) (các điểm đã sửa so với báo cáo v2).
