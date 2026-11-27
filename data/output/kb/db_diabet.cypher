// Knowledge Graph đái tháo đường — xuất trực tiếp từ Neo4j bằng scripts/export_cypher.py
// Thời điểm xuất: 2026-11-13 10:25
// 43 nút (Complication: 5, Disease: 3, Drug: 12, Intervention: 5, RiskFactor: 10, Test: 8)
// 55 cạnh (CAUSES: 7, CONTRAINDICATED_IN: 3, DIAGNOSED_BY: 7, HAS_RISK_FACTOR: 11, PREVENTED_BY: 2, SCREENED_BY: 4, TREATED_BY: 21)
// Mỗi cạnh có source_doc, evidence và chunk_ids trỏ về đoạn văn trong kb_chunks (RAG).

// 1. Ràng buộc duy nhất
CREATE CONSTRAINT complication_name IF NOT EXISTS FOR (n:Complication) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (n:Disease) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (n:Drug) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT intervention_name IF NOT EXISTS FOR (n:Intervention) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT riskfactor_name IF NOT EXISTS FOR (n:RiskFactor) REQUIRE n.name IS UNIQUE;
CREATE CONSTRAINT test_name IF NOT EXISTS FOR (n:Test) REQUIRE n.name IS UNIQUE;

// 2. Nút
MERGE (:Complication {name: "Bệnh thần kinh ngoại biên"});
MERGE (:Complication {name: "Bệnh thận ĐTĐ"});
MERGE (:Complication {name: "Bệnh tim mạch xơ vữa"});
MERGE (:Complication {name: "Bệnh võng mạc ĐTĐ"});
MERGE (:Complication {name: "Hạ đường huyết"});
MERGE (:Disease {name: "Suy thận nặng (eGFR < 30)"});
MERGE (:Disease {name: "Tiền đái tháo đường"});
MERGE (:Disease {name: "Đái tháo đường típ 2"});
MERGE (:Drug {name: "Dapagliflozin (Forxiga 10mg)"});
MERGE (:Drug {name: "Empagliflozin (Jardiance 25mg)"});
MERGE (:Drug {name: "Gliclazide MR (Diamicron MR 60mg)"});
MERGE (:Drug {name: "Insulin"});
MERGE (:Drug {name: "Insulin Glargine (Lantus SoloStar)"});
MERGE (:Drug {name: "Linagliptin (Trajenta 5mg)"});
MERGE (:Drug {name: "Metformin"});
MERGE (:Drug {name: "Metformin (Glucophage 500mg/850mg)"});
MERGE (:Drug {name: "Sitagliptin (Januvia 100mg)"});
MERGE (:Drug {name: "Statin"});
MERGE (:Drug {name: "Đồng vận GLP-1"});
MERGE (:Drug {name: "Ức chế SGLT2"});
MERGE (:Intervention {name: "Giảm ≥ 7% cân nặng"});
MERGE (:Intervention {name: "Phối hợp Metformin + SGLT2i (Forxiga/Jardiance)"});
MERGE (:Intervention {name: "Phối hợp Metformin + SGLT2i/GLP-1 RA"});
MERGE (:Intervention {name: "Thay đổi lối sống"});
MERGE (:Intervention {name: "Vận động 150 phút/tuần"});
MERGE (:RiskFactor {name: "Béo bụng"});
MERGE (:RiskFactor {name: "Hút thuốc lá"});
MERGE (:RiskFactor {name: "Rối loạn lipid máu"});
MERGE (:RiskFactor {name: "Thừa cân/béo phì"});
MERGE (:RiskFactor {name: "Tiền sử bệnh tim mạch"});
MERGE (:RiskFactor {name: "Tiền sử gia đình ĐTĐ"});
MERGE (:RiskFactor {name: "Tiền đái tháo đường"});
MERGE (:RiskFactor {name: "Tuổi ≥ 35"});
MERGE (:RiskFactor {name: "Tăng huyết áp"});
MERGE (:RiskFactor {name: "Ít vận động"});
MERGE (:Test {name: "Bộ mỡ máu và huyết áp"});
MERGE (:Test {name: "Glucose bất kỳ"});
MERGE (:Test {name: "Glucose huyết tương lúc đói"});
MERGE (:Test {name: "HbA1c"});
MERGE (:Test {name: "Khám bàn chân (monofilament 10 g)"});
MERGE (:Test {name: "Nghiệm pháp dung nạp glucose (OGTT)"});
MERGE (:Test {name: "Soi đáy mắt / chụp võng mạc"});
MERGE (:Test {name: "uACR và eGFR"});

// 3. Cạnh
MATCH (h:Drug {name: "Gliclazide MR (Diamicron MR 60mg)"}), (t:Complication {name: "Hạ đường huyết"}) MERGE (h)-[r:CAUSES]->(t) SET r.chunk_ids = [48], r.evidence = "Sulfonylurea (Diamicron MR 60mg) có nguy cơ gây hạ đường huyết (vã mồ hôi, run tay, đói cồn cào) nếu nhịn ăn sau khi uống thuốc.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Drug {name: "Insulin"}), (t:Complication {name: "Hạ đường huyết"}) MERGE (h)-[r:CAUSES]->(t) SET r.chunk_ids = [26], r.evidence = "Insulin có nguy cơ hạ đường huyết, cần tự theo dõi đường huyết.", r.source_doc = "dieu-tri-thuoc";
MATCH (h:RiskFactor {name: "Tăng huyết áp"}), (t:Complication {name: "Bệnh tim mạch xơ vữa"}) MERGE (h)-[r:CAUSES]->(t) SET r.chunk_ids = [6], r.evidence = "Huyết áp mục tiêu < 130/80 mmHg ở người đái tháo đường.", r.source_doc = "ada-2024-muc-tieu";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Complication {name: "Bệnh thần kinh ngoại biên"}) MERGE (h)-[r:CAUSES]->(t) SET r.chunk_ids = [14], r.evidence = "Tầm soát bệnh thần kinh ngoại biên khi chẩn đoán và hằng năm.", r.source_doc = "bien-chung-tam-soat";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Complication {name: "Bệnh thận ĐTĐ"}) MERGE (h)-[r:CAUSES]->(t) SET r.chunk_ids = [13], r.evidence = "Đánh giá uACR và eGFR ít nhất mỗi năm.", r.source_doc = "bien-chung-tam-soat";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Complication {name: "Bệnh tim mạch xơ vữa"}) MERGE (h)-[r:CAUSES]->(t) SET r.chunk_ids = [15], r.evidence = "Bệnh tim mạch là nguyên nhân tử vong hàng đầu ở người đái tháo đường.", r.source_doc = "bien-chung-tam-soat";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Complication {name: "Bệnh võng mạc ĐTĐ"}) MERGE (h)-[r:CAUSES]->(t) SET r.chunk_ids = [12], r.evidence = "Khám mắt giãn đồng tử hoặc chụp ảnh võng mạc ngay khi chẩn đoán.", r.source_doc = "bien-chung-tam-soat";
MATCH (h:Drug {name: "Gliclazide MR (Diamicron MR 60mg)"}), (t:Disease {name: "Suy thận nặng (eGFR < 30)"}) MERGE (h)-[r:CONTRAINDICATED_IN]->(t) SET r.chunk_ids = [48], r.evidence = "Cần tránh hoặc chống chỉ định Sulfonylurea ở bệnh nhân suy thận nặng do nguy cơ tích lũy thuốc gây hạ đường huyết kéo dài.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Drug {name: "Metformin"}), (t:Disease {name: "Suy thận nặng (eGFR < 30)"}) MERGE (h)-[r:CONTRAINDICATED_IN]->(t) SET r.chunk_ids = [24, 330], r.evidence = "Chống chỉ định khi eGFR < 30; không khởi trị khi eGFR 30–45.", r.source_doc = "dieu-tri-thuoc";
MATCH (h:Drug {name: "Metformin (Glucophage 500mg/850mg)"}), (t:Disease {name: "Suy thận nặng (eGFR < 30)"}) MERGE (h)-[r:CONTRAINDICATED_IN]->(t) SET r.chunk_ids = [44], r.evidence = "Glucophage chống chỉ định tuyệt đối khi suy thận nặng (eGFR < 30 mL/phút/1,73 m²); cần kiểm tra Vitamin B12 định kỳ khi dùng kéo dài.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Disease {name: "Tiền đái tháo đường"}), (t:Test {name: "Glucose huyết tương lúc đói"}) MERGE (h)-[r:DIAGNOSED_BY]->(t) SET r.chunk_ids = [8], r.evidence = "FPG 100–125 mg/dL (5,6–6,9 mmol/L).", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:Disease {name: "Tiền đái tháo đường"}), (t:Test {name: "HbA1c"}) MERGE (h)-[r:DIAGNOSED_BY]->(t) SET r.chunk_ids = [8], r.evidence = "HbA1c 5,7–6,4%.", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:Disease {name: "Tiền đái tháo đường"}), (t:Test {name: "Nghiệm pháp dung nạp glucose (OGTT)"}) MERGE (h)-[r:DIAGNOSED_BY]->(t) SET r.chunk_ids = [8], r.evidence = "Glucose 2 giờ sau OGTT 140–199 mg/dL.", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Test {name: "Glucose bất kỳ"}) MERGE (h)-[r:DIAGNOSED_BY]->(t) SET r.chunk_ids = [1], r.evidence = "Glucose bất kỳ ≥ 200 mg/dL kèm triệu chứng tăng đường huyết.", r.source_doc = "ada-2024-chan-doan";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Test {name: "Glucose huyết tương lúc đói"}) MERGE (h)-[r:DIAGNOSED_BY]->(t) SET r.chunk_ids = [1, 258, 55], r.evidence = "FPG ≥ 126 mg/dL (7,0 mmol/L).", r.source_doc = "ada-2024-chan-doan";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Test {name: "HbA1c"}) MERGE (h)-[r:DIAGNOSED_BY]->(t) SET r.chunk_ids = [1, 258, 55], r.evidence = "HbA1c ≥ 6,5% (48 mmol/mol).", r.source_doc = "ada-2024-chan-doan";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Test {name: "Nghiệm pháp dung nạp glucose (OGTT)"}) MERGE (h)-[r:DIAGNOSED_BY]->(t) SET r.chunk_ids = [1, 258, 55], r.evidence = "Glucose 2 giờ sau OGTT 75 g ≥ 200 mg/dL.", r.source_doc = "ada-2024-chan-doan";
MATCH (h:Disease {name: "Tiền đái tháo đường"}), (t:RiskFactor {name: "Thừa cân/béo phì"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [9], r.evidence = "Mục tiêu giảm ít nhất 7% cân nặng ban đầu.", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Béo bụng"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [33, 262], r.evidence = "Béo bụng khi vòng eo ≥ 90 cm ở nam và ≥ 80 cm ở nữ (IDF, người châu Á).", r.source_doc = "sang-loc-nguoi-viet";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Hút thuốc lá"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [30], r.evidence = "Khuyên mọi người bệnh ngừng hút thuốc lá.", r.source_doc = "loi-song-dinh-duong";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Rối loạn lipid máu"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [32], r.evidence = "HDL-C < 35 mg/dL và/hoặc triglycerid > 250 mg/dL.", r.source_doc = "sang-loc-nguoi-viet";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Thừa cân/béo phì"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [32, 258, 92], r.evidence = "Người trưởng thành có BMI ≥ 23 kg/m² (người châu Á) kèm yếu tố nguy cơ cần được sàng lọc.", r.source_doc = "sang-loc-nguoi-viet";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Tiền sử bệnh tim mạch"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [32], r.evidence = "Tiền sử bệnh tim mạch.", r.source_doc = "sang-loc-nguoi-viet";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Tiền sử gia đình ĐTĐ"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [32, 258, 92], r.evidence = "Có bố mẹ, anh chị em ruột mắc đái tháo đường.", r.source_doc = "sang-loc-nguoi-viet";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Tiền đái tháo đường"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [8], r.evidence = "Tiền đái tháo đường làm tăng nguy cơ tiến triển thành đái tháo đường típ 2.", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Tuổi ≥ 35"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [31], r.evidence = "Mọi người từ 35 tuổi trở lên cần được sàng lọc.", r.source_doc = "sang-loc-nguoi-viet";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Tăng huyết áp"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [32], r.evidence = "Tăng huyết áp ≥ 130/80 mmHg hoặc đang điều trị thuốc hạ áp.", r.source_doc = "sang-loc-nguoi-viet";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:RiskFactor {name: "Ít vận động"}) MERGE (h)-[r:HAS_RISK_FACTOR]->(t) SET r.chunk_ids = [32], r.evidence = "Ít hoạt động thể lực.", r.source_doc = "sang-loc-nguoi-viet";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Intervention {name: "Giảm ≥ 7% cân nặng"}) MERGE (h)-[r:PREVENTED_BY]->(t) SET r.chunk_ids = [9], r.evidence = "Can thiệp lối sống trong DPP giảm khoảng 58% tỷ lệ mới mắc sau 3 năm.", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Intervention {name: "Vận động 150 phút/tuần"}) MERGE (h)-[r:PREVENTED_BY]->(t) SET r.chunk_ids = [9, 288], r.evidence = "Hoạt động thể lực cường độ trung bình tối thiểu 150 phút mỗi tuần.", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:Complication {name: "Bệnh thần kinh ngoại biên"}), (t:Test {name: "Khám bàn chân (monofilament 10 g)"}) MERGE (h)-[r:SCREENED_BY]->(t) SET r.chunk_ids = [14, 268], r.evidence = "Khám bàn chân toàn diện ít nhất mỗi năm, gồm thử cảm giác bằng monofilament 10 g.", r.source_doc = "bien-chung-tam-soat";
MATCH (h:Complication {name: "Bệnh thận ĐTĐ"}), (t:Test {name: "uACR và eGFR"}) MERGE (h)-[r:SCREENED_BY]->(t) SET r.chunk_ids = [13], r.evidence = "Đánh giá tỷ số albumin/creatinin niệu và eGFR ít nhất mỗi năm.", r.source_doc = "bien-chung-tam-soat";
MATCH (h:Complication {name: "Bệnh tim mạch xơ vữa"}), (t:Test {name: "Bộ mỡ máu và huyết áp"}) MERGE (h)-[r:SCREENED_BY]->(t) SET r.chunk_ids = [15], r.evidence = "Đánh giá yếu tố nguy cơ tim mạch ít nhất mỗi năm.", r.source_doc = "bien-chung-tam-soat";
MATCH (h:Complication {name: "Bệnh võng mạc ĐTĐ"}), (t:Test {name: "Soi đáy mắt / chụp võng mạc"}) MERGE (h)-[r:SCREENED_BY]->(t) SET r.chunk_ids = [12], r.evidence = "Khám lại mỗi 1–2 năm nếu chưa có tổn thương; ít nhất mỗi năm nếu đã có.", r.source_doc = "bien-chung-tam-soat";
MATCH (h:Complication {name: "Bệnh thận ĐTĐ"}), (t:Drug {name: "Empagliflozin (Jardiance 25mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [45], r.evidence = "Ưu tiên cho bệnh nhân bệnh thận mạn (CKD có albumin niệu hoặc eGFR 20–60).", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Complication {name: "Bệnh thận ĐTĐ"}), (t:Drug {name: "Ức chế SGLT2"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [25, 330], r.evidence = "Ức chế SGLT2 giảm tiến triển bệnh thận mạn (eGFR 20–60 hoặc có albumin niệu).", r.source_doc = "dieu-tri-thuoc";
MATCH (h:Complication {name: "Bệnh tim mạch xơ vữa"}), (t:Drug {name: "Dapagliflozin (Forxiga 10mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [45], r.evidence = "Chỉ định ưu tiên cho bệnh nhân ĐTĐ típ 2 kèm bệnh tim mạch xơ vữa (ASCVD) hoặc suy tim.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Complication {name: "Bệnh tim mạch xơ vữa"}), (t:Drug {name: "Đồng vận GLP-1"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [25], r.evidence = "Dùng GLP-1 RA có lợi ích tim mạch, độc lập với HbA1c.", r.source_doc = "dieu-tri-thuoc";
MATCH (h:Complication {name: "Bệnh tim mạch xơ vữa"}), (t:Drug {name: "Ức chế SGLT2"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [25], r.evidence = "Dùng ức chế SGLT2 có lợi ích tim mạch đã chứng minh.", r.source_doc = "dieu-tri-thuoc";
MATCH (h:RiskFactor {name: "Rối loạn lipid máu"}), (t:Drug {name: "Statin"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [7, 298], r.evidence = "Người 40–75 tuổi không có bệnh tim mạch xơ vữa: statin cường độ trung bình.", r.source_doc = "ada-2024-muc-tieu";
MATCH (h:Disease {name: "Suy thận nặng (eGFR < 30)"}), (t:Drug {name: "Linagliptin (Trajenta 5mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [46], r.evidence = "Linagliptin (Trajenta 5mg) thải trừ qua đường mật – ruột nên không cần chỉnh liều ở mọi mức độ suy thận (kể cả eGFR < 30).", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:RiskFactor {name: "Thừa cân/béo phì"}), (t:Intervention {name: "Phối hợp Metformin + SGLT2i/GLP-1 RA"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [49], r.evidence = "Ở người đái tháo đường thừa cân/béo phì (BMI ≥ 23), ưu tiên phác đồ GLP-1 RA hoặc SGLT2i phối hợp Metformin để vừa kiểm soát đường huyết vừa giảm cân.", r.source_doc = "vinmec-phac-do-dieu-tri-ket-hop-ada";
MATCH (h:Disease {name: "Tiền đái tháo đường"}), (t:Drug {name: "Metformin"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [10], r.evidence = "Cân nhắc metformin ở người 25–59 tuổi có BMI ≥ 35, FPG ≥ 110 hoặc HbA1c ≥ 6,0%.", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:Disease {name: "Tiền đái tháo đường"}), (t:Intervention {name: "Thay đổi lối sống"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [9], r.evidence = "Chương trình DPP: giảm ≥ 7% cân nặng, vận động 150 phút/tuần.", r.source_doc = "ada-2024-tien-dtd";
MATCH (h:RiskFactor {name: "Tăng huyết áp"}), (t:Intervention {name: "Phối hợp Metformin + SGLT2i (Forxiga/Jardiance)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [50], r.evidence = "Phác đồ Metformin + SGLT2i tối ưu cho bệnh nhân ĐTĐ kèm tăng huyết áp hoặc bệnh tim mạch.", r.source_doc = "vinmec-phac-do-dieu-tri-ket-hop-ada";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Dapagliflozin (Forxiga 10mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [45, 232], r.evidence = "Ức chế SGLT2 (Forxiga 10mg) giúp thải glucose qua nước tiểu, hạ đường huyết, giảm cân nhẹ (2–3 kg) và hạ huyết áp tâm thu.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Empagliflozin (Jardiance 25mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [45, 232], r.evidence = "Jardiance (10mg/25mg) kiểm soát đường huyết kèm bảo vệ tim mạch và thận vượt trội.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Gliclazide MR (Diamicron MR 60mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [48], r.evidence = "Diamicron MR 30mg/60mg kích thích tế bào beta tụy tiết insulin, hạ HbA1c nhanh, uống ngay trước hoặc trong bữa ăn sáng.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Insulin"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [26, 385], r.evidence = "Cân nhắc insulin khi HbA1c > 10%, glucose ≥ 300 mg/dL hoặc có dị hóa.", r.source_doc = "dieu-tri-thuoc";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Insulin Glargine (Lantus SoloStar)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [47], r.evidence = "Bút tiêm Lantus SoloStar 100IU/ml là insulin nền tác dụng kéo dài suốt 24 giờ, ít tạo đỉnh, kiểm soát đường huyết lúc đói.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Linagliptin (Trajenta 5mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [46, 234], r.evidence = "Trajenta 5mg kiểm soát đường huyết an toàn, không gây tăng cân hay hạ đường huyết.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Metformin"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [24, 330, 381], r.evidence = "Metformin là thuốc nền tảng khi khởi trị.", r.source_doc = "dieu-tri-thuoc";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Metformin (Glucophage 500mg/850mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [44, 381, 232], r.evidence = "Metformin (Glucophage 500mg/850mg, Glucophage XR) là thuốc nền tảng đầu tay, uống trong hoặc ngay sau bữa ăn để giảm tác dụng phụ tiêu hóa.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Drug {name: "Sitagliptin (Januvia 100mg)"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [46, 234], r.evidence = "Ức chế DPP-4 (Januvia 100mg) dung nạp rất tốt, trung tính với cân nặng và nguy cơ hạ đường huyết thấp.", r.source_doc = "longchau-thuoc-dai-thao-duong";
MATCH (h:Disease {name: "Đái tháo đường típ 2"}), (t:Intervention {name: "Thay đổi lối sống"}) MERGE (h)-[r:TREATED_BY]->(t) SET r.chunk_ids = [27], r.evidence = "Ít nhất 150 phút mỗi tuần hoạt động thể lực; giảm ≥ 5% cân nặng.", r.source_doc = "loi-song-dinh-duong";
