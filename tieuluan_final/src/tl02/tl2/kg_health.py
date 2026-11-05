"""Đồ thị tri thức y tế: quản lý đái tháo đường típ 2 tại Việt Nam.

Ba phân hệ nối với nhau: chỉ số chẩn đoán -> bệnh -> phác đồ thuốc -> tương tác và nguy cơ -> dinh dưỡng, lối sống.
Mỗi cạnh mang `source` là mã tài liệu trong kho tri thức (data/corpus/health) để tác tử trích dẫn được nguồn.
"""
from .kg_base import KnowledgeGraph, Ontology


def health_ontology():
    o = Ontology()
    for c, parent in [
        ("Thing", None), ("MedicalEntity", "Thing"),
        ("Disease", "MedicalEntity"), ("Condition", "MedicalEntity"), ("Complication", "Condition"),
        ("Biomarker", "MedicalEntity"), ("RiskFactor", "MedicalEntity"),
        ("Drug", "MedicalEntity"), ("Biguanide", "Drug"), ("Sulfonylurea", "Drug"), ("DPP4Inhibitor", "Drug"),
        ("SGLT2Inhibitor", "Drug"), ("GLP1Agonist", "Drug"), ("Insulin", "Drug"),
        ("Substance", "MedicalEntity"), ("RiskSubstance", "Substance"),
        ("Food", "Substance"), ("FoodGroup", "Food"), ("SafeFood", "Food"), ("ModerateFood", "Food"),
        ("RiskFood", "Food"), ("Lifestyle", "MedicalEntity"), ("Guideline", "Thing"),
    ]:
        o.add_class(c, parent)
    P = o.add_predicate
    P("TIEU_CHUAN_CHAN_DOAN", "Biomarker", "Disease", "là tiêu chuẩn chẩn đoán")
    P("CHI_DINH_DAU_TAY", "Disease", "Drug", "chỉ định đầu tay", inverse="DIEU_TRI")
    P("DIEU_TRI", "Drug", "Disease", "điều trị")
    P("UU_TIEN_KHI", "Drug", "Condition", "ưu tiên khi có")
    P("CHONG_CHI_DINH_KHI", "Drug", "Condition", "chống chỉ định khi")
    P("TUONG_TAC_NGUY_HIEM", "Drug", "RiskSubstance", "tương tác nguy hiểm với")
    P("TANG_NGUY_CO", "MedicalEntity", "Condition", "làm tăng nguy cơ")
    P("YEU_TO_NGUY_CO", "RiskFactor", "Disease", "là yếu tố nguy cơ của")
    P("BIEN_CHUNG_CUA", "Complication", "Disease", "là biến chứng của")
    P("KHUYEN_NGHI_AN", "Disease", "Food", "khuyến nghị ăn")
    P("HAN_CHE_AN", "Disease", "Food", "khuyến nghị hạn chế")
    P("AN_DINH_LUONG", "Disease", "Food", "ăn có định lượng")
    P("KHUYEN_NGHI_LOI_SONG", "Disease", "Lifestyle", "khuyến nghị lối sống")
    P("LA_MOT_DANG_CUA", "Food", "FoodGroup", "là một dạng của", transitive=True)
    P("TIEN_TRIEN_THANH", "Disease", "Disease", "có thể tiến triển thành")
    P("THEO_HUONG_DAN", "Disease", "Guideline", "theo hướng dẫn")
    P("XU_TRI_BANG", "Complication", "Lifestyle", "xử trí bằng")
    return o


def build_health_kg():
    kg = KnowledgeGraph("VietnamDiabetesKG", health_ontology())
    E = kg.add_entity
    # bệnh và hướng dẫn
    E("DTD_Tip2", "Disease", "Đái tháo đường típ 2", ["đái tháo đường", "tiểu đường", "đtđ", "típ 2", "type 2"], icd10="E11")
    E("DTD_Tip1", "Disease", "Đái tháo đường típ 1", ["típ 1", "type 1"], icd10="E10")
    E("Tien_DTD", "Disease", "Tiền đái tháo đường", ["tiền đái tháo đường", "tiền tiểu đường"], icd10="R73.03")
    E("QD_5481", "Guideline", "Quyết định 5481/QĐ-BYT", ["5481", "bộ y tế"], year=2020)
    # chỉ số chẩn đoán
    E("HbA1c_Cao", "Biomarker", "HbA1c ≥ 6,5%", ["hba1c"], threshold=6.5, unit="%")
    E("Glucose_Doi_Cao", "Biomarker", "Glucose lúc đói ≥ 126 mg/dL", ["glucose lúc đói", "đường huyết đói", "đường huyết lúc đói"], threshold=126, unit="mg/dL")
    E("OGTT_2h_Cao", "Biomarker", "Glucose 2 giờ sau nghiệm pháp ≥ 200 mg/dL", ["nghiệm pháp", "ogtt"], threshold=200, unit="mg/dL")
    E("HbA1c_Tien", "Biomarker", "HbA1c 5,7 - 6,4%", [], low=5.7, high=6.4, unit="%")
    # yếu tố nguy cơ
    E("Beo_Phi", "RiskFactor", "BMI ≥ 25 (béo phì, chuẩn châu Á)", ["béo phì", "thừa cân", "bmi"], threshold=25)
    E("It_Van_Dong", "RiskFactor", "Ít vận động", ["ít vận động"])
    E("Hut_Thuoc", "RiskFactor", "Hút thuốc lá", ["hút thuốc", "thuốc lá"])
    E("Tang_Huyet_Ap", "RiskFactor", "Tăng huyết áp", ["huyết áp", "tăng huyết áp"])
    # tình trạng, biến chứng
    E("Suy_Than_Nang", "Condition", "Suy thận nặng (eGFR < 30)", ["suy thận", "egfr"], egfr_max=30)
    E("Suy_Tim", "Condition", "Suy tim", ["suy tim"])
    E("Benh_Than_Man", "Condition", "Bệnh thận mạn", ["bệnh thận mạn"])
    E("Chup_Can_Quang", "Condition", "Chụp cản quang có iod", ["cản quang", "chụp ct"])
    E("Nhiem_Toan_Lactic", "Complication", "Nhiễm toan lactic", ["nhiễm toan lactic", "toan lactic", "lactic"], severity="nguy kịch")
    E("Ha_Duong_Huyet", "Complication", "Hạ glucose huyết (< 70 mg/dL)", ["hạ đường huyết", "hạ glucose"], threshold=70)
    E("Toan_Ceton", "Complication", "Nhiễm toan ceton", ["toan ceton", "ceton"])
    E("Benh_Tim_Mach", "Complication", "Bệnh tim mạch do xơ vữa", ["tim mạch", "xơ vữa", "đột quỵ"])
    E("Benh_Vong_Mac", "Complication", "Bệnh võng mạc", ["võng mạc", "mắt", "mù"])
    E("Benh_Than_DTD", "Complication", "Bệnh thận do đái tháo đường", ["albumin niệu"])
    E("Loet_Ban_Chan", "Complication", "Loét bàn chân", ["bàn chân", "loét"])
    E("Thieu_B12", "Complication", "Thiếu vitamin B12", ["b12"])
    # thuốc
    E("Metformin", "Biguanide", "Metformin", ["metformin", "glucophage"], first_line=True)
    E("Gliclazide", "Sulfonylurea", "Gliclazide", ["gliclazide", "diamicron", "sulfonylurea"])
    E("Sitagliptin", "DPP4Inhibitor", "Sitagliptin", ["sitagliptin", "januvia", "dpp-4"])
    E("Empagliflozin", "SGLT2Inhibitor", "Empagliflozin", ["empagliflozin", "dapagliflozin", "sglt2"])
    E("Semaglutide", "GLP1Agonist", "Semaglutide", ["semaglutide", "liraglutide", "glp-1"])
    E("Insulin_Nen", "Insulin", "Insulin", ["insulin"])
    # chất, thực phẩm
    E("Ruou_Bia", "RiskSubstance", "Rượu bia (ethanol)", ["rượu", "bia", "rượu bia", "cồn", "nhậu", "say"], kcal_per_g=7)
    E("Thit_Ga", "FoodGroup", "Thịt gà", ["thịt gà", "gà"])
    E("Uc_Ga_Bo_Da", "SafeFood", "Ức gà bỏ da luộc/hấp", ["ức gà", "gà luộc", "gà hấp"], protein_g=31, carb_g=0)
    E("Ga_Ran", "RiskFood", "Gà rán, gà chiên, da gà", ["gà rán", "gà chiên", "da gà"], saturated_fat="cao")
    E("Tinh_Bot", "FoodGroup", "Thực phẩm giàu tinh bột", ["tinh bột", "carbohydrate"])
    E("Com_Trang", "ModerateFood", "Cơm trắng", ["cơm", "cơm trắng", "gạo trắng"], gi=73, gl=29)
    E("Gao_Lut", "ModerateFood", "Gạo lứt", ["gạo lứt"], gi=68)
    E("Bun_Pho", "ModerateFood", "Bún, phở", ["bún", "phở"], gi=53)
    E("Khoai_Tay", "RiskFood", "Khoai tây luộc", ["khoai tây"], gi=78)
    E("Xoi_Nep", "RiskFood", "Xôi nếp", ["xôi", "nếp"], gi="rất cao")
    E("Rau_Xanh", "SafeFood", "Rau xanh (rau muống, cải, bông cải)", ["rau", "rau xanh", "rau muống"], gi="rất thấp")
    E("Trai_Cay_GI_Thap", "SafeFood", "Táo, cam (GI thấp)", ["táo", "cam", "trái cây"], gi=36)
    # lối sống
    E("Van_Dong_150", "Lifestyle", "Vận động ≥ 150 phút/tuần", ["vận động", "tập thể dục", "đi bộ"], minutes=150)
    E("Giam_Can", "Lifestyle", "Giảm 5 - 7% cân nặng", ["giảm cân"], pct=7)
    E("Bo_Thuoc_La", "Lifestyle", "Bỏ thuốc lá", ["bỏ thuốc"])
    E("Quy_Tac_15_15", "Lifestyle", "Quy tắc 15-15 (15 g carbohydrate, chờ 15 phút)", ["15-15", "15 15"])

    F = kg.add_fact
    F("HbA1c_Cao", "TIEU_CHUAN_CHAN_DOAN", "DTD_Tip2", source="h01-chan-doan")
    F("Glucose_Doi_Cao", "TIEU_CHUAN_CHAN_DOAN", "DTD_Tip2", source="h01-chan-doan")
    F("OGTT_2h_Cao", "TIEU_CHUAN_CHAN_DOAN", "DTD_Tip2", source="h01-chan-doan")
    F("HbA1c_Tien", "TIEU_CHUAN_CHAN_DOAN", "Tien_DTD", source="h01-chan-doan")
    F("Tien_DTD", "TIEN_TRIEN_THANH", "DTD_Tip2", source="h08-tien-dai-thao-duong", rate="5-10%/năm")
    F("DTD_Tip2", "THEO_HUONG_DAN", "QD_5481", source="h01-chan-doan")
    for rf in ("Beo_Phi", "It_Van_Dong", "Hut_Thuoc", "Tang_Huyet_Ap"):
        F(rf, "YEU_TO_NGUY_CO", "DTD_Tip2", source="h08-tien-dai-thao-duong")
    F("DTD_Tip2", "CHI_DINH_DAU_TAY", "Metformin", source="h02-thuoc-dieu-tri", dose="500 mg x 1-2 lần/ngày")
    F("DTD_Tip1", "CHI_DINH_DAU_TAY", "Insulin_Nen", source="h02-thuoc-dieu-tri", note="bắt buộc suốt đời")
    for d in ("Gliclazide", "Sitagliptin", "Empagliflozin", "Semaglutide", "Insulin_Nen"):
        F(d, "DIEU_TRI", "DTD_Tip2", source="h02-thuoc-dieu-tri")
    F("Metformin", "CHONG_CHI_DINH_KHI", "Suy_Than_Nang", source="h02-thuoc-dieu-tri", rule="eGFR < 30 mL/phút/1,73 m2")
    F("Metformin", "CHONG_CHI_DINH_KHI", "Chup_Can_Quang", source="h02-thuoc-dieu-tri", rule="tạm ngưng trước chụp")
    F("Empagliflozin", "UU_TIEN_KHI", "Suy_Tim", source="h02-thuoc-dieu-tri")
    F("Empagliflozin", "UU_TIEN_KHI", "Benh_Than_Man", source="h02-thuoc-dieu-tri")
    F("Metformin", "TUONG_TAC_NGUY_HIEM", "Ruou_Bia", source="h03-ruou-bia", severity="cao",
      effect="tăng nguy cơ nhiễm toan lactic; tránh uống quá mức và say rượu")
    F("Gliclazide", "TUONG_TAC_NGUY_HIEM", "Ruou_Bia", source="h03-ruou-bia", severity="cao",
      effect="tăng mạnh nguy cơ hạ glucose huyết")
    F("Insulin_Nen", "TUONG_TAC_NGUY_HIEM", "Ruou_Bia", source="h03-ruou-bia", severity="cao",
      effect="tăng mạnh nguy cơ hạ glucose huyết")
    F("Metformin", "TANG_NGUY_CO", "Nhiem_Toan_Lactic", source="h02-thuoc-dieu-tri", freq="hiếm")
    F("Metformin", "TANG_NGUY_CO", "Thieu_B12", source="h02-thuoc-dieu-tri", freq="dùng lâu dài")
    F("Ruou_Bia", "TANG_NGUY_CO", "Nhiem_Toan_Lactic", source="h03-ruou-bia")
    F("Ruou_Bia", "TANG_NGUY_CO", "Ha_Duong_Huyet", source="h03-ruou-bia", window="tới 24 giờ")
    F("Gliclazide", "TANG_NGUY_CO", "Ha_Duong_Huyet", source="h02-thuoc-dieu-tri")
    F("Insulin_Nen", "TANG_NGUY_CO", "Ha_Duong_Huyet", source="h02-thuoc-dieu-tri")
    F("Ga_Ran", "TANG_NGUY_CO", "Benh_Tim_Mach", source="h04-dinh-duong", reason="chất béo bão hòa, tăng LDL")
    for c in ("Benh_Tim_Mach", "Benh_Vong_Mac", "Benh_Than_DTD", "Loet_Ban_Chan", "Ha_Duong_Huyet", "Toan_Ceton"):
        F(c, "BIEN_CHUNG_CUA", "DTD_Tip2", source="h06-bien-chung" if c not in ("Ha_Duong_Huyet", "Toan_Ceton")
          else "h07-ha-duong-huyet")
    F("Ha_Duong_Huyet", "XU_TRI_BANG", "Quy_Tac_15_15", source="h07-ha-duong-huyet")
    F("Uc_Ga_Bo_Da", "LA_MOT_DANG_CUA", "Thit_Ga", source="h04-dinh-duong")
    F("Ga_Ran", "LA_MOT_DANG_CUA", "Thit_Ga", source="h04-dinh-duong")
    for f in ("Com_Trang", "Gao_Lut", "Bun_Pho", "Khoai_Tay", "Xoi_Nep"):
        F(f, "LA_MOT_DANG_CUA", "Tinh_Bot", source="h04-dinh-duong")
    F("DTD_Tip2", "KHUYEN_NGHI_AN", "Uc_Ga_Bo_Da", source="h04-dinh-duong", reason="đạm nạc 31 g/100 g, không carbohydrate")
    F("DTD_Tip2", "KHUYEN_NGHI_AN", "Rau_Xanh", source="h04-dinh-duong", reason="chất xơ làm chậm hấp thu; ít nhất nửa đĩa")
    F("DTD_Tip2", "KHUYEN_NGHI_AN", "Trai_Cay_GI_Thap", source="h04-dinh-duong", reason="GI thấp, chia nhỏ")
    F("DTD_Tip2", "HAN_CHE_AN", "Ga_Ran", source="h04-dinh-duong", reason="nhiều chất béo bão hòa, tăng đề kháng insulin")
    F("DTD_Tip2", "HAN_CHE_AN", "Khoai_Tay", source="h04-dinh-duong", reason="GI 78")
    F("DTD_Tip2", "HAN_CHE_AN", "Xoi_Nep", source="h04-dinh-duong", reason="GI rất cao")
    F("DTD_Tip2", "AN_DINH_LUONG", "Com_Trang", source="h04-dinh-duong", reason="GI 73, GL 29; tối đa ~1 bát con/bữa, ăn rau trước")
    F("DTD_Tip2", "AN_DINH_LUONG", "Gao_Lut", source="h04-dinh-duong", reason="GI 68, giàu chất xơ; thay một phần gạo trắng")
    F("DTD_Tip2", "AN_DINH_LUONG", "Bun_Pho", source="h04-dinh-duong", reason="GI 53; nước dùng trong, nhiều rau")
    F("DTD_Tip2", "KHUYEN_NGHI_LOI_SONG", "Van_Dong_150", source="h05-van-dong")
    F("DTD_Tip2", "KHUYEN_NGHI_LOI_SONG", "Giam_Can", source="h05-van-dong")
    F("DTD_Tip2", "KHUYEN_NGHI_LOI_SONG", "Bo_Thuoc_La", source="h05-van-dong")
    F("Tien_DTD", "KHUYEN_NGHI_LOI_SONG", "Van_Dong_150", source="h08-tien-dai-thao-duong", effect="giảm 58% nguy cơ (DPP)")
    F("Tien_DTD", "KHUYEN_NGHI_LOI_SONG", "Giam_Can", source="h08-tien-dai-thao-duong")
    return kg


def drug_warnings(kg, drug, substance):
    """Cảnh báo tương tác: cạnh trực tiếp drug -> substance và mọi nguy cơ mà cả hai cùng làm tăng (đường 2 bước)."""
    direct = list(kg.facts(drug, "TUONG_TAC_NGUY_HIEM", substance))
    risk_d = {o for _, _, o, _ in kg.facts(drug, "TANG_NGUY_CO")}
    risk_s = {o for _, _, o, _ in kg.facts(substance, "TANG_NGUY_CO")}
    return direct, sorted(risk_d & risk_s)


def food_advice(kg, disease, food):
    """Lời khuyên cho một món; nếu là nhóm thực phẩm (Thịt gà) thì duyệt các dạng cụ thể qua LA_MOT_DANG_CUA."""
    items = [food] + [s for s, _, _, _ in kg.facts(p="LA_MOT_DANG_CUA", o=food)]
    out = []
    for it in items:
        for rel in ("KHUYEN_NGHI_AN", "AN_DINH_LUONG", "HAN_CHE_AN"):
            out += list(kg.facts(disease, rel, it))
    return out
