"""Đồ thị tri thức tài chính: tác động lan truyền giữa biến số vĩ mô, chính sách, vàng và cổ phiếu VN30.

Cạnh nhân quả mang dấu: TAC_DONG_CUNG_CHIEU (+1) nghĩa là biến nguồn tăng thì biến đích có xu hướng tăng,
TAC_DONG_NGUOC_CHIEU (−1) thì ngược lại. Dấu của một đường đi nhiều bước là tích các dấu trên đường.
Cạnh cấu trúc (thành viên chỉ số, ngành, sửa đổi văn bản) không tham gia lan truyền.
"""
import networkx as nx

from .kg_base import KnowledgeGraph, Ontology

SIGN = {"TAC_DONG_CUNG_CHIEU": +1, "TAC_DONG_NGUOC_CHIEU": -1}


def finance_ontology():
    o = Ontology()
    for c, parent in [
        ("Thing", None), ("FinancialEntity", "Thing"),
        ("MacroFactor", "FinancialEntity"), ("PolicyTool", "MacroFactor"), ("GlobalFactor", "MacroFactor"),
        ("MarketFlow", "MacroFactor"), ("Event", "FinancialEntity"),
        ("Regulation", "FinancialEntity"), ("Asset", "FinancialEntity"), ("Commodity", "Asset"),
        ("GoldAsset", "Commodity"), ("Indicator", "FinancialEntity"),
        ("Stock", "Asset"), ("Index", "Asset"), ("Exchange", "FinancialEntity"), ("Sector", "FinancialEntity"),
    ]:
        o.add_class(c, parent)
    P = o.add_predicate
    P("TAC_DONG_CUNG_CHIEU", "FinancialEntity", "FinancialEntity", "tác động cùng chiều (+)")
    P("TAC_DONG_NGUOC_CHIEU", "FinancialEntity", "FinancialEntity", "tác động ngược chiều (−)")
    P("THUOC_VE", "FinancialEntity", "FinancialEntity", "thuộc về", transitive=True)
    P("THUOC_NGANH", "Stock", "Sector", "thuộc ngành")
    P("SUA_DOI", "Regulation", "Regulation", "sửa đổi, bổ sung")
    P("QUY_DINH_VE", "Regulation", "Asset", "quy định về")
    return o


def build_finance_kg():
    kg = KnowledgeGraph("VietnamFinanceKG", finance_ontology())
    E = kg.add_entity
    E("Lai_Suat_Fed", "GlobalFactor", "Lãi suất Fed", ["fed", "lãi suất mỹ", "lãi suất fed"])
    E("DXY", "GlobalFactor", "Chỉ số USD (DXY)", ["dxy", "đồng usd mạnh", "đô la mỹ", "đồng đô"], series="DXY")
    E("Bat_On_Dia_Chinh_Tri", "GlobalFactor", "Bất ổn địa chính trị", ["địa chính trị", "chiến tranh", "bất ổn"])
    E("NHTW_Mua_Vang", "GlobalFactor", "Ngân hàng trung ương mua vàng", ["ngân hàng trung ương mua vàng"])
    E("Dong_Yen", "GlobalFactor", "Sức mạnh đồng yên (JPY)", ["yên", "jpy", "đồng yên"])
    E("Gia_Quang_Sat", "GlobalFactor", "Giá quặng sắt, than cốc", ["quặng sắt", "than cốc", "than mỡ"])
    E("USD_VND", "MacroFactor", "Tỷ giá USD/VND", ["tỷ giá", "usd/vnd", "usdvnd", "đô la", "usd"], series="USDVND")
    E("Lam_Phat", "MacroFactor", "Lạm phát", ["lạm phát"])
    E("Lai_Suat_NHNN", "PolicyTool", "Lãi suất điều hành NHNN", ["lãi suất", "lãi suất điều hành", "tái cấp vốn", "nhnn"],
      refinancing_rate=4.5)
    E("Tin_Phieu_NHNN", "PolicyTool", "Phát hành tín phiếu NHNN", ["tín phiếu"])
    E("Dau_Tu_Cong", "MacroFactor", "Giải ngân đầu tư công", ["đầu tư công", "hạ tầng", "cao tốc"])
    E("Khoi_Ngoai", "MarketFlow", "Khối ngoại mua ròng", ["khối ngoại", "nhà đầu tư nước ngoài", "vốn ngoại"])
    E("Nang_Hang_FTSE", "Event", "FTSE nâng hạng thị trường mới nổi thứ cấp", ["nâng hạng", "ftse"], effective="9/2026")
    E("ND_24_2012", "Regulation", "Nghị định 24/2012/NĐ-CP", ["nghị định 24", "24/2012", "độc quyền vàng"], year=2012)
    E("ND_232_2025", "Regulation", "Nghị định 232/2025/NĐ-CP", ["nghị định 232", "232/2025"], year=2025)
    E("NHNN_Ban_Vang_2024", "PolicyTool", "NHNN đấu thầu và bán vàng miếng (2024)", ["đấu thầu vàng", "bán vàng"], year=2024)
    E("XAU_USD", "GoldAsset", "Giá vàng thế giới (XAU/USD)", ["giá vàng thế giới", "xau", "vàng thế giới"], series="XAU")
    E("Vang_Quy_Doi", "GoldAsset", "Giá vàng thế giới quy đổi (VND/lượng)", ["giá quy đổi", "quy đổi"], series="GOLD_VND",
      factor=1.2057)
    E("Chenh_Lech_SJC", "Indicator", "Phần bù vàng miếng SJC so với thế giới", ["chênh lệch", "phần bù"])
    E("Vang_SJC", "GoldAsset", "Vàng miếng SJC", ["vàng sjc", "sjc", "vàng miếng", "vàng"])
    E("Vang_Nhan", "GoldAsset", "Vàng nhẫn 9999", ["vàng nhẫn", "9999"])
    E("FPT", "Stock", "FPT (Tập đoàn FPT)", ["fpt"], series="FPT")
    E("HPG", "Stock", "HPG (Hòa Phát)", ["hpg", "hòa phát", "thép"], series="HPG")
    E("VCB", "Stock", "VCB (Vietcombank)", ["vcb", "vietcombank"], series="VCB")
    E("VIC", "Stock", "VIC (Vingroup)", ["vic", "vingroup", "vinhomes", "vinfast"], series="VIC")
    E("VN30", "Index", "Chỉ số VN30", ["vn30"], series="VN30ETF")
    E("HOSE", "Exchange", "Sở GDCK TP.HCM (HOSE)", ["hose"])
    E("Nganh_CNTT", "Sector", "Công nghệ thông tin", ["công nghệ"])
    E("Nganh_Thep", "Sector", "Thép", ["ngành thép"])
    E("Nganh_Ngan_Hang", "Sector", "Ngân hàng", ["ngân hàng"])
    E("Nganh_BDS", "Sector", "Bất động sản", ["bất động sản"])

    def up(s, o, src, why):
        kg.add_fact(s, "TAC_DONG_CUNG_CHIEU", o, source=src, mechanism=why)

    def down(s, o, src, why):
        kg.add_fact(s, "TAC_DONG_NGUOC_CHIEU", o, source=src, mechanism=why)

    # kênh vàng
    down("Lai_Suat_Fed", "XAU_USD", "f02-gia-vang-the-gioi", "chi phí cơ hội nắm giữ vàng tăng")
    up("Lai_Suat_Fed", "DXY", "f02-gia-vang-the-gioi", "lãi suất USD cao làm USD mạnh lên")
    down("DXY", "XAU_USD", "f02-gia-vang-the-gioi", "vàng định giá bằng USD, biến động ngược chiều DXY")
    up("Bat_On_Dia_Chinh_Tri", "XAU_USD", "f02-gia-vang-the-gioi", "nhu cầu tài sản trú ẩn")
    up("NHTW_Mua_Vang", "XAU_USD", "f02-gia-vang-the-gioi", "cầu vàng của ngân hàng trung ương")
    up("XAU_USD", "Vang_Quy_Doi", "f02-gia-vang-the-gioi", "XAU/USD × USD/VND × 1,2057")
    up("USD_VND", "Vang_Quy_Doi", "f02-gia-vang-the-gioi", "XAU/USD × USD/VND × 1,2057")
    up("Vang_Quy_Doi", "Vang_SJC", "f02-gia-vang-the-gioi", "giá trong nước = quy đổi + phần bù")
    up("Vang_Quy_Doi", "Vang_Nhan", "f01-vang-mieng-sjc", "vàng nhẫn bám sát giá quy đổi")
    up("Chenh_Lech_SJC", "Vang_SJC", "f01-vang-mieng-sjc", "giá trong nước = quy đổi + phần bù")
    up("ND_24_2012", "Chenh_Lech_SJC", "f01-vang-mieng-sjc", "độc quyền, không nhập vàng nguyên liệu, cung giới hạn")
    down("NHNN_Ban_Vang_2024", "Chenh_Lech_SJC", "f01-vang-mieng-sjc", "tăng cung, chênh lệch thu hẹp")
    down("ND_232_2025", "Chenh_Lech_SJC", "f01-vang-mieng-sjc", "bãi bỏ độc quyền, tăng cung hợp pháp")
    up("Lam_Phat", "Chenh_Lech_SJC", "f01-vang-mieng-sjc", "nhu cầu tích trữ tăng khi lạm phát")
    # kênh tỷ giá và chính sách tiền tệ
    up("DXY", "USD_VND", "f03-ty-gia", "USD mạnh lên toàn cầu gây áp lực tỷ giá")
    down("Lai_Suat_NHNN", "USD_VND", "f04-chinh-sach-tien-te", "lãi suất VND thấp hơn USD gây áp lực tỷ giá")
    down("Tin_Phieu_NHNN", "USD_VND", "f04-chinh-sach-tien-te", "hút thanh khoản VND, giảm đầu cơ USD")
    down("USD_VND", "Khoi_Ngoai", "f03-ty-gia", "lợi nhuận quy đổi ra USD giảm, khối ngoại bán ròng")
    up("Nang_Hang_FTSE", "Khoi_Ngoai", "f09-vn30", "thu hút vốn ngoại")
    up("Khoi_Ngoai", "VN30", "f09-vn30", "dòng vốn vào cổ phiếu vốn hóa lớn")
    # kênh doanh nghiệp
    up("USD_VND", "FPT", "f05-fpt", "doanh thu xuất khẩu phần mềm bằng ngoại tệ, chi phí bằng VND")
    up("Dong_Yen", "FPT", "f05-fpt", "Nhật Bản là thị trường xuất khẩu lớn nhất")
    down("USD_VND", "HPG", "f06-hpg", "nhập quặng sắt, than cốc bằng USD; lỗ tỷ giá nợ vay")
    down("Gia_Quang_Sat", "HPG", "f06-hpg", "giá nguyên liệu tăng làm co biên lợi nhuận")
    up("Dau_Tu_Cong", "HPG", "f06-hpg", "tăng nhu cầu thép xây dựng")
    down("Lai_Suat_NHNN", "HPG", "f06-hpg", "lãi suất cao làm giảm nhu cầu xây dựng, bất động sản")
    down("Lai_Suat_NHNN", "VCB", "f07-vcb", "lãi suất thấp, tín dụng tăng làm lợi nhuận ngân hàng tăng")
    down("Lai_Suat_NHNN", "VIC", "f08-vic", "lãi suất vay mua nhà và chi phí tài chính")
    down("USD_VND", "VIC", "f08-vic", "gánh nặng nợ vay ngoại tệ tăng")
    # cấu trúc
    for s, sec in (("FPT", "Nganh_CNTT"), ("HPG", "Nganh_Thep"), ("VCB", "Nganh_Ngan_Hang"), ("VIC", "Nganh_BDS")):
        kg.add_fact(s, "THUOC_VE", "VN30", source="f09-vn30")
        kg.add_fact(s, "THUOC_NGANH", sec, source="f09-vn30")
    kg.add_fact("VN30", "THUOC_VE", "HOSE", source="f09-vn30")
    kg.add_fact("ND_232_2025", "SUA_DOI", "ND_24_2012", source="f01-vang-mieng-sjc")
    kg.add_fact("ND_24_2012", "QUY_DINH_VE", "Vang_SJC", source="f01-vang-mieng-sjc")
    return kg


def causal_graph(kg):
    """Đồ thị con chỉ gồm cạnh nhân quả, trọng số là dấu ±1."""
    C = nx.DiGraph()
    for s, p, o, d in kg.facts():
        if p in SIGN:
            C.add_edge(s, o, sign=SIGN[p], mechanism=d.get("mechanism"), source=d.get("source"),
                       empirical_corr=d.get("empirical_corr"))
    return C


def impact_paths(kg, factor, target, max_hops=4):
    """Liệt kê mọi đường nhân quả factor ⇝ target dài ≤ max_hops, kèm dấu tổng hợp.

    Tác động ròng được ước lượng bằng tổng có trọng số các dấu, đường ngắn nặng hơn: Σ sign(π) / |π|.
    Khi các đường mâu thuẫn dấu, tác tử phải nói rõ là "phụ thuộc kênh truyền dẫn" thay vì kết luận một chiều.
    """
    C = causal_graph(kg)
    if factor not in C or target not in C:
        return {"paths": [], "net": 0.0, "verdict": "không có đường nhân quả"}
    paths = []
    for path in nx.all_simple_paths(C, factor, target, cutoff=max_hops):
        sign, steps = 1, []
        for u, v in zip(path, path[1:]):
            e = C.edges[u, v]
            sign *= e["sign"]
            steps.append({"from": u, "to": v, "sign": e["sign"], "mechanism": e["mechanism"], "source": e["source"]})
        paths.append({"nodes": path, "sign": sign, "hops": len(path) - 1, "steps": steps})
    paths.sort(key=lambda p: p["hops"])
    net = sum(p["sign"] / p["hops"] for p in paths)
    pos, neg = sum(p["sign"] > 0 for p in paths), sum(p["sign"] < 0 for p in paths)
    verdict = ("không có đường nhân quả" if not paths else "tích cực (+)" if neg == 0 else "tiêu cực (−)" if pos == 0
               else f"hỗn hợp: {pos} kênh (+), {neg} kênh (−), ròng {'+' if net > 0 else '−'}")
    return {"paths": paths, "net": net, "verdict": verdict}


def downstream(kg, factor, max_hops=3):
    """Mọi thực thể chịu ảnh hưởng của một cú sốc, theo dấu của đường ngắn nhất (lan truyền theo chiều rộng)."""
    C = causal_graph(kg)
    out, frontier, seen = {}, [(factor, 1, [factor])], {factor}
    for _ in range(max_hops):
        nxt = []
        for u, s, path in frontier:
            for v in C.successors(u):
                if v in seen:
                    continue
                seen.add(v)
                sv = s * C.edges[u, v]["sign"]
                out[v] = {"sign": sv, "path": path + [v]}
                nxt.append((v, sv, path + [v]))
        frontier = nxt
    return out
