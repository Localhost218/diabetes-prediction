"""GraphRAG thu nhỏ: hợp nhất KG, phát hiện cộng đồng (Louvain), tóm tắt cộng đồng, tìm kiếm cục bộ và toàn cục.

Local Search: liên kết thực thể trong câu hỏi -> lân cận k bước -> bộ ba kèm nguồn -> đoạn văn của tài liệu nguồn.
Global Search: xếp hạng bản tóm tắt cộng đồng theo độ tương đồng với câu hỏi -> tổng hợp nhiều cộng đồng.
"""
import re
import unicodedata

import networkx as nx


def strip_accents(s):
    s = unicodedata.normalize("NFD", s.lower())
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn").replace("đ", "d")


class EntityLinker:
    """Khớp bí danh dài nhất trước (khớp nguyên từ, không dấu) để "gà rán" không bị hiểu thành "gà"."""

    def __init__(self, kgs):
        self.table = []
        for kg in kgs:
            for n, d in kg.G.nodes(data=True):
                for a in set(d.get("aliases", [])) | {d["label"].lower()}:
                    if len(a) >= 2:
                        self.table.append((strip_accents(a), n, kg.name))
        self.table.sort(key=lambda t: -len(t[0]))

    def link(self, text):
        t = " " + re.sub(r"[^\w/%.-]+", " ", strip_accents(text)) + " "
        found, used = [], []
        for alias, node, kg in self.table:
            for m in re.finditer(r"(?<=\s)" + re.escape(alias) + r"(?=\s)", t):
                span = (m.start(), m.end())
                if any(a < span[1] and span[0] < b for a, b in used):
                    continue
                used.append(span)
                if node not in [f[0] for f in found]:
                    found.append((node, kg, alias))
        return found


def merged_graph(kgs):
    """Đồ thị vô hướng có trọng số cho phát hiện cộng đồng; cạnh nhiều quan hệ được cộng trọng số."""
    U = nx.Graph()
    for kg in kgs:
        for n, d in kg.G.nodes(data=True):
            U.add_node(n, label=d["label"], type=d["type"], kg=kg.name)
        for s, p, o, _ in kg.facts():
            w = U.edges[s, o]["weight"] + 1 if U.has_edge(s, o) else 1
            U.add_edge(s, o, weight=w)
    return U


def communities(U, resolution=1.0, seed=42):
    comms = nx.community.louvain_communities(U, weight="weight", resolution=resolution, seed=seed)
    comms = sorted(comms, key=len, reverse=True)
    return comms, nx.community.modularity(U, comms, weight="weight")


def community_report(kgs, members):
    """Bộ ba nằm trọn trong cộng đồng, đã diễn đạt thành câu, làm nguyên liệu cho bản tóm tắt."""
    facts, sources = [], set()
    for kg in kgs:
        for f in kg.facts():
            if f[0] in members and f[2] in members:
                facts.append(kg.verbalize(f))
                if f[3].get("source"):
                    sources.add(f[3]["source"])
    return facts, sorted(sources)


def summarize_communities(kgs, comms, llm=None, max_facts=18):
    """Tóm tắt mỗi cộng đồng bằng LLM cục bộ; nếu không truyền LLM thì dùng bản tóm tắt trích xuất."""
    label = {n: d["label"] for kg in kgs for n, d in kg.G.nodes(data=True)}
    out = []
    for i, members in enumerate(comms):
        facts, sources = community_report(kgs, members)
        hub = sorted(members, key=lambda n: -sum(1 for kg in kgs if n in kg.G for _ in kg.G.edges(n)))
        title = ", ".join(label[n] for n in hub[:3])
        extractive = f"Cụm gồm {len(members)} thực thể, trung tâm là {title}. " + " ".join(f + "." for f in facts[:6])
        summary, tokens = extractive, 0
        if llm is not None and facts:
            bullets = "\n".join(f"- {f}" for f in facts[:max_facts])
            prompt = ("Viết một đoạn tóm tắt 3-4 câu bằng tiếng Việt cho cụm tri thức sau, nêu chủ đề chung và các quan "
                      "hệ quan trọng nhất, không thêm thông tin ngoài danh sách.\n" + bullets)
            summary, tokens, _ = llm.chat(prompt, system="Bạn là chuyên gia tóm tắt đồ thị tri thức.", max_new_tokens=220)
        out.append({"id": i, "size": len(members), "members": sorted(members), "title": title, "facts": facts,
                    "sources": sources, "summary": summary, "extractive": extractive, "tokens": tokens})
    return out


class GraphRAG:
    def __init__(self, kgs, chunks, embedder=None, reports=None):
        self.kgs = {kg.name: kg for kg in kgs}
        self.linker = EntityLinker(kgs)
        self.chunks_by_doc = {}
        for c in chunks:
            self.chunks_by_doc.setdefault(c.doc_id, []).append(c)
        self.emb, self.reports = embedder, reports or []
        if embedder is not None and self.reports:
            self.R = embedder.encode([r["summary"] for r in self.reports])

    def local_search(self, question, hops=1, max_facts=24):
        """Trả (thực thể liên kết, bộ ba đã diễn đạt kèm nguồn, tài liệu nguồn)."""
        ents = self.linker.link(question)
        facts, docs, seen = [], [], set()
        for node, kgname, _ in ents:
            kg = self.kgs[kgname]
            hood = kg.neighbors(node, hops)
            for f in kg.facts():
                if (f[0] == node or f[2] == node or (f[0] in hood and f[2] in hood and hops > 1)) and \
                        (f[0], f[1], f[2]) not in seen:
                    seen.add((f[0], f[1], f[2]))
                    facts.append((kg.verbalize(f), f[3].get("source")))
                    if f[3].get("source") and f[3]["source"] not in docs:
                        docs.append(f[3]["source"])
        return ents, facts[:max_facts], docs

    def local_context(self, question, retriever=None, k_chunks=4, hops=1):
        """Ngữ cảnh lai: bộ ba KG + đoạn văn tốt nhất trong các tài liệu nguồn của bộ ba (hoặc toàn kho)."""
        ents, facts, docs = self.local_search(question, hops)
        passages = []
        if retriever is not None:
            pool = retriever.search(question, 40)
            ranked = [c for c, _ in pool if c.doc_id in docs] or [c for c, _ in pool]
            passages = ranked[:k_chunks]
        return {"entities": ents, "facts": facts, "docs": docs, "passages": passages}

    def global_search(self, question, top=3):
        q = self.emb.encode([question], kind="query")[0]
        s = self.R @ q
        order = s.argsort()[::-1][:top]
        return [(self.reports[i], float(s[i])) for i in order]
