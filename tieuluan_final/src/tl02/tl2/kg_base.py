"""Lõi đồ thị tri thức dùng chung: đa đồ thị có hướng gán nhãn + bản thể học + xuất RDF + suy luận.

Một KG ở đây có hai lớp:
  * lớp lược đồ (ontology): lớp khái niệm, quan hệ subClassOf, vị từ kèm miền (domain) và miền giá trị (range),
    tính chất của vị từ (đối xứng, bắc cầu, nghịch đảo);
  * lớp dữ liệu (instances): thực thể có kiểu, nhãn tiếng Việt, bí danh; cạnh có thuộc tính và nguồn trích dẫn.
"""
from dataclasses import dataclass, field

import networkx as nx
from rdflib import RDF, RDFS, Graph, Literal, Namespace, OWL, URIRef
from rdflib.namespace import XSD

EX = Namespace("http://tl02.ptit.local/kg#")


@dataclass
class Predicate:
    name: str
    domain: str
    range: str
    label: str
    symmetric: bool = False
    transitive: bool = False
    inverse: str = None


@dataclass
class Ontology:
    classes: dict = field(default_factory=dict)        # lớp -> lớp cha (None nếu là gốc)
    predicates: dict = field(default_factory=dict)     # tên -> Predicate

    def add_class(self, name, parent=None):
        self.classes[name] = parent

    def add_predicate(self, *args, **kw):
        p = Predicate(*args, **kw)
        self.predicates[p.name] = p

    def ancestors(self, cls):
        """Bao đóng bắc cầu của subClassOf: SafeFood -> Food -> MedicalEntity -> Thing."""
        out = []
        while cls is not None:
            out.append(cls)
            cls = self.classes.get(cls)
        return out

    def is_a(self, cls, target):
        return target in self.ancestors(cls)


class KnowledgeGraph:
    def __init__(self, name, ontology):
        self.name, self.onto = name, ontology
        self.G = nx.MultiDiGraph()

    # ------------------------------------------------------------ xây dựng
    def add_entity(self, eid, etype, label, aliases=(), **attrs):
        if etype not in self.onto.classes:
            raise ValueError(f"Lớp '{etype}' không có trong bản thể học")
        self.G.add_node(eid, type=etype, label=label, aliases=list(aliases), **attrs)

    def add_fact(self, s, p, o, source=None, **attrs):
        """Thêm bộ ba (s, p, o) sau khi kiểm tra ràng buộc domain/range của vị từ."""
        self.validate(s, p, o)
        self.G.add_edge(s, o, key=p, relation=p, source=source, **attrs)

    def validate(self, s, p, o):
        pred = self.onto.predicates.get(p)
        if pred is None:
            raise ValueError(f"Vị từ '{p}' chưa khai báo")
        for node, want, role in ((s, pred.domain, "chủ thể"), (o, pred.range, "đối tượng")):
            if node not in self.G:
                raise ValueError(f"Thực thể '{node}' chưa tồn tại")
            have = self.G.nodes[node]["type"]
            if not self.onto.is_a(have, want):
                raise ValueError(f"Vi phạm ràng buộc: {role} của {p} phải thuộc lớp {want}, '{node}' là {have}")

    # ------------------------------------------------------------ truy vấn
    def facts(self, s=None, p=None, o=None):
        """Khớp mẫu bộ ba, None là biến tự do: facts(p='TUONG_TAC_NGUY_HIEM') ~ SELECT ?s ?o WHERE {?s p ?o}."""
        for u, v, k, d in self.G.edges(keys=True, data=True):
            if (s is None or u == s) and (p is None or k == p) and (o is None or v == o):
                yield u, k, v, d

    def neighbors(self, eid, hops=1):
        """Tập đỉnh trong bán kính k bước (bỏ qua chiều cạnh) — nền của Local Search."""
        U = self.G.to_undirected(as_view=True)
        return set(nx.single_source_shortest_path_length(U, eid, cutoff=hops))

    def subgraph_facts(self, nodes):
        return [f for f in self.facts() if f[0] in nodes and f[2] in nodes]

    def label(self, eid):
        return self.G.nodes[eid].get("label", eid)

    def verbalize(self, fact):
        """Chuyển bộ ba thành câu tiếng Việt để đưa vào ngữ cảnh của LLM."""
        s, p, o, d = fact
        pl = self.onto.predicates[p].label
        extra = "; ".join(f"{k}: {v}" for k, v in d.items() if k not in ("relation", "source") and v is not None)
        return f"{self.label(s)} —[{pl}]→ {self.label(o)}" + (f" ({extra})" if extra else "")

    def stats(self):
        types = {}
        for _, d in self.G.nodes(data=True):
            types[d["type"]] = types.get(d["type"], 0) + 1
        rels = {}
        for _, _, k in self.G.edges(keys=True):
            rels[k] = rels.get(k, 0) + 1
        U = nx.Graph(self.G.to_undirected())
        return {"nodes": self.G.number_of_nodes(), "edges": self.G.number_of_edges(), "node_types": types,
                "relations": rels, "classes": len(self.onto.classes), "predicates": len(self.onto.predicates),
                "density": nx.density(U), "components": nx.number_connected_components(U),
                "avg_degree": 2 * U.number_of_edges() / max(1, U.number_of_nodes()),
                "with_source": sum(1 for *_, d in self.facts() if d.get("source"))}

    # ------------------------------------------------------------ RDF
    def to_rdf(self):
        """Ánh xạ sang RDF: lớp -> rdfs:Class, vị từ -> rdf:Property kèm rdfs:domain/range, cạnh -> bộ ba.

        Thuộc tính của cạnh (LPG) không có chỗ trong một bộ ba RDF, nên được giữ bằng reification
        dạng rdf:Statement — đúng điểm khác biệt giữa mô hình LPG và RDF.
        """
        g = Graph()
        g.bind("ex", EX)
        for c, parent in self.onto.classes.items():
            g.add((EX[c], RDF.type, RDFS.Class))
            if parent:
                g.add((EX[c], RDFS.subClassOf, EX[parent]))
        for p in self.onto.predicates.values():
            g.add((EX[p.name], RDF.type, RDF.Property))
            g.add((EX[p.name], RDFS.domain, EX[p.domain]))
            g.add((EX[p.name], RDFS.range, EX[p.range]))
            g.add((EX[p.name], RDFS.label, Literal(p.label, lang="vi")))
            if p.symmetric:
                g.add((EX[p.name], RDF.type, OWL.SymmetricProperty))
            if p.transitive:
                g.add((EX[p.name], RDF.type, OWL.TransitiveProperty))
            if p.inverse:
                g.add((EX[p.name], OWL.inverseOf, EX[p.inverse]))
        for n, d in self.G.nodes(data=True):
            g.add((EX[n], RDF.type, EX[d["type"]]))
            g.add((EX[n], RDFS.label, Literal(d["label"], lang="vi")))
            for k, v in d.items():
                if k in ("type", "label", "aliases") or v is None:
                    continue
                g.add((EX[n], EX[k], _lit(v)))
        for i, (s, p, o, d) in enumerate(self.facts()):
            g.add((EX[s], EX[p], EX[o]))
            extra = {k: v for k, v in d.items() if k != "relation" and v is not None}
            if extra:
                st = EX[f"stmt{i}"]
                g.add((st, RDF.type, RDF.Statement))
                g.add((st, RDF.subject, EX[s]))
                g.add((st, RDF.predicate, EX[p]))
                g.add((st, RDF.object, EX[o]))
                for k, v in extra.items():
                    g.add((st, EX[k], _lit(v)))
        return g


def _lit(v):
    if isinstance(v, bool):
        return Literal(v, datatype=XSD.boolean)
    if isinstance(v, (int, float)):
        return Literal(v)
    return Literal(str(v), lang="vi")


def rdfs_closure(g):
    """Suy luận tiến (forward chaining) các luật RDFS/OWL cần thiết cho đến khi không sinh thêm bộ ba mới.

    Luật: (1) subClassOf bắc cầu; (2) kiểu kế thừa: x a C, C ⊑ D ⇒ x a D; (3) vị từ đối xứng;
    (4) vị từ bắc cầu; (5) nghịch đảo: p inverseOf q, (x p y) ⇒ (y q x). Trả về đồ thị mới và số bộ ba suy ra.
    """
    g2 = Graph()
    for t in g:
        g2.add(t)
    sym = set(g2.subjects(RDF.type, OWL.SymmetricProperty))
    trans = set(g2.subjects(RDF.type, OWL.TransitiveProperty))
    inv = {(p, q) for p, q in g2.subject_objects(OWL.inverseOf)}
    base = len(g2)
    while True:
        new = set()
        sub = list(g2.subject_objects(RDFS.subClassOf))
        for a, b in sub:
            for c in g2.objects(b, RDFS.subClassOf):
                new.add((a, RDFS.subClassOf, c))
        for x, c in list(g2.subject_objects(RDF.type)):
            for d in g2.objects(c, RDFS.subClassOf):
                new.add((x, RDF.type, d))
        for p in sym:
            for x, y in g2.subject_objects(p):
                new.add((y, p, x))
        for p in trans:
            for x, y in g2.subject_objects(p):
                for z in g2.objects(y, p):
                    new.add((x, p, z))
        for p, q in inv:
            for x, y in g2.subject_objects(p):
                new.add((y, q, x))
            for x, y in g2.subject_objects(q):
                new.add((y, p, x))
        new = {t for t in new if t not in g2}
        if not new:
            return g2, len(g2) - base
        for t in new:
            g2.add(t)


def short(term):
    """Rút gọn URI để in kết quả SPARQL: ex:Metformin -> Metformin."""
    s = str(term)
    return s.split("#")[-1] if isinstance(term, URIRef) else s
