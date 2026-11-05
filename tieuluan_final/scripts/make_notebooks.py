"""Sinh và CHẠY notebook của từng chương (nbformat + nbconvert), để báo cáo trích dẫn ô và kết quả thật.

    python scripts/make_notebooks.py ch2 [ch3 ch4 ch5 ch6 ch7]      # ghi notebooks/chN_*.ipynb đã có output

Mỗi ô có mã định danh dạng "ML-01" ở dòng chú thích đầu, báo cáo dùng mã này khi giải thích từng ô.
"""
import sys
from pathlib import Path

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

TLF = Path(__file__).resolve().parents[1]
NB = TLF / "notebooks"
NB.mkdir(exist_ok=True)

SETUP = """# {tag}-00 · Thiết lập đường dẫn: mã chương dùng chung gói của tiểu luận (src/dm) và mã kế thừa của nhóm (src/tl01, tl03)
import os, sys, json, time, warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
warnings.filterwarnings("ignore")
TLF = Path.cwd().parent
for p in ["src", "src/tl01", "src/tl03"]:
    sys.path.insert(0, str(TLF / p))
import numpy as np, pandas as pd
np.set_printoptions(precision=4, suppress=True)
print("Thư mục tiểu luận:", TLF)"""

CHAPTERS = {
    "ch2": ("ch2_hoc_may.ipynb", "ML", "Chương 2 — Học máy cơ bản: MLP cài đặt bằng ba cách trên dữ liệu tiểu đường", [
        ("md", "Notebook chạy lại phần cốt lõi của Chương 2 trên bộ **Diabetes Prediction (100.000 hồ sơ)**: tiền xử lý không rò rỉ, "
               "cùng một MLP `Dense(128)–ReLU–Dropout–Dense(64)–ReLU–Dropout–Dense(1)` viết bằng NumPy, Keras và PyTorch, "
               "rồi đọc lại kết quả đầy đủ do `scripts/run_ch2_*.py` ghi ra."),
        ("code", """# ML-01 · Nạp dữ liệu và xem phân bố nhãn
from tl import data_tabular as dt
df = dt.diabetes_frame()
print(df.shape)
print(df["diabetes"].value_counts(normalize=True).round(4))
df.head()"""),
        ("code", """# ML-02 · Chia 80/20 phân tầng TRƯỚC, rồi mới khớp bộ chuẩn hóa trên tập huấn luyện (không rò rỉ)
d = dt.load_diabetes()
Xtr, Xte, ytr, yte = d["X_train"], d["X_test"], d["y_train"], d["y_test"]
print("train", Xtr.shape, "test", Xte.shape, "tỷ lệ dương", ytr.mean().round(4), yte.mean().round(4))
num = d["pre"].named_transformers_["num"]
print("trung bình dùng để chuẩn hóa (chỉ tính trên train):", np.round(num.mean_, 3))"""),
        ("code", """# ML-03 · MLP viết tay bằng NumPy: lan truyền tiến, ngược và Adam — huấn luyện thử 5 epoch
from tl.mlp_scratch import MLPScratch
from sklearn.metrics import roc_auc_score
m = MLPScratch(Xtr.shape[1], output_dim=1, task="binary")
t0 = time.time()
m.fit(Xtr[:60000], ytr[:60000], Xtr[60000:], ytr[60000:], epochs=5, log=lambda *a: None)
p = m.predict_raw(Xte).ravel()
print(f"NumPy: AUC = {roc_auc_score(yte, p):.4f} sau 5 epoch, {time.time()-t0:.1f}s")"""),
        ("code", """# ML-04 · Cùng kiến trúc bằng Keras và PyTorch (5 epoch) để so cách viết
import os; os.environ["TL_TF_THREADS"] = "4"
from tl.mlp_keras import build_keras_mlp, fit as kfit
from tl.mlp_torch import TorchMLP, fit as tfit, predict_raw
km = build_keras_mlp(Xtr.shape[1], task="binary")
kfit(km, Xtr[:60000], ytr[:60000], Xtr[60000:], ytr[60000:], epochs=5)
print("Keras  : AUC =", round(roc_auc_score(yte, km.predict(Xte, verbose=0).ravel()), 4))
tm = TorchMLP(Xtr.shape[1])
tfit(tm, Xtr[:60000], ytr[:60000], Xtr[60000:], ytr[60000:], "binary", epochs=5, log=lambda *a: None)
print("PyTorch: AUC =", round(roc_auc_score(yte, predict_raw(tm, Xte).ravel()), 4))
print(km.summary())"""),
        ("code", """# ML-05 · Đọc kết quả đầy đủ (40 epoch, dừng sớm) của ba framework và năm thuật toán cổ điển
M = TLF / "src/tl01/outputs/metrics"
rows = []
for fw in ["scratch", "keras", "torch"]:
    r = json.loads((M / f"ch2_mlp_{fw}.json").read_text())["diabetes"]
    rows.append({"mô hình": f"MLP – {fw}", **{k: r[k] for k in ["acc", "f1", "auc", "train_s", "infer_ms"]}})
sk = json.loads((M / "ch2_sklearn.json").read_text())          # khoá dạng "diabetes/logreg"
rows += [{"mô hình": k.split("/")[1], **{c: v[c] for c in ["acc", "f1", "auc", "train_s", "infer_ms"]}}
         for k, v in sk.items() if k.startswith("diabetes/")]
pd.DataFrame(rows).round(4)"""),
        ("code", """# ML-06 · Ngưỡng quyết định: Youden J so với 0,5 cố định (minh họa trên tập kiểm tra; thực tế chọn trên validation)
from sklearn.metrics import roc_curve
fpr, tpr, thr = roc_curve(yte, p)
j = np.argmax(tpr - fpr)
print(f"ngưỡng Youden = {thr[j]:.3f}: độ nhạy {tpr[j]:.3f}, độ đặc hiệu {1-fpr[j]:.3f}")
for t in [0.3, 0.5, 0.7]:
    pred = p >= t
    print(f"ngưỡng {t}: độ nhạy {pred[yte==1].mean():.3f}, độ đặc hiệu {(~pred)[yte==0].mean():.3f}")"""),
    ]),

    "ch3": ("ch3_cnn_day_mat.ipynb", "CNN", "Chương 3 — Mạng nơ-ron tích chập: im2col/col2im và ba cách cài đặt trên RetinaMNIST", [
        ("md", "Notebook kiểm chứng tích chập viết tay bằng `im2col`/`col2im`, đối chiếu đạo hàm giải tích với sai phân hữu hạn, "
               "và so sánh 5 mô hình CNN (NumPy, Keras, PyTorch) trên bộ ảnh đáy mắt **RetinaMNIST** cùng hai bộ ảnh đối chiếu **MNIST**, **Fashion-MNIST**."),
        ("code", """# CNN-01 · Nạp dữ liệu ảnh đáy mắt RetinaMNIST và xem phân bố 5 mức bệnh võng mạc ĐTĐ
from tl3.ch3_data import load_raw, CLASSES
Xtr, ytr, Xte, yte = load_raw("retina")
print("Train+Val:", Xtr.shape, "Test:", Xte.shape)
dist = pd.Series(yte).value_counts().sort_index()
for k, cnt in dist.items():
    print(f"  Mức {k} ({CLASSES['retina'][k]}): {cnt} ảnh ({cnt/len(yte):.1%})")"""),
        ("code", """# CNN-02 · Kiểm tra đạo hàm giải tích của conv_backward (im2col/col2im) so với sai phân hữu hạn
from tl3.ch3_scratch import conv_forward, conv_backward
rng = np.random.default_rng(42)
X = rng.standard_normal((2, 3, 6, 6))
W = rng.standard_normal((4, 3, 3, 3)) * 0.1
b = rng.standard_normal(4) * 0.1
out, cache = conv_forward(X, W, b, pad=1)
dOut = rng.standard_normal(out.shape)
dX, dW, db = conv_backward(dOut, cache)
eps = 1e-5
num_dW = np.zeros_like(W)
for idx in np.ndindex(W.shape):
    W[idx] += eps; p1, _ = conv_forward(X, W, b, pad=1)
    W[idx] -= 2*eps; p2, _ = conv_forward(X, W, b, pad=1)
    W[idx] += eps
    num_dW[idx] = np.sum((p1 - p2) * dOut) / (2 * eps)
rel_err = np.max(np.abs(dW - num_dW) / np.maximum(1e-8, np.abs(dW) + np.abs(num_dW)))
print(f"Kích thước đầu ra conv: {out.shape}, sai số tương đối cực đại của dW = {rel_err:.2e}")"""),
        ("code", """# CNN-03 · Đánh giá 5 mô hình đã huấn luyện trên 400 ảnh kiểm tra RetinaMNIST (Accuracy, Macro-F1, QWK)
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from tl3.serve_ch3 import _load, preprocess
from PIL import Image
rows = []
for mid in ["scratch", "tf_basic", "torch_basic", "tf_improved", "torch_improved"]:
    fn, sz = _load("retina", mid)
    xs = np.concatenate([preprocess("retina", Image.fromarray(im), sz) for im in Xte], axis=0)
    pred = fn(xs).argmax(axis=1)
    rows.append({"mô hình": mid, "kích thước": f"{sz}x{sz}",
                 "acc": accuracy_score(yte, pred),
                 "macro_f1": f1_score(yte, pred, average="macro"),
                 "qwk": cohen_kappa_score(yte, pred, weights="quadratic")})
pd.DataFrame(rows).round(4)"""),
        ("code", """# CNN-04 · Đối chiếu cấu trúc hàm hợp 5 khối của BasicCNN và số tham số từng lớp
from tl3.ch3_torch import BasicCNN
net = BasicCNN(in_ch=3, n_classes=5, size=48)
total = sum(p.numel() for p in net.parameters())
for name, mod in net.named_children():
    n_p = sum(p.numel() for p in mod.parameters())
    print(f"Lớp {name:4s} ({mod.__class__.__name__:10s}): {n_p:9,d} tham số ({n_p/total:6.2%})")
print(f"Tổng cộng: {total:,d} tham số")"""),
    ]),

    "ch4": ("ch4_rnn_cgm.ipynb", "RNN", "Chương 4 — Mạng nơ-ron hồi quy: BPTT và dự báo đường huyết CGM 30 phút bằng ba cách", [
        ("md", "Notebook trình bày chia dữ liệu CGM theo bệnh nhân, kiểm chứng BPTT của RNN viết tay bằng NumPy, "
               "so sánh 3 cách cài đặt (NumPy, Keras, PyTorch) cùng LSTM xếp chồng 5 lớp, và đối chiếu với 3 chuỗi tài chính/thương mại."),
        ("code", """# RNN-01 · Nạp dữ liệu Shanghai CGM (chia theo bệnh nhân 78/17/17) và tạo đặc trưng [glucose chuẩn hóa, sin(giờ), cos(giờ)]
from dm.ch4_cgm import CGMData
data = CGMData()
for split in ("train", "val", "test"):
    X, H, Y = data.raw[split]
    print(f"{split:5s}: {data.n_patients[split]:2d} bệnh nhân, {len(Y):6,d} cửa sổ 12 bước, glucose TB = {Y.mean():.1f} ± {Y.std():.1f} mg/dL")"""),
        ("code", """# RNN-02 · Kiểm chứng lan truyền ngược qua thời gian (BPTT) của RNN NumPy trên một cửa sổ CGM
from tl3.ch4_scratch import RNNScratch
Xte, Hte, Yte = data.raw["test"]
Fte = data.features(Xte[:5], Hte[:5])
target = (Yte[:5] - Xte[:5, -1]) / data.sd
rnn = RNNScratch(3, 16, seed=0)
yhat = rnn.forward(Fte[0])
grads = rnn.backward(yhat, float(target[0]))
eps = 1e-5
p0 = rnn.p["Whh"][0, 0]
rnn.p["Whh"][0, 0] = p0 + eps; l1 = 0.5 * (rnn.forward(Fte[0]) - float(target[0])) ** 2
rnn.p["Whh"][0, 0] = p0 - eps; l2 = 0.5 * (rnn.forward(Fte[0]) - float(target[0])) ** 2
rnn.p["Whh"][0, 0] = p0
num_g = (l1 - l2) / (2 * eps)
print(f"Đạo hàm BPTT dL/dWhh[0,0] = {grads['Whh'][0,0]:.6f}, sai phân hữu hạn = {num_g:.6f}, sai lệch = {abs(grads['Whh'][0,0]-num_g):.2e}")"""),
        ("code", """# RNN-03 · Tổng hợp kết quả dự báo glucose 30 phút của 3 cách cài đặt trên 18.056 cửa sổ kiểm tra
rows = []
for fw in ("numpy", "keras", "torch"):
    d = json.loads((TLF / f"outputs/metrics/ch4_cgm_{fw}.json").read_text())["models"]
    for k, v in d.items():
        rows.append({"framework": fw, "mô hình": k, "RMSE": v["rmse"], "MAE": v["mae"],
                     "MARD (%)": v["mard"], "Clarke A": v["clarke_a"], "tham số": v["params"],
                     "train (s)": v.get("train_s", 0.0)})
pd.DataFrame(rows).round(3)"""),
        ("code", """# RNN-04 · Đối chiếu trên 3 chuỗi tài chính và thương mại (AAPL, vàng GLD, doanh thu TMĐT)
dom_rows = []
M3 = TLF / "src/tl03/outputs/metrics"
for exp in ("aapl", "gold", "ecommerce"):
    for fw in ("torch", "keras"):
        d = json.loads((M3 / f"ch4_{exp}_{fw}.json").read_text())
        naive_rmse = d["naive"]["test"]["rmse"]
        for m in ("lstm", "gru"):
            t = d[m]["test"]
            dom_rows.append({"chuỗi": exp, "framework": fw, "mô hình": m,
                             "RMSE": t["rmse"], "naive RMSE": naive_rmse,
                             "chênh lệch (%)": 100 * (t["rmse"] - naive_rmse) / naive_rmse,
                             "đúng hướng": t["dir_acc"]})
pd.DataFrame(dom_rows).round(3)"""),
    ]),

    "ch5": ("ch5_kg_rag.ipynb", "RAG", "Chương 5 — Đồ thị tri thức và RAG: truy xuất lai BM25 + Dense + Graph trên y văn đái tháo đường", [
        ("md", "Notebook khảo sát cấu trúc Đồ thị tri thức (KG) tiểu đường, thử nghiệm truy xuất BM25 và liên kết thực thể đồ thị, "
               "và tổng hợp kết quả đánh giá truy xuất / hỏi đáp có căn cứ."),
        ("code", """# RAG-01 · Thống kê kho tài liệu và Đồ thị tri thức tiểu đường (96.188 tài liệu, 96.175 nút, 684.414 cạnh)
kb = json.loads((TLF / "src/tl03/outputs/metrics/ch5_kb_diabetes.json").read_text())
print(f"Số tài liệu: {kb['n_docs']:,d} (hồ sơ: {kb['n_records']:,d}, bản tổng hợp nhóm: {kb['n_summaries']})")
print(f"Số bộ ba (triples): {kb['n_triples']:,d} | Nút đồ thị: {kb['graph']['nodes']:,d} | Cạnh: {kb['graph']['edges']:,d}")
pd.Series(kb["graph"]["relations"], name="số cạnh").to_frame()"""),
        ("code", """# RAG-02 · Truy xuất BM25 trên kho MedQuAD-NIDDK (1.192 cặp hỏi–đáp của NIH) cho câu hỏi lâm sàng
from dm.ch7_medqa import KnowledgeTool
tool = KnowledgeTool()
hits = tool.search("What is the A1C threshold for diagnosing diabetes and prediabetes?", k=3)
for i, (title, snippet, score) in enumerate(hits, 1):
    print(f"[{i}] (BM25 = {score:.2f}) {title}\\n    {snippet[:180]}...")"""),
        ("code", """# RAG-03 · Công cụ truy vấn chính xác trên Đồ thị tri thức tiểu đường (Domain.facts & Domain.link)
import pickle
with open(TLF / "src/tl03/models/ch5/diabetes_domain.pkl", "rb") as f:
    D = pickle.load(f)
q = "Tỷ lệ mắc tiểu đường của bệnh nhân nữ nhóm tuổi 60–79 có tăng huyết áp là bao nhiêu?"
print("Liên kết thực thể -> mã tài liệu:", D.link(q))
ans = D.facts(q)
print("Trả lời từ KG:", ans["answer"])
print("Nguồn / công cụ:", {k: v for k, v in ans.items() if k != "answer"})"""),
        ("code", """# RAG-04 · So sánh hiệu năng các chiến lược truy xuất và xếp hạng lại trên KB tiếng Việt & MedQuAD
ret = json.loads((TLF / "src/tl02/outputs/metrics/diabetes.json").read_text()) if (TLF / "src/tl02/outputs/metrics/diabetes.json").exists() else {}
print("Thống kê KG lâm sàng (TL02):", json.loads((TLF / "src/tl02/outputs/metrics/kg.json").read_text()).get("summary", {}))"""),
    ]),

    "ch6": ("ch6_gnn_pubmed.ipynb", "GNN", "Chương 6 — Học sâu trên đồ thị: GCN, GraphSAGE, GAT tự cài đặt trên PubMed-Diabetes, Cora và CiteSeer", [
        ("md", "Notebook kiểm chứng phép chuẩn hóa ma trận kề thưa $\\hat{A} = \\tilde{D}^{-1/2}(A+I)\\tilde{D}^{-1/2}$, "
               "biểu diễn GCN dưới dạng hợp các hàm, so sánh phân loại nút bán giám sát trên ba đồ thị trích dẫn, "
               "và khảo sát hiện tượng làm trơn quá mức (over-smoothing) khi tăng số lớp."),
        ("code", """# GNN-01 · Đặc trưng cấu trúc của ba đồ thị trích dẫn: PubMed-Diabetes, Cora và CiteSeer
d_stats = json.loads((TLF / "src/tl03/outputs/metrics/ch6_data.json").read_text())
pd.DataFrame(d_stats).T.round(3)"""),
        ("code", """# GNN-02 · Kiểm chứng lớp GCN tự cài đặt bằng nhân ma trận thưa và so sánh số tham số với MLP
import torch
from tl3.ch6_gnn import load_planetoid, gcn_norm_adj, GNN
data = load_planetoid("PubMed")
adj = gcn_norm_adj(data.edge_index, data.num_nodes)
gcn = GNN("gcn", data.num_features, 64, 3)
mlp = GNN("mlp", data.num_features, 64, 3)
print("Ma trận kề chuẩn hóa Â:", adj.shape, "số phần tử khác 0:", adj._nnz())
print("Số tham số GCN:", sum(p.numel() for p in gcn.parameters()), "| MLP:", sum(p.numel() for p in mlp.parameters()))"""),
        ("code", """# GNN-03 · Kết quả phân loại nút bán giám sát (5 hạt giống) trên PubMed-Diabetes, Cora và CiteSeer
node_res = json.loads((TLF / "src/tl03/outputs/metrics/ch6_node.json").read_text())
rows = []
for ds, models in node_res.items():
    for m, v in models.items():
        rows.append({"đồ thị": ds, "mô hình": m, "test_acc": v["test_mean"], "std": v.get("test_std", 0.0),
                     "macro_f1": v.get("f1_mean", 0.0), "tham số": v["params"]})
pd.DataFrame(rows).pivot(index="mô hình", columns="đồ thị", values=["test_acc", "std"]).round(4)"""),
        ("code", """# GNN-04 · Khảo sát độ sâu (1–8 lớp) và hiện tượng làm trơn quá mức (MAD) khi có/không có kết nối tắt
depth_res = json.loads((TLF / "src/tl03/outputs/metrics/ch6_depth.json").read_text())["rows"]
pd.DataFrame(depth_res).round(4)"""),
    ]),

    "ch7": ("ch7_mas_tieu_duong.ipynb", "MAS", "Chương 7 — Hệ đa tác tử: học tăng cường (Mèo đuổi chuột) và tác tử LLM tranh luận (MedMCQA)", [
        ("md", "Notebook tổng hợp hai bài thực nghiệm của Chương 7: (1) hệ đa tác tử học tăng cường trên lưới 8×8 "
               "(một mèo, hai mèo hợp tác IQL, mèo–chuột học đối kháng) và (2) hệ đa tác tử LLM trả lời 150 câu hỏi trắc nghiệm đái tháo đường MedMCQA."),
        ("code", """# MAS-01 · Kết quả ba thí nghiệm học tăng cường Mèo đuổi chuột (đơn tác tử, hợp tác 2 mèo IQL, đối kháng)
cm = json.loads((TLF / "outputs/metrics/ch7_catmouse.json").read_text())
print("Thí nghiệm A (1 mèo vs chuột luật):")
display(pd.DataFrame(cm["A"]["eval"]).round(3))
print("Thí nghiệm B (2 mèo hợp tác):", {k: cm["B"][k] for k in ("n_states", "rule_rule", "iql")})
print("Thí nghiệm C (Ma trận đối kháng 3 mèo × 2 chuột):")
pd.DataFrame(cm["C"]["table"]).round(3)"""),
        ("code", """# MAS-02 · Kết quả 4 hệ tác tử LLM trên 150 câu hỏi trắc nghiệm đái tháo đường (MedMCQA + MedQuAD)
med = json.loads((TLF / "outputs/metrics/ch7_medqa.json").read_text())["systems"]
pd.DataFrame(med).T[["n", "accuracy", "calls", "tokens_out", "seconds", "errors"]].round(3)"""),
        ("code", """# MAS-03 · Phân tích cổng đồng thuận (Verifier) và cận trên Oracle của hệ đa tác tử trên 150 câu MedMCQA
from dm.ch7_medqa import to_letter
rows = {s: {r["id"]: r for r in map(json.loads, (TLF / f"outputs/ch7/rows_{s}.jsonl").read_text().splitlines())}
        for s in ("direct", "cot", "rag", "multi")}
agree, dis, oracle = [], [], 0
for i in range(150):
    m = rows["multi"][i]
    disagreed = any(s["role"] == "Coordinator" for s in m["steps"])
    (dis if disagreed else agree).append(m["correct"])
    picks = [rows["cot"][i]["pred"], rows["rag"][i]["pred"]] + [to_letter(s["text"]) for s in m["steps"] if s["role"] == "Critic"]
    oracle += (m["gold"] in picks)
print(f"Đồng thuận vòng 1: {len(agree)}/150 câu ({len(agree)/150:.1%}) -> Accuracy = {np.mean(agree):.3f}")
print(f"Bất đồng (cần Critic + Vote): {len(dis)}/150 câu ({len(dis)/150:.1%}) -> Accuracy = {np.mean(dis):.3f}")
print(f"Cận trên Oracle (ít nhất 1 trong CoT/RAG/Critic đúng): {oracle}/150 ({oracle/150:.3f})")"""),
        ("code", """# MAS-04 · Xem vết tranh luận thật ở câu #18 (RAG cứu CoT) và câu #101 (CoT cứu RAG)
from dm.ch7_medqa import load_questions
qs = {q["id"]: q for q in load_questions(150)}
for qid in (18, 101):
    q, m = qs[qid], rows["multi"][qid]
    print(f"=== Câu #{qid} ({q['subject']}) | Đáp án đúng: {q['gold']} | Dự đoán Multi: {m['pred']} ===")
    print("Câu hỏi:", q["question"])
    print("Phương án:", {chr(65+k): v for k, v in enumerate(q["options"])})
    for s in m["steps"]:
        print(f"  [{s['role']:12s}]: {s['text'][:160].replace(chr(10), ' ')}")"""),
    ]),
}


def build(ch):
    fname, tag, title, cells = CHAPTERS[ch]
    nb = nbformat.v4.new_notebook()
    nb.cells.append(nbformat.v4.new_markdown_cell(f"# {title}"))
    nb.cells.append(nbformat.v4.new_code_cell(SETUP.format(tag=tag)))
    for kind, src in cells:
        nb.cells.append(nbformat.v4.new_markdown_cell(src) if kind == "md" else nbformat.v4.new_code_cell(src))
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    try:
        ep.preprocess(nb, {"metadata": {"path": str(NB)}})
    except Exception:
        ep = ExecutePreprocessor(timeout=1800, kernel_name="tieuluan-final")
        ep.preprocess(nb, {"metadata": {"path": str(NB)}})
    nbformat.write(nb, NB / fname)
    print("đã chạy và ghi", NB / fname, flush=True)


if __name__ == "__main__":
    for ch in sys.argv[1:] or list(CHAPTERS):
        build(ch)
