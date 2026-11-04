"""
Chương 7 — Đánh giá hệ đa tác tử (Orchestrator + 8 tác tử, MessageBus, hội đồng tranh biện).

Ca kiểm thử lấy từ tập KIỂM TRA của NHANES (cùng phép chia 70/10/20, random_state=42 với scripts/train_model.py,
nên PredictionAgent chưa từng thấy các ca này), chỉ giữ người có đủ glucose lúc đói và HbA1c.

  K1 — đủ xét nghiệm (FPG + HbA1c + yếu tố nguy cơ, nhập qua form): hội đồng có phân loại đường huyết đúng ngưỡng
       ADA và áp dụng đúng quy tắc "2 xét nghiệm bất thường" không?
  K2 — chỉ có FPG: so sánh tác tử Tri thức đơn lẻ (ngưỡng), tác tử Dự đoán đơn lẻ (ML) và cả hội đồng, với nhãn
       ĐTĐ thật của NHANES (gồm cả người đã được chẩn đoán và người có HbA1c ≥ 6,5% mà FPG bình thường).
  K3 — tin nhắn tiếng Việt tự do sinh từ mẫu câu: trích xuất bằng regex (không LLM) và bằng Gemini.
Cùng với thống kê vận hành: số thông điệp, vòng tranh biện, guardrail, độ trễ từng tác tử, và số câu hỏi làm rõ
(ClarificationAgent) mà hệ thống đặt ra trên các ca nằm trong phạm vi sàng lọc.

    .venv/bin/python lab/ch7_agents.py [--n 400] [--llm-n 40]
"""

import argparse
import asyncio
import os
import random
import time
from collections import Counter, defaultdict

os.environ.setdefault("LLM_ENABLED", "true")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from common import ACCENT, GREY, PALETTE, ROOT, plt, r4, save_fig, save_results  # noqa: E402
from src.agents.orchestrator import OrchestratorAgent  # noqa: E402
from src.config import settings  # noqa: E402
from src.llm.client import llm  # noqa: E402
from src.models.datasets import NHANES_FEATURES  # noqa: E402
from src.schemas.patient_profile import PatientProfile  # noqa: E402

BOOL = ["high_bp", "high_chol", "family_history", "smoker", "phys_activity", "heart_disease", "stroke"]
TIERS = ["low", "moderate", "high", "very_high"]


def test_cases(n, seed=0):
    df = pd.read_csv(ROOT / "data" / "nhanes" / "nhanes_diabetes.csv")
    X, y = df[NHANES_FEATURES], df["diabetes"]
    _, te = train_test_split(df.index, test_size=0.2, stratify=y, random_state=42)
    t = df.loc[te].dropna(subset=["fasting_glucose", "hba1c", "bmi", "age_years"])
    return t.sample(min(n, len(t)), random_state=seed).reset_index(drop=True)


def ada_status(fpg, a1c):
    if fpg >= 126 or a1c >= 6.5:
        return "diabetes_range"
    if fpg >= 100 or a1c >= 5.7:
        return "prediabetes_range"
    return "normal"


def profile_from(row, with_a1c=True):
    p = PatientProfile()
    for f in ["age_years", "sex", "bmi", "waist_cm", "systolic_bp", "diastolic_bp"]:
        if not pd.isna(row[f]):
            p.set(f, row[f], "form")
    for f in BOOL:
        if not pd.isna(row[f]):
            p.set(f, bool(row[f]), "form")
    p.set("glucose_fasting_mgdl", row["fasting_glucose"], "form")
    p.set("pregnant", False, "form")  # bộ NHANES đã loại thai phụ (scripts/fetch_nhanes.py)
    if with_a1c:
        p.set("hba1c_percent", row["hba1c"], "form")
    p.derive()
    return p


# ------------------------------------------------------------------ tin nhắn tiếng Việt
def vi_num(x, nd=1, comma=True):
    s = f"{x:.{nd}f}"
    return s.replace(".", ",") if comma else s


def message_from(row, rng):
    """Sinh tin nhắn tự do theo nhiều cách viết: BMI hoặc chiều cao–cân nặng, mmol/L hoặc mg/dL, phủ định."""
    parts = []
    age, male = int(row.age_years), row.sex == 1
    who = rng.choice(["Tôi", "Mình", "Em"])
    parts.append(rng.choice([f"{who} {age} tuổi, {'nam' if male else 'nữ'}",
                             f"{who} là {'nam' if male else 'nữ'}, năm nay {age} tuổi"]))
    truth = {"age_years": age, "sex": int(male), "bmi": round(float(row.bmi), 1)}
    if rng.random() < 0.5:
        parts.append(rng.choice([f"BMI {vi_num(row.bmi, 1, rng.random() < 0.5)}",
                                 f"chỉ số BMI là {vi_num(row.bmi, 1, rng.random() < 0.5)}"]))
    else:
        h = rng.integers(150, 178) / 100
        w = round(row.bmi * h * h)
        truth["bmi"] = round(w / h / h, 1)
        cm = int(round(h * 100))
        parts.append(rng.choice([f"cao {cm} cm, nặng {w} kg", f"nặng {w} ký cao 1m{cm - 100:02d}"]))
    if not pd.isna(row.waist_cm) and rng.random() < 0.7:
        parts.append(f"vòng eo {int(round(row.waist_cm))} cm")
        truth["waist_cm"] = int(round(row.waist_cm))
    if not pd.isna(row.systolic_bp) and rng.random() < 0.6:
        s, d = int(round(row.systolic_bp)), int(round(row.diastolic_bp))
        parts.append(f"huyết áp đo được {s}/{d}")
        truth["systolic_bp"], truth["diastolic_bp"] = s, d
    for f, yes, no in [("high_bp", "có bị tăng huyết áp", "không bị tăng huyết áp"),
                       ("family_history", "mẹ bị tiểu đường", "gia đình không ai bị tiểu đường"),
                       ("smoker", "có hút thuốc lá", "không hút thuốc")]:
        if not pd.isna(row[f]) and rng.random() < 0.7:
            parts.append(yes if row[f] == 1 else no)
            truth[f] = bool(row[f])
    fpg = float(row.fasting_glucose)
    if rng.random() < 0.5:
        parts.append(rng.choice([f"đường huyết lúc đói {vi_num(fpg / 18, 1)} mmol/L",
                                 f"sáng nay thử đường máu lúc chưa ăn được {vi_num(fpg / 18, 1)} mmol/l"]))
        truth["glucose_fasting_mgdl"] = round(round(fpg / 18, 1) * 18, 1)
    else:
        parts.append(f"glucose đói {int(round(fpg))} mg/dL")
        truth["glucose_fasting_mgdl"] = float(int(round(fpg)))
    return ", ".join(parts) + ".", truth


def field_correct(f, got, want):
    if got is None:
        return False
    if f in ("bmi",):
        return abs(got - want) <= 0.6
    if f == "glucose_fasting_mgdl":
        return abs(got - want) <= 2
    if isinstance(want, bool):
        return bool(got) == want
    return abs(float(got) - float(want)) <= 0.5


# ------------------------------------------------------------------ chạy
async def run_profile(orc, profile, text="đánh giá nguy cơ giúp tôi"):
    t0 = time.perf_counter()
    out = await orc.run_turn(profile, text, [])
    return out, (time.perf_counter() - t0) * 1000


def transcript_stats(tr):
    per_agent = defaultdict(list)
    for m in tr:
        if m.get("elapsed_ms") is not None and m["performative"] in ("INFORM", "RESPONSE", "FAILURE"):
            per_agent[m["sender"]].append(m["elapsed_ms"])
    return {"n_messages": len(tr), "performatives": Counter(m["performative"] for m in tr),
            "per_agent_ms": {k: sum(v) for k, v in per_agent.items()}}


def binary(y, pred):
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    tp, fp = (y & pred).sum(), (~y & pred).sum()
    fn, tn = (y & ~pred).sum(), (~y & ~pred).sum()
    sens, spec = tp / max(tp + fn, 1), tn / max(tn + fp, 1)
    ppv = tp / max(tp + fp, 1)
    return {"sensitivity": r4(sens), "specificity": r4(spec), "ppv": r4(ppv),
            "f1": r4(2 * ppv * sens / max(ppv + sens, 1e-9)), "positives": int(pred.sum())}


async def main_async(n, llm_n):
    llm.enabled = False  # K1, K2, phần đầu K3: chạy tất định, không LLM
    orc = OrchestratorAgent()
    cases = test_cases(n)
    print(f"{len(cases)} ca kiểm tra, ĐTĐ {cases.diabetes.mean():.1%}")

    k1, k2, stats, slot_filling = [], [], [], []
    for _, row in cases.iterrows():
        out, ms = await run_profile(orc, profile_from(row, True))
        if "consensus" not in out["data"]:  # giá trị ngoài miền sinh lý bị loại → hệ thống hỏi lại
            slot_filling.append({"missing": out["data"].get("missing_required"),
                                 "bmi": float(row.bmi), "age": float(row.age_years)})
            continue
        c = out["data"]["consensus"]
        truth = ada_status(row.fasting_glucose, row.hba1c)
        both = row.fasting_glucose >= 126 and row.hba1c >= 6.5
        k1.append({"truth": truth, "pred": c["glycemic_status"], "tier": c["risk_level"],
                   "confirmation": c["confirmation_required"], "expect_confirmation": truth == "diabetes_range" and not both,
                   "rounds": c["debate_rounds"], "issues": [d["issue"] for d in c["debate"]],
                   "guardrails": [g["level"] for g in c["safety_guardrails"]], "plus": len(c["positive_evidence_plus"]),
                   "minus": len(c["negative_contradictions_minus"]), "ms": ms,
                   "rl_action": out["data"]["treatment"].get("action_id"),
                   "questions": [q["key"] for q in out["data"]["clarification"]["questions"]]})
        stats.append(transcript_stats(out["transcript"]))
        out2, ms2 = await run_profile(orc, profile_from(row, False))
        c2 = out2["data"]["consensus"]
        k2.append({"label": int(row.diabetes), "prediabetes": int(row.prediabetes), "diagnosed": int(row.diagnosed),
                   "fpg": row.fasting_glucose, "ml": c2["ml_probability"], "tier": c2["risk_level"],
                   "score": c2["consensus_risk_score"], "status": c2["glycemic_status"], "ms": ms2})

    # ---- K1
    acc = np.mean([r["truth"] == r["pred"] for r in k1])
    conf_acc = np.mean([r["confirmation"] == r["expect_confirmation"] for r in k1 if r["truth"] == "diabetes_range"])
    k1_res = {"n_evaluated": len(k1), "slot_filling": slot_filling, "glycemic_accuracy": r4(acc), "confirmation_rule_accuracy": r4(conf_acc),
              "truth_counts": dict(Counter(r["truth"] for r in k1)),
              "tier_counts": dict(Counter(r["tier"] for r in k1)),
              "rounds": dict(Counter(r["rounds"] for r in k1)),
              "issues": dict(Counter(i for r in k1 for i in r["issues"])),
              "guardrails": dict(Counter(g for r in k1 for g in r["guardrails"])),
              "plus_mean": r4(np.mean([r["plus"] for r in k1])), "minus_mean": r4(np.mean([r["minus"] for r in k1])),
              "rl_actions": dict(Counter(r["rl_action"] for r in k1)),
              "clarification": {"turns_with_question": r4(np.mean([bool(r["questions"]) for r in k1])),
                                "keys": dict(Counter(q.split(":")[0] for r in k1 for q in r["questions"]))},
              "latency_ms": {"median": r4(np.median([r["ms"] for r in k1])),
                             "p95": r4(np.percentile([r["ms"] for r in k1], 95))}}
    msgs = [s["n_messages"] for s in stats]
    k1_res["messages"] = {"mean": r4(np.mean(msgs)), "min": int(min(msgs)), "max": int(max(msgs))}
    k1_res["performatives"] = dict(sum((s["performatives"] for s in stats), Counter()))
    agent_ms = defaultdict(list)
    for s in stats:
        for a, v in s["per_agent_ms"].items():
            agent_ms[a].append(v)
    k1_res["agent_ms_median"] = {a: r4(np.median(v)) for a, v in agent_ms.items()}
    print("K1:", k1_res)

    # ---- K2
    y = np.array([r["label"] for r in k2])
    kb_only = np.array([r["fpg"] >= 126 for r in k2])
    kb_wide = np.array([r["fpg"] >= 100 for r in k2])
    ml = np.array([r["ml"] for r in k2])
    council = np.array([TIERS.index(r["tier"]) >= 2 for r in k2])
    k2_res = {
        "n": len(k2), "prevalence": r4(y.mean()),
        "knowledge_only_fpg126": binary(y, kb_only), "knowledge_only_fpg100": binary(y, kb_wide),
        "prediction_only_ml30": binary(y, ml >= 0.30),
        "council_high_or_very_high": binary(y, council),
        "auc": {"fpg": r4(roc_auc_score(y, [r["fpg"] for r in k2])), "ml": r4(roc_auc_score(y, ml)),
                "council_score": r4(roc_auc_score(y, [r["score"] for r in k2]))},
        "tier_vs_label": {t: {"n": int(sum(r["tier"] == t for r in k2)),
                              "diabetes_rate": r4(np.mean([r["label"] for r in k2 if r["tier"] == t] or [0]))}
                          for t in TIERS},
        "missed_by_fpg_caught_by_council": int(sum(1 for r, c in zip(k2, council) if r["label"] and r["fpg"] < 126 and c)),
        "missed_by_fpg": int(sum(1 for r in k2 if r["label"] and r["fpg"] < 126)),
        "latency_ms_median": r4(np.median([r["ms"] for r in k2])),
    }
    print("K2:", k2_res)

    # ---- K3: tin nhắn tự do
    rng = np.random.default_rng(7)
    msgs3 = [message_from(row, rng) for _, row in cases.iterrows()]

    async def extract(subset, use_llm):
        llm.enabled = use_llm
        per_field, exact, tiers, lat = defaultdict(list), [], [], []
        asked, blocked = [], 0
        for (msg, truth), r2 in subset:
            p = PatientProfile()
            t0 = time.perf_counter()
            out = await orc.run_turn(p, msg, [])
            lat.append(time.perf_counter() - t0)
            questions = (out["data"].get("clarification") or {}).get("questions", [])
            asked.append([q["key"].split(":")[0] for q in questions])
            if any(q["blocking"] for q in questions):
                # Hệ thống dừng lại hỏi (thai kỳ) → người dùng mô phỏng trả lời đúng sự thật của NHANES
                blocked += 1
                history = [{"sender": "user", "text": msg}, {"sender": "bot", "text": out["text"]}]
                out = await orc.run_turn(p, "không", [], history)
            ok = [field_correct(f, p.get(f), v) for f, v in truth.items()]
            for (f, _), o in zip(truth.items(), ok):
                per_field[f].append(o)
            exact.append(all(ok))
            cons = out["data"].get("consensus")
            tiers.append(cons["risk_level"] == r2["tier"] if cons else False)
        return {"n": len(subset), "exact_match": r4(np.mean(exact)),
                "field_accuracy": {f: r4(np.mean(v)) for f, v in per_field.items()},
                "micro_accuracy": r4(np.mean([o for v in per_field.values() for o in v])),
                "same_tier_as_form": r4(np.mean(tiers)), "latency_s_median": r4(np.median(lat)),
                "clarification": {"turns_with_question": r4(np.mean([bool(a) for a in asked])),
                                  "questions_per_turn": r4(np.mean([len(a) for a in asked])),
                                  "blocked_then_answered": blocked,
                                  "keys": dict(Counter(k for a in asked for k in a))}}

    evaluated = [i for i, (_, row) in enumerate(cases.iterrows())
                 if not any(sf["bmi"] == float(row.bmi) and sf["age"] == float(row.age_years) for sf in slot_filling)]
    pairs = [(msgs3[i], r2) for i, r2 in zip(evaluated, k2)]
    k3 = {"regex": await extract(pairs, False), "examples": [m for m, _ in msgs3[:4]]}
    if llm_n and settings.llm_enabled:
        k3["regex_subset"] = await extract(pairs[:llm_n], False)
        k3["llm"] = await extract(pairs[:llm_n], True)
    print("K3:", {k: v for k, v in k3.items() if k != "examples"})
    llm.enabled = False
    return cases, k1, k2, k1_res, k2_res, k3


# ------------------------------------------------------------------ hình
def plot_k2(k2, k2_res):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.2), gridspec_kw={"width_ratios": [1.3, 1]})
    names = [("knowledge_only_fpg126", "Tri thức\n(FPG ≥ 126)"), ("knowledge_only_fpg100", "Tri thức\n(FPG ≥ 100)"),
             ("prediction_only_ml30", "Dự đoán\n(ML ≥ 30%)"), ("council_high_or_very_high", "Hội đồng\n(Cao/Rất cao)")]
    x = np.arange(len(names))
    for j, (m, lab) in enumerate([("sensitivity", "Độ nhạy"), ("specificity", "Độ đặc hiệu"), ("f1", "F1")]):
        axes[0].bar(x + (j - 1) * 0.27, [k2_res[k][m] for k, _ in names], 0.27, color=PALETTE[j], label=lab)
    axes[0].set_xticks(x, [n for _, n in names], fontsize=8)
    axes[0].set(title="K2 — chỉ có FPG: phát hiện ĐTĐ (nhãn NHANES)", ylim=(0, 1.05))
    axes[0].legend(fontsize=8, ncol=3)
    tv = k2_res["tier_vs_label"]
    axes[1].bar(range(4), [tv[t]["diabetes_rate"] * 100 for t in TIERS], color=[GREY, PALETTE[6], PALETTE[1], PALETTE[3]])
    for i, t in enumerate(TIERS):
        axes[1].text(i, tv[t]["diabetes_rate"] * 100 + 1.5, f"n={tv[t]['n']}", ha="center", fontsize=8)
    axes[1].set_xticks(range(4), ["Thấp", "Trung bình", "Cao", "Rất cao"])
    axes[1].set(title="Tỷ lệ ĐTĐ thật theo mức nguy cơ hội đồng (%)")
    fig.tight_layout()
    save_fig(fig, "ch7_k2")


def plot_ops(k1_res):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3))
    am = sorted(k1_res["agent_ms_median"].items(), key=lambda t: t[1])
    axes[0].barh([a for a, _ in am], [v for _, v in am], color=ACCENT)
    axes[0].set(title="Thời gian xử lý trung vị mỗi lượt theo tác tử (ms, không LLM)")
    axes[0].tick_params(axis="y", labelsize=8)
    iss = sorted(k1_res["issues"].items(), key=lambda t: t[1])
    axes[1].barh([a for a, _ in iss], [v for _, v in iss], color=PALETTE[1])
    axes[1].set(title="Số lần mỗi loại chất vấn được mở (K1)")
    axes[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    save_fig(fig, "ch7_ops")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--llm-n", type=int, default=40)
    args = ap.parse_args()
    random.seed(0)
    cases, k1, k2, k1_res, k2_res, k3 = asyncio.run(main_async(args.n, args.llm_n))
    plot_k2(k2, k2_res)
    plot_ops(k1_res)
    save_results("ch7", {
        "dataset": {"source": "NHANES 2011–2018, tập kiểm tra (20%), có FPG + HbA1c", "n": len(cases),
                    "diabetes_rate": r4(cases.diabetes.mean()), "diagnosed": int(cases.diagnosed.sum()),
                    "undiagnosed": int(cases.undiagnosed.sum()), "prediabetes": int(cases.prediabetes.sum()),
                    "head": cases[["age_years", "sex", "bmi", "waist_cm", "high_bp", "family_history",
                                   "fasting_glucose", "hba1c", "diabetes"]].head(5).to_dict("records")},
        "k1": k1_res, "k2": k2_res, "k3": k3,
    })


if __name__ == "__main__":
    main()
