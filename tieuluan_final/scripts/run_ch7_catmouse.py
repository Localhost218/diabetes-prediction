"""Chương 7B — chạy ba thí nghiệm mèo đuổi chuột, ghi outputs/metrics/ch7_catmouse.json và bảng Q vào models/ch7/."""
import json
import sys
import time
from pathlib import Path

import numpy as np

TLF = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TLF / "src"))
from dm import ch7_catmouse as CM  # noqa: E402

OUT, MOD = TLF / "outputs" / "metrics", TLF / "models" / "ch7"
OUT.mkdir(parents=True, exist_ok=True)
MOD.mkdir(parents=True, exist_ok=True)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


res = {}
t0 = time.time()
res["A"], env, q = CM.experiment_a()
res["A"]["train_s"] = time.time() - t0
np.save(MOD / "cat_q_single.npy", q.Q)
log("A", res["A"]["eval"])
t0 = time.time()
res["B"], _, (Q1, Q2) = CM.experiment_b()
res["B"]["train_s"] = time.time() - t0
np.savez_compressed(MOD / "cats_iql.npz", Q1=Q1, Q2=Q2)
log("B", {k: v for k, v in res["B"].items() if k != "curve"})
t0 = time.time()
res["C"], _, (Qc, Qm) = CM.experiment_c(q)
res["C"]["train_s"] = time.time() - t0
np.savez_compressed(MOD / "adversarial.npz", Qc=Qc, Qm=Qm)
log("C", res["C"]["table"])
(OUT / "ch7_catmouse.json").write_text(json.dumps(res, ensure_ascii=False, indent=1))
log("XONG")
