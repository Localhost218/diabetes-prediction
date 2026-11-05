"""Chương 7B — Hệ đa tác tử học tăng cường: mèo đuổi chuột.

Mở rộng bài Agent của tác giả (catmouse_base.py: lưới 8×8 có tường, một mèo Q-learning, chuột chạy trốn theo luật)
thành ba thí nghiệm đa tác tử:
  A. Một mèo:   Random / Rule-Based (Manhattan) / Q-learning  vs  chuột chạy trốn theo luật   (bài gốc)
  B. Hai mèo hợp tác (cùng phần thưởng):  Rule+Rule  vs  hai mèo Q-learning ĐỘC LẬP (IQL)  vs chuột chạy trốn
  C. Đối kháng, CẢ HAI CÙNG HỌC: mèo Q-learning vs chuột Q-learning (môi trường không dừng với mỗi bên)
Mọi đánh giá dùng cùng một tập vị trí xuất phát cố định (giống evaluate() của bài gốc).
"""
import random
from typing import Dict, List, Tuple

import numpy as np

from .catmouse_base import ACTIONS, R_CATCH, R_STEP, R_WALL, CatMouseEnv, QLearningCat, RandomCat, RuleBasedCat

Cell = Tuple[int, int]


def manhattan_action(cat, target):
    def d(a):
        return abs(cat[0] + ACTIONS[a][0] - target[0]) + abs(cat[1] + ACTIONS[a][1] - target[1])
    return min(range(4), key=d)


def curve(hist, n_bins=40):
    n = len(hist)
    size = max(1, n // n_bins)
    return [{"episode": min(s + size, n), "catch_rate": float(np.mean([h[0] for h in hist[s:s + size]])),
             "avg_steps": float(np.mean([h[1] for h in hist[s:s + size]]))} for s in range(0, n, size)]


# ======================================================================== A. một mèo (bài gốc)
def experiment_a(episodes=15000, n_eval=300, seed=0):
    from .catmouse_base import evaluate
    env = CatMouseEnv()
    q = QLearningCat(env)
    h = q.train(episodes=episodes, seed=seed)
    hist = list(zip(h["caught"], h["steps"]))
    rows = [evaluate(env, a, n_episodes=n_eval) for a in (RandomCat(), RuleBasedCat(), q)]
    return {"n_cells": env.n_cells, "n_states": env.n_states, "eval": rows, "curve": curve(hist)}, env, q


# ======================================================================== B. hai mèo hợp tác
class TwoCatEnv(CatMouseEnv):
    """Hai mèo đi lần lượt, rồi chuột chạy xa mèo GẦN NHẤT. Phần thưởng chung: −1 mỗi lượt, −3 mỗi lần đâm tường,
    +100 khi một mèo bất kỳ bắt được chuột. Trạng thái chung (mèo 1, mèo 2, chuột): 54³ ≈ 157 nghìn."""

    def jstate(self, c1, c2, m):
        n = self.n_cells
        return (self.index[c1] * n + self.index[c2]) * n + self.index[m]

    def start(self, rng, min_dist=3):
        while True:
            c1, c2, m = rng.sample(self.cells, 3)
            if min(self.dist[self.index[c1], self.index[m]], self.dist[self.index[c2], self.index[m]]) >= min_dist:
                return c1, c2, m

    def _move(self, cat, a):
        nr, nc = cat[0] + ACTIONS[a][0], cat[1] + ACTIONS[a][1]
        return ((nr, nc), 0.0) if self._free(nr, nc) else (cat, R_WALL - R_STEP)

    def move_mouse2(self, m, c1, c2, rng):
        if rng.random() > self.mouse_move_prob:
            return m
        opts = [n for n in self.neighbors(m) if n not in (c1, c2)]
        if not opts:
            return m
        if rng.random() < 0.8:
            def safety(n):
                return min(self.dist[self.index[c1], self.index[n]], self.dist[self.index[c2], self.index[n]])
            best = max(safety(n) for n in opts)
            opts = [n for n in opts if safety(n) == best]
        return rng.choice(opts)

    def step2(self, c1, c2, m, a1, a2, rng):
        c1, p1 = self._move(c1, a1)
        if c1 == m:
            return c1, c2, m, R_CATCH, True
        c2, p2 = self._move(c2, a2)
        if c2 == m:
            return c1, c2, m, R_CATCH, True
        return c1, c2, self.move_mouse2(m, c1, c2, rng), R_STEP + p1 + p2, False


def train_iql(env, episodes=40000, alpha=0.2, gamma=0.95, seed=0, eps_min=0.05):
    """Independent Q-learning: mỗi mèo một bảng Q riêng trên trạng thái chung, coi mèo kia là một phần môi trường."""
    rng = random.Random(seed)
    nS = env.n_cells ** 3
    Q1, Q2 = np.zeros((nS, 4)), np.zeros((nS, 4))
    hist = []
    for ep in range(episodes):
        eps = max(eps_min, 1.0 - ep / (0.7 * episodes))
        c1, c2, m = env.start(rng)
        s = env.jstate(c1, c2, m)
        n, caught = 0, False
        while n < env.max_steps:
            a1 = rng.randrange(4) if rng.random() < eps else int(Q1[s].argmax())
            a2 = rng.randrange(4) if rng.random() < eps else int(Q2[s].argmax())
            c1, c2, m, r, caught = env.step2(c1, c2, m, a1, a2, rng)
            s2 = env.jstate(c1, c2, m)
            t1 = r if caught else r + gamma * Q1[s2].max()
            t2 = r if caught else r + gamma * Q2[s2].max()
            Q1[s, a1] += alpha * (t1 - Q1[s, a1])
            Q2[s, a2] += alpha * (t2 - Q2[s, a2])
            s, n = s2, n + 1
            if caught:
                break
        hist.append((float(caught), n))
    return Q1, Q2, hist


def eval_two(env, policy, n=300, seed=123):
    """policy(c1, c2, m) -> (a1, a2). Cùng tập vị trí xuất phát cho mọi chính sách."""
    caught, steps = [], []
    for i in range(n):
        rng = random.Random(seed + i)
        c1, c2, m = env.start(rng)
        k, ok = 0, False
        while k < env.max_steps:
            a1, a2 = policy(c1, c2, m)
            c1, c2, m, _, ok = env.step2(c1, c2, m, a1, a2, rng)
            k += 1
            if ok:
                break
        caught.append(ok)
        steps.append(k)
    return {"catch_rate": float(np.mean(caught)), "avg_steps": float(np.mean(steps))}


def experiment_b(episodes=40000, seed=0):
    env = TwoCatEnv()
    Q1, Q2, hist = train_iql(env, episodes=episodes, seed=seed)
    rule = lambda c1, c2, m: (manhattan_action(c1, m), manhattan_action(c2, m))   # noqa: E731
    learned = lambda c1, c2, m: (int(Q1[env.jstate(c1, c2, m)].argmax()), int(Q2[env.jstate(c1, c2, m)].argmax()))  # noqa: E731
    return {"n_states": env.n_cells ** 3, "rule_rule": eval_two(env, rule), "iql": eval_two(env, learned),
            "curve": curve(hist)}, env, (Q1, Q2)


# ======================================================================== C. đối kháng, cả hai cùng học
def train_adversarial(env, episodes=30000, alpha=0.2, gamma=0.95, seed=0, eps_min=0.05):
    """Mèo và chuột cùng học Q-learning trên cùng trạng thái (ô mèo, ô chuột) — trò chơi đối kháng.
    Mèo: −1/bước, −3 đâm tường, +100 bắt được (như bài gốc). Chuột: +1 mỗi lượt sống sót, −100 khi bị bắt.
    Kết quả nước đi của chuột chỉ biết sau lượt kế tiếp của mèo, nên cập nhật Q của chuột được hoãn một lượt.
    Chuột vẫn chỉ di chuyển với xác suất mouse_move_prob như bài gốc, nhưng hướng đi do bảng Q của chuột chọn."""
    rng = random.Random(seed)
    Qc, Qm = np.zeros((env.n_states, 4)), np.zeros((env.n_states, 4))
    hist = []
    for ep in range(episodes):
        eps = max(eps_min, 1.0 - ep / (0.7 * episodes))
        cat, mouse = env.random_start(rng)
        pend, n, caught = None, 0, False            # pend = (trạng thái, hành động) của chuột chờ cập nhật
        while n < env.max_steps:
            s = env.state(cat, mouse)
            ac = rng.randrange(4) if rng.random() < eps else int(Qc[s].argmax())
            nr, nc = cat[0] + ACTIONS[ac][0], cat[1] + ACTIONS[ac][1]
            rc = R_STEP
            if env._free(nr, nc):
                cat = (nr, nc)
            else:
                rc = R_WALL
            n += 1
            if cat == mouse:
                Qc[s, ac] += alpha * (R_CATCH - Qc[s, ac])
                if pend:
                    Qm[pend] += alpha * (-100.0 - Qm[pend])
                caught = True
                break
            sm = env.state(cat, mouse)              # chuột quan sát sau khi mèo đi
            if pend:                                # nước trước của chuột đã sống sót qua lượt mèo
                Qm[pend] += alpha * (1.0 + gamma * Qm[sm].max() - Qm[pend])
            am = rng.randrange(4) if rng.random() < eps else int(Qm[sm].argmax())
            if rng.random() <= env.mouse_move_prob:
                mr, mc = mouse[0] + ACTIONS[am][0], mouse[1] + ACTIONS[am][1]
                if env._free(mr, mc) and (mr, mc) != cat:
                    mouse = (mr, mc)
            pend = (sm, am)
            s2 = env.state(cat, mouse)
            Qc[s, ac] += alpha * (rc + gamma * Qc[s2].max() - Qc[s, ac])
        if not caught and pend:                     # hết giờ: chuột thắng ván
            Qm[pend] += alpha * (1.0 - Qm[pend])
        hist.append((float(caught), n))
    return Qc, Qm, hist


def eval_pair(env, cat_pol, mouse_pol, n=300, seed=123):
    caught, steps = [], []
    for i in range(n):
        rng = random.Random(seed + i)
        cat, mouse = env.random_start(rng)
        k, ok = 0, False
        while k < env.max_steps:
            a = cat_pol(cat, mouse, rng)
            nr, nc = cat[0] + ACTIONS[a][0], cat[1] + ACTIONS[a][1]
            if env._free(nr, nc):
                cat = (nr, nc)
            k += 1
            if cat == mouse:
                ok = True
                break
            mouse = mouse_pol(cat, mouse, rng)
        caught.append(ok)
        steps.append(k)
    return {"catch_rate": float(np.mean(caught)), "avg_steps": float(np.mean(steps))}


def experiment_c(q_cat_vs_rule, episodes=30000, seed=0):
    """So bốn cặp: mèo (Q học với chuột luật | Q học đối kháng) × chuột (luật chạy trốn | Q học đối kháng)."""
    env = CatMouseEnv()
    Qc, Qm, hist = train_adversarial(env, episodes=episodes, seed=seed)

    def cat_from(Q):
        return lambda c, m, rng: int(Q[env.state(c, m)].argmax())

    def mouse_rule(c, m, rng):
        return env.move_mouse(m, c, rng)

    def mouse_learned(c, m, rng):
        if rng.random() > env.mouse_move_prob:
            return m
        a = int(Qm[env.state(c, m)].argmax())
        nr, nc = m[0] + ACTIONS[a][0], m[1] + ACTIONS[a][1]
        return (nr, nc) if env._free(nr, nc) and (nr, nc) != c else m

    cats = {"Q (học với chuột luật)": cat_from(q_cat_vs_rule.Q), "Q (học đối kháng)": cat_from(Qc),
            "Rule-Based": lambda c, m, rng: manhattan_action(c, m)}
    mice = {"chuột luật (chạy trốn)": mouse_rule, "chuột Q (học đối kháng)": mouse_learned}
    table = [{"cat": cn, "mouse": mn, **eval_pair(env, cp, mp)} for cn, cp in cats.items() for mn, mp in mice.items()]
    return {"table": table, "curve": curve(hist)}, env, (Qc, Qm)
