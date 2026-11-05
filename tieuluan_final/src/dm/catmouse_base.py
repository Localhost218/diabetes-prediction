#!/usr/bin/env python3
"""
DEMO Q-LEARNING: MÈO ĐUỔI CHUỘT TRÊN MA TRẬN LƯỚI (Cat & Mouse Gridworld)

- Môi trường: lưới 8x8 có tường '#'. Mèo (tác tử) đi trước, sau đó chuột di chuyển.
- Trạng thái s = (ô của mèo, ô của chuột)  ->  ~54 x 54 = 2.916 trạng thái rời rạc.
- Hành động a ∈ {Lên, Xuống, Trái, Phải}.
- Phần thưởng: -1 mỗi bước, -3 khi đâm tường, +100 khi bắt được chuột.
- Chuột: "random" (đi ngẫu nhiên) hoặc "flee" (chạy xa mèo theo khoảng cách BFS, có lúc lưỡng lự).

So sánh 3 tác tử mèo:
1. Random Agent         - chọn hướng ngẫu nhiên (mốc dưới).
2. Rule-Based Chaser    - luật "đi về phía làm giảm khoảng cách Manhattan tới chuột" (dễ kẹt sau tường).
3. Q-Learning Agent     - học bảng Q(s, a) bằng TD(0) off-policy với epsilon-greedy giảm dần.
"""

import random
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np

GRID_LAYOUT = [
    "........",
    "..#..#..",
    "..#.....",
    "....##..",
    ".#......",
    ".#..#.#.",
    "....#...",
    "........",
]
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ACTION_NAMES = ["Lên", "Xuống", "Trái", "Phải"]

R_STEP = -1.0
R_WALL = -3.0
R_CATCH = 100.0

Cell = Tuple[int, int]


class CatMouseEnv:
    """Môi trường mèo đuổi chuột. Mèo đi trước; nếu chưa bắt được thì chuột mới di chuyển."""

    def __init__(self, layout: List[str] = GRID_LAYOUT, mouse_mode: str = "flee",
                 mouse_move_prob: float = 0.8, max_steps: int = 100):
        self.layout = layout
        self.R, self.C = len(layout), len(layout[0])
        self.mouse_mode = mouse_mode
        self.mouse_move_prob = mouse_move_prob
        self.max_steps = max_steps

        self.cells: List[Cell] = [(r, c) for r in range(self.R) for c in range(self.C) if layout[r][c] != "#"]
        self.index: Dict[Cell, int] = {cell: i for i, cell in enumerate(self.cells)}
        self.n_cells = len(self.cells)
        self.n_states = self.n_cells * self.n_cells
        self.dist = self._all_pairs_bfs()

    def _free(self, r: int, c: int) -> bool:
        return 0 <= r < self.R and 0 <= c < self.C and self.layout[r][c] != "#"

    def neighbors(self, cell: Cell) -> List[Cell]:
        return [(cell[0] + dr, cell[1] + dc) for dr, dc in ACTIONS if self._free(cell[0] + dr, cell[1] + dc)]

    def _all_pairs_bfs(self) -> np.ndarray:
        """Khoảng cách đường đi ngắn nhất (có tính tường) giữa mọi cặp ô, dùng cho chuột chạy trốn."""
        dist = np.full((self.n_cells, self.n_cells), 999, dtype=int)
        for src in self.cells:
            i = self.index[src]
            dist[i, i] = 0
            q = deque([src])
            while q:
                u = q.popleft()
                for v in self.neighbors(u):
                    j = self.index[v]
                    if dist[i, j] == 999:
                        dist[i, j] = dist[i, self.index[u]] + 1
                        q.append(v)
        return dist

    def state(self, cat: Cell, mouse: Cell) -> int:
        return self.index[cat] * self.n_cells + self.index[mouse]

    def random_start(self, rng: random.Random, min_dist: int = 3) -> Tuple[Cell, Cell]:
        while True:
            cat, mouse = rng.sample(self.cells, 2)
            if self.dist[self.index[cat], self.index[mouse]] >= min_dist:
                return cat, mouse

    def move_mouse(self, mouse: Cell, cat: Cell, rng: random.Random) -> Cell:
        if rng.random() > self.mouse_move_prob:
            return mouse  # chuột lưỡng lự, đứng yên
        options = [n for n in self.neighbors(mouse) if n != cat]
        if not options:
            return mouse
        if self.mouse_mode == "flee" and rng.random() < 0.8:
            ci = self.index[cat]
            best = max(self.dist[ci, self.index[n]] for n in options)
            options = [n for n in options if self.dist[ci, self.index[n]] == best]
        return rng.choice(options)

    def step(self, cat: Cell, mouse: Cell, a: int, rng: random.Random) -> Tuple[Cell, Cell, float, bool]:
        """Trả về (mèo mới, chuột mới, phần thưởng, đã bắt được chưa)."""
        nr, nc = cat[0] + ACTIONS[a][0], cat[1] + ACTIONS[a][1]
        reward = R_STEP
        if self._free(nr, nc):
            cat = (nr, nc)
        else:
            reward = R_WALL  # đâm tường: đứng yên và bị phạt
        if cat == mouse:
            return cat, mouse, R_CATCH, True
        mouse = self.move_mouse(mouse, cat, rng)
        return cat, mouse, reward, False


# ==============================================================================
# CÁC TÁC TỬ MÈO
# ==============================================================================
class RandomCat:
    name = "Random Agent"

    def act(self, env: CatMouseEnv, cat: Cell, mouse: Cell, rng: random.Random) -> int:
        return rng.randrange(4)


class RuleBasedCat:
    """Luật phản xạ: chọn hướng làm khoảng cách Manhattan tới chuột giảm nhiều nhất (không biết tường)."""
    name = "Rule-Based Chaser"

    def act(self, env: CatMouseEnv, cat: Cell, mouse: Cell, rng: random.Random) -> int:
        def manhattan(a: int) -> int:
            r, c = cat[0] + ACTIONS[a][0], cat[1] + ACTIONS[a][1]
            return abs(r - mouse[0]) + abs(c - mouse[1])
        return min(range(4), key=manhattan)


class QLearningCat:
    """Tác tử học tăng cường Q-Learning dạng bảng (off-policy TD control)."""
    name = "Q-Learning Agent"

    def __init__(self, env: CatMouseEnv, alpha: float = 0.2, gamma: float = 0.95):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.Q = np.zeros((env.n_states, 4), dtype=float)

    def act(self, env: CatMouseEnv, cat: Cell, mouse: Cell, rng: random.Random) -> int:
        return int(np.argmax(self.Q[env.state(cat, mouse)]))

    def train(self, episodes: int = 15000, eps_min: float = 0.05, seed: int = 0) -> Dict[str, List[float]]:
        env, rng = self.env, random.Random(seed)
        alpha, gamma = self.alpha, self.gamma
        Q = self.Q.tolist()  # list thuần Python nhanh hơn numpy khi truy cập từng phần tử
        decay_until = 0.7 * episodes  # epsilon giảm tuyến tính trong 70% số tập đầu
        history = {"reward": [], "steps": [], "caught": []}
        for ep in range(episodes):
            eps = max(eps_min, 1.0 - ep / decay_until)
            cat, mouse = env.random_start(rng)
            s = env.state(cat, mouse)
            total, caught, n = 0.0, False, 0
            while n < env.max_steps:
                q = Q[s]
                a = rng.randrange(4) if rng.random() < eps else q.index(max(q))
                cat, mouse, r, caught = env.step(cat, mouse, a, rng)
                s2 = env.state(cat, mouse)
                target = r if caught else r + gamma * max(Q[s2])
                q[a] += alpha * (target - q[a])
                s, total, n = s2, total + r, n + 1
                if caught:
                    break
            history["reward"].append(total)
            history["steps"].append(n)
            history["caught"].append(float(caught))
        self.Q = np.array(Q)
        return history


# ==============================================================================
# CHẠY THỬ & ĐÁNH GIÁ
# ==============================================================================
def run_episode(env: CatMouseEnv, agent, cat: Optional[Cell] = None, mouse: Optional[Cell] = None,
                seed: int = 0) -> Dict:
    """Chạy một ván với chính sách cố định, trả về toàn bộ quỹ đạo để vẽ lại."""
    rng = random.Random(seed)
    if cat is None or mouse is None:
        cat, mouse = env.random_start(rng)
    frames = [{"cat": list(cat), "mouse": list(mouse), "action": None, "reward": 0.0}]
    total, caught = 0.0, False
    for _ in range(env.max_steps):
        a = agent.act(env, cat, mouse, rng)
        cat, mouse, r, caught = env.step(cat, mouse, a, rng)
        total += r
        frames.append({"cat": list(cat), "mouse": list(mouse), "action": ACTION_NAMES[a], "reward": r})
        if caught:
            break
    return {"agent": agent.name, "caught": caught, "steps": len(frames) - 1, "total_reward": total, "frames": frames}


def evaluate(env: CatMouseEnv, agent, n_episodes: int = 300, seed: int = 123) -> Dict[str, float]:
    """Đánh giá trên cùng một tập vị trí xuất phát cho mọi tác tử để so sánh công bằng."""
    results = [run_episode(env, agent, seed=seed + i) for i in range(n_episodes)]
    caught = [r for r in results if r["caught"]]
    return {
        "agent": agent.name,
        "catch_rate": len(caught) / n_episodes,
        "avg_steps": float(np.mean([r["steps"] for r in results])),
        "avg_reward": float(np.mean([r["total_reward"] for r in results])),
    }


def learning_curve(history: Dict[str, List[float]], n_bins: int = 50) -> List[Dict[str, float]]:
    """Gộp lịch sử huấn luyện thành n_bins điểm (trung bình trượt theo khối) để vẽ biểu đồ."""
    n = len(history["steps"])
    size = max(1, n // n_bins)
    curve = []
    for start in range(0, n, size):
        sl = slice(start, start + size)
        curve.append({
            "episode": min(start + size, n),
            "avg_steps": float(np.mean(history["steps"][sl])),
            "avg_reward": float(np.mean(history["reward"][sl])),
            "catch_rate": float(np.mean(history["caught"][sl])),
        })
    return curve


if __name__ == "__main__":
    env = CatMouseEnv()
    print(f"Lưới {env.R}x{env.C}, {env.n_cells} ô trống, {env.n_states} trạng thái")
    t0 = time.time()
    ql = QLearningCat(env)
    hist = ql.train(episodes=15000)
    print(f"Huấn luyện xong trong {time.time() - t0:.1f}s")
    for agent in (RandomCat(), RuleBasedCat(), ql):
        m = evaluate(env, agent)
        print(f"{m['agent']:<20} bắt được {m['catch_rate']:6.1%} | số bước TB {m['avg_steps']:5.1f} | reward TB {m['avg_reward']:7.1f}")
