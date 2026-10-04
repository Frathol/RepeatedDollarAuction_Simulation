"""
calibrate_regimes.py  --  run BEFORE Stage 1a/1b.

For a given (budget, stake, Bob params), play every threshold arm theta
FIXED against (a) an always-rational Bob and (b) an always-escalating Bob
and print the mean reward per arm. Stage 1 only makes sense if the best arm
DIFFERS between the two regimes (otherwise there is nothing for EXP3.S /
Page-Hinkley to track). If both rows show the same best theta and nearly
identical rewards, retune Bob / budget / stake.

Run from the project root:
    python -m experiments.calibrate_regimes
"""
import numpy as np

from src.algorithms.base import BanditAlgorithm
from src.simulation.config import SimConfig
from src.simulation.runner import make_simulation

GRID = [  # (budget, stake, bob_sunk_cost_threshold)
    (12, 10.0, None),
    (12, 5.0, None),
    (12, 5.0, 0.5),
    (8, 10.0, None),
]
T, SEEDS, BURN = 800, 3, 50


class FixedArm(BanditAlgorithm):
    def __init__(self, n, arm):
        super().__init__(n)
        self.arm = arm

    def select_arm(self):
        return self.arm

    def update(self, arm, reward, info=None):
        pass


def arm_rewards(budget, stake, thr, gain):
    out = []
    for th in range(budget + 1):
        vals = []
        for seed in range(SEEDS):
            cfg = SimConfig(budget=budget, stake=stake, T=T, seed=seed,
                            bob_gain=gain, bob_sunk_cost_threshold=thr)
            sim = make_simulation(cfg, FixedArm(budget + 1, th))
            sim.run()
            vals.append(np.mean([r["reward"] for r in sim.records[BURN:]]))
        out.append(np.mean(vals))
    return np.array(out)


if __name__ == "__main__":
    for budget, stake, thr in GRID:
        print(f"\n=== budget={budget} stake={stake} sunk_cost_thr={thr} ===")
        rows = {}
        for label, gain in (("rational", 0.0), ("escalating", 100.0)):
            r = arm_rewards(budget, stake, thr, gain)
            rows[label] = r
            print(f"  {label:10s} best theta={r.argmax():2d}  "
                  f"{np.round(r, 2).tolist()}")
        same = rows["rational"].argmax() == rows["escalating"].argmax()
        gap = np.abs(rows["rational"] - rows["escalating"]).max()
        print(f"  -> best arm differs between regimes: {not same} | "
              f"max reward gap: {gap:.3f}")