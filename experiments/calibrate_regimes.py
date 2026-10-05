"""
calibrate_regimes.py  --  run BEFORE Stage 1 (and again whenever Bob changes).

For each configuration: pin Bob to Rational and to Escalating, play every
threshold arm theta FIXED, print the mean reward per arm and the separation
score (see src/simulation/calibration.py). Stage 1 only makes sense if the
best arm differs between regimes and using the wrong one is costly.

The control row "old Bob" (prior_sunk_cost = 0) shows why prior_sunk_cost
was introduced: without it Bob only gets hooked after HE bids, so when the
learner opens the auction he folds like a rational player and escalation can
only matter in the rounds where Bob starts -> a much weaker regime effect.

For a systematic search use experiments/sweep_bob.py.

    python -m experiments.calibrate_regimes
"""
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.simulation.calibration import arm_rewards, opening_bid, separation_score

T, SEEDS = 600, 4

# (label, budget, stake, bob kwargs)
def bob(mu=0.5, prior=3.0, rate=3.0, thr=0.5):
    return dict(mu=mu, sunk_cost_threshold=thr, prior_sunk_cost=prior,
                escalation_rate=rate)

CONFIGS = [
    ("RECOMMENDED  b=12 s=8  (prior_sunk_cost=3)", 12, 8.0, bob()),
    ("alt          b=14 s=10 (prior_sunk_cost=3)", 14, 10.0, bob()),
    ("control: old Bob (prior_sunk_cost=0), b=12 s=8", 12, 8.0, bob(prior=0.0)),
    ("control: Stage-0 style b=12 s=5, mu=0.8, prior=0", 12, 5.0, bob(mu=0.8, prior=0.0, thr=0.5)),
    ("control: b=12 s=10, mu=0.8, prior=3", 12, 10.0, bob(mu=0.8, prior=3.0)),
]

if __name__ == "__main__":
    warnings.simplefilter("ignore")
    for label, b, s, kw in CONFIGS:
        kw = dict(kw, escalation_ceiling=float(b))
        rr = arm_rewards(b, s, kw, False, T=T, seeds=SEEDS)
        re = arm_rewards(b, s, kw, True, T=T, seeds=SEEDS)
        sc = separation_score(rr, re)
        print(f"\n=== {label} | x0={opening_bid(b, s)} ===")
        print(f"  rational   {np.round(rr, 2).tolist()}")
        print(f"  escalating {np.round(re, 2).tolist()}")
        print(f"  best arms: rational={sc['best_rational']}  escalating={sc['best_escalating']}")
        print(f"  cross-regime loss A={sc['A']:.3f} B={sc['B']:.3f}  ->  score={sc['score']:.3f}"
              f"  {'OK' if sc['score'] >= 0.10 else 'too weak for Stage 1'}")