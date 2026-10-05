"""
calibration.py

Pre-flight measurements for Stage 1: how much does Bob's regime matter to
the learner? Used by experiments/calibrate_regimes.py (one config) and
experiments/sweep_bob.py (grid search). No bandit algorithm involved.

Idea
----
Pin Bob to one regime (Bob(fixed_regime=...)), play every threshold arm
theta FIXED against him, and record the mean reward per arm:

    r_rat[theta]   against an always-Rational Bob
    r_esc[theta]   against an always-Escalating Bob

Stage 1 only makes sense if the best arm DIFFERS between the regimes AND
using the wrong regime's best arm is costly. `separation_score` measures
exactly that:

    A = best_esc - max_{theta in best arms of Rational} r_esc[theta]
        (what you lose in the escalating regime by sticking to what was
         optimal while Bob was rational)
    B = best_rat - max_{theta in best arms of Escalating} r_rat[theta]
        (what you lose in the rational regime by sticking to what was
         optimal while Bob was escalating)
    score = min(A, B)

Both directions must hurt, otherwise there is nothing for a change-
tracking learner (EXP3.S / Page-Hinkley) to gain. Rewards live in [0,1]
(DollarAuction.normalize_reward), so score is in reward units.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from src.environment.dollar_auction import DollarAuction
from src.environment.strategies import build_threshold_arm_set
from src.opponents.bob_sunkcost import Bob


def arm_rewards(
    budget: int,
    stake: float,
    bob_kwargs: Dict,
    escalating: bool,
    T: int = 250,
    seeds: int = 3,
    seed0: int = 1000,
) -> np.ndarray:
    """Mean reward of each fixed threshold arm vs. a Bob pinned to one regime."""
    env = DollarAuction(stake=stake, budget=budget)
    arms = build_threshold_arm_set(budget, stake)
    out = np.zeros(len(arms))
    for i, arm in enumerate(arms):
        total, n = 0.0, 0
        for k in range(seeds):
            start_rng = np.random.default_rng(seed0 + k)
            bob = Bob(
                stake=stake, budget=budget, fixed_regime=escalating,
                rng=np.random.default_rng(seed0 + 10_000 + k), **bob_kwargs,
            )
            for _ in range(T):
                bob.begin_round()
                starts = bool(start_rng.integers(0, 2))
                res = env.run(arm, bob.get_strategy(),
                              agent_starts=starts, rng=start_rng)
                total += DollarAuction.normalize_reward(
                    res.agent_payoff, budget, stake)
                n += 1
        out[i] = total / n
    return out


def separation_score(r_rat: np.ndarray, r_esc: np.ndarray, tol: float = 0.01) -> Dict:
    """See module docstring. Arms within `tol` of the best count as tied-best."""
    best_rat = np.where(r_rat >= r_rat.max() - tol)[0]
    best_esc = np.where(r_esc >= r_esc.max() - tol)[0]
    A = float(r_esc.max() - r_esc[best_rat].max())
    B = float(r_rat.max() - r_rat[best_esc].max())
    return {
        "score": min(A, B), "A": A, "B": B,
        "best_rational": best_rat.tolist(), "best_escalating": best_esc.tolist(),
        "max_rational": float(r_rat.max()), "max_escalating": float(r_esc.max()),
    }


def opening_bid(budget: int, stake: float) -> int:
    """O'Neill's opening bid x0 = (b-1) mod (s-1) + 1 (Waniek Sec. 5.1)."""
    return int((budget - 1) % (stake - 1) + 1)