"""
test_bwk.py

Unit tests for SWUCBBwK and UCBBwK (src/algorithms/bwk.py), verifying:
1. Strict compliance with the BanditAlgorithm interface in src/algorithms/base.py.
2. Exact 1D Linear Programming solution (including the 2-arm mixing property).
3. Sliding-window eviction of stale observations in SWUCBBwK vs. full memory in UCBBwK.
4. Budget tracking and automatic fallback to the null arm (arm 0) upon depletion.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.base import BanditAlgorithm
from src.algorithms.bwk import SWUCBBwK, UCBBwK, solve_1d_knapsack_lp


# =====================================================================
# 1. INTERFACE & PARAMETER VALIDATION TESTS
# =====================================================================

@pytest.mark.parametrize("algo_cls", [UCBBwK, SWUCBBwK])
def test_bwk_subclasses_bandit_algorithm(algo_cls) -> None:
    """Both UCBBwK and SWUCBBwK must subclass BanditAlgorithm."""
    agent = algo_cls(n_arms=5, budget=100.0, horizon=500, rng=np.random.default_rng(42))
    assert isinstance(agent, BanditAlgorithm)
    assert agent.n_arms == 5
    assert agent.t == 0
    assert agent.name == algo_cls.__name__


def test_bwk_invalid_init_parameters_raise() -> None:
    """Invalid budget, horizon, window_size, or null_arm must raise ValueError."""
    with pytest.raises(ValueError, match="budget must be positive"):
        SWUCBBwK(n_arms=4, budget=0.0, horizon=100)
    with pytest.raises(ValueError, match="horizon must be a positive integer"):
        SWUCBBwK(n_arms=4, budget=50.0, horizon=0)
    with pytest.raises(ValueError, match="window_size"):
        SWUCBBwK(n_arms=4, budget=50.0, horizon=100, window_size=0)
    with pytest.raises(ValueError, match="null_arm"):
        SWUCBBwK(n_arms=4, budget=50.0, horizon=100, null_arm=4)


# =====================================================================
# 2. EXACT 1D KNAPSACK LP SOLVER TESTS
# =====================================================================

def test_lp_solver_slide6_exact_mixture() -> None:
    """
    Verify the exact 2-arm LP mixture from Slide 6:
    - Arm 0 (Null arm): r = 0.0, c = 0.0
    - Arm 1 (Hemat / B): r = 0.5, c = 0.2
    - Arm 2 (Agresif / A): r = 0.9, c = 0.6
    Target budget rate rho = 0.30.
    Optimal LP mixture MUST be 75% Arm 1 (Hemat) + 25% Arm 2 (Agresif),
    yielding expected reward 0.60 and expected cost 0.30.
    """
    ucb_rewards = np.array([0.0, 0.50, 0.90])
    lcb_costs = np.array([0.0, 0.20, 0.60])
    rho = 0.30

    probs = solve_1d_knapsack_lp(ucb_rewards, lcb_costs, rho=rho, null_arm=0)
    expected_probs = np.array([0.0, 0.75, 0.25])

    np.testing.assert_allclose(probs, expected_probs, atol=1e-12)
    assert np.isclose(np.dot(probs, ucb_rewards), 0.60)
    assert np.isclose(np.dot(probs, lcb_costs), 0.30)


def test_lp_solver_unconstrained_when_best_arm_is_affordable() -> None:
    """If the highest-reward arm has cost <= rho, LP puts probability 1.0 on it."""
    ucb_rewards = np.array([0.0, 0.40, 0.95])
    lcb_costs = np.array([0.0, 0.50, 0.20])
    rho = 0.30

    probs = solve_1d_knapsack_lp(ucb_rewards, lcb_costs, rho=rho, null_arm=0)
    np.testing.assert_allclose(probs, np.array([0.0, 0.0, 1.0]), atol=1e-12)


# =====================================================================
# 3. SLIDING WINDOW & BUDGET DEPLETION TESTS
# =====================================================================

def test_sw_ucb_bwk_evicts_observations_outside_window() -> None:
    """
    SWUCBBwK with window_size = 5 must only retain the 5 most recent rounds,
    whereas UCBBwK retains all rounds.
    """
    sw_agent = SWUCBBwK(
        n_arms=3, budget=100.0, horizon=200, window_size=5, rng=np.random.default_rng(1)
    )
    stat_agent = UCBBwK(
        n_arms=3, budget=100.0, horizon=200, rng=np.random.default_rng(1)
    )

    # Phase 1: 5 rounds pulling Arm 1 with reward = 1.0, cost = 0.1
    for _ in range(5):
        sw_agent.update(arm=1, reward=1.0, info={"cost": 0.1})
        stat_agent.update(arm=1, reward=1.0, info={"cost": 0.1})

    assert sw_agent.win_counts[1] == 5
    assert stat_agent.win_counts[1] == 5

    # Phase 2: 5 rounds pulling Arm 2 with reward = 0.4, cost = 0.2
    # This must completely push Phase 1 out of sw_agent's window of length 5!
    for _ in range(5):
        sw_agent.update(arm=2, reward=0.4, info={"cost": 0.2})
        stat_agent.update(arm=2, reward=0.4, info={"cost": 0.2})

    assert sw_agent.win_counts[1] == 0
    assert sw_agent.win_counts[2] == 5
    assert stat_agent.win_counts[1] == 5
    assert stat_agent.win_counts[2] == 5


def test_bwk_falls_back_to_null_arm_when_budget_exhausted() -> None:
    """Once remaining_budget hits 0, select_arm() must deterministically return null_arm=0."""
    agent = SWUCBBwK(
        n_arms=4, budget=1.0, horizon=100, window_size=20, null_arm=0, rng=np.random.default_rng(9)
    )
    # Spend 0.6 twice -> remaining_budget becomes 0.0
    agent.update(arm=2, reward=0.8, info={"cost": 0.6})
    assert np.isclose(agent.remaining_budget, 0.4)
    agent.update(arm=3, reward=0.9, info={"cost": 0.6})
    assert np.isclose(agent.remaining_budget, 0.0)

    for _ in range(10):
        assert agent.select_arm() == 0


def test_sw_ucb_bwk_adapts_faster_than_ucb_bwk_under_regime_shift() -> None:
    """
    Simulate an LP-mixing regime shift at round 250 with target rate rho = 0.30:
    - Regime 1 (t < 250): Arm 1 is high-reward/expensive (r=0.95, c=0.45) and
      Arm 2 is moderate-reward/cheap (r=0.50, c=0.15). The optimal LP mixes both
      arms, so both accumulate ~100+ historical pulls.
    - Regime 2 (t >= 250): Arm 1 collapses (r=0.10, c=0.45), while Arm 2 becomes
      strictly optimal and affordable (r=0.90, c=0.15).
    Stationary UCBBwK remains anchored to Arm 1's ~150 historical pulls and
    keeps mixing Arm 1 for ~150 rounds, whereas SWUCBBwK evicts Regime 1 within
    `window_size = 40` rounds and commits to Arm 2.
    """
    horizon = 500
    budget = 150.0  # rho = 0.30
    sw_agent = SWUCBBwK(
        n_arms=3,
        budget=budget,
        horizon=horizon,
        window_size=40,
        conf_scale=0.15,
        rng=np.random.default_rng(2026),
    )
    stat_agent = UCBBwK(
        n_arms=3,
        budget=budget,
        horizon=horizon,
        conf_scale=0.15,
        rng=np.random.default_rng(2026),
    )

    sw_post_shift_arm2 = 0
    stat_post_shift_arm2 = 0

    for t in range(horizon):
        for agent, is_sw in [(sw_agent, True), (stat_agent, False)]:
            arm = agent.select_arm()
            if t < 250:
                rew = 0.95 if arm == 1 else (0.50 if arm == 2 else 0.0)
                cost = 0.45 if arm == 1 else (0.15 if arm == 2 else 0.0)
            else:
                rew = 0.10 if arm == 1 else (0.90 if arm == 2 else 0.0)
                cost = 0.45 if arm == 1 else (0.15 if arm == 2 else 0.0)
                if arm == 2:
                    if is_sw:
                        sw_post_shift_arm2 += 1
                    else:
                        stat_post_shift_arm2 += 1
            agent.update(arm=arm, reward=rew, info={"cost": cost})

    assert sw_post_shift_arm2 > stat_post_shift_arm2 + 50