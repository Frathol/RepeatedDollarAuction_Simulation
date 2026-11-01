"""
test_bwk.py

Unit tests for SWUCBBwK, UCBBwK, and ContextualUCBBwK (src/algorithms/bwk.py),
verifying:
1. Strict compliance with the BanditAlgorithm interface in src/algorithms/base.py.
2. Exact 1D Linear Programming solution (including the 2-arm mixing property).
3. Sliding-window eviction of stale observations in SWUCBBwK vs. full memory in UCBBwK.
4. Budget tracking and automatic fallback to the null arm (arm 0) upon depletion.
5. Non-zero `null_arm_reward = b / (b + s)` behavior when all active arms are losing
   (e.g., against Escalating Bob).
6. Coupled Policy-Level LP (`solve_policy_knapsack_lp`) and `ContextualUCBBwK`
   cross-regime budget pacing and causal price regime discretization.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.base import BanditAlgorithm
from src.algorithms.bwk import (
    ContextualUCBBwK,
    SWUCBBwK,
    UCBBwK,
    discretize_price_causal,
    solve_1d_knapsack_lp,
    solve_policy_knapsack_lp,
)


# =====================================================================
# 1. INTERFACE & PARAMETER VALIDATION TESTS
# =====================================================================

@pytest.mark.parametrize("algo_cls", [UCBBwK, SWUCBBwK, ContextualUCBBwK])
def test_bwk_subclasses_bandit_algorithm(algo_cls) -> None:
    """UCBBwK, SWUCBBwK, and ContextualUCBBwK must subclass BanditAlgorithm."""
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
    with pytest.raises(ValueError, match="n_contexts"):
        ContextualUCBBwK(n_arms=4, n_contexts=0, budget=50.0, horizon=100)


# =====================================================================
# 2. EXACT 1D & POLICY KNAPSACK LP SOLVER TESTS
# =====================================================================

def test_lp_solver_slide6_exact_mixture() -> None:
    """
    Verify the exact 2-arm LP mixture:
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


def test_null_arm_reward_prevents_suicidal_bidding_under_escalation() -> None:
    """
    When rewards are Min-Max shifted so Null Arm (theta=0) yields b/(b+s) = 0.60
    and all active arms yield < 0.60 (as when Bob is Escalating), setting
    `null_arm_reward = 0.60` ensures the LP puts 100% probability on Arm 0!
    """
    agent = SWUCBBwK(
        n_arms=3,
        budget=50.0,
        horizon=100,
        window_size=20,
        conf_scale=0.0,  # zero bonus to test pure mean comparison
        null_arm=0,
        null_arm_reward=0.60,
    )
    # Active arms 1 and 2 have losing rewards (0.35 and 0.40 < 0.60)
    agent.update(arm=1, reward=0.35, info={"cost": 0.25})
    agent.update(arm=2, reward=0.40, info={"cost": 0.50})

    chosen = agent.select_arm()
    assert chosen == 0
    np.testing.assert_allclose(agent.last_lp_distribution, np.array([1.0, 0.0, 0.0]))


def test_policy_knapsack_lp_shifts_budget_from_expensive_to_cheap_regime() -> None:
    """
    Verify that `solve_policy_knapsack_lp` allocates budget across regimes:
    Suppose 2 equiprobable regimes (p_0 = 0.5 Cheap, p_1 = 0.5 Expensive),
    target average rate rho = 0.20, and Null Arm (arm 0) has r = 0.60, c = 0.0:
    - Regime 0 (Cheap): Arm 1 gives r = 0.80 at cost c = 0.40.
    - Regime 1 (Expensive): Arm 1 gives r = 0.80 at cost c = 0.80.
    A per-context LP with cap rho = 0.20 could only play Arm 1 50% in Regime 0
    and 25% in Regime 1 (expected reward = 0.5*0.70 + 0.5*0.65 = 0.675).
    The coupled Policy LP saves 100% budget in Regime 1 (playing Arm 0, cost 0)
    and spends 100% on Arm 1 in Regime 0 (cost 0.5 * 0.40 = 0.20 == rho),
    achieving strictly higher expected reward: 0.5*0.80 + 0.5*0.60 = 0.700!
    """
    ucb_2d = np.array([
        [0.60, 0.80],  # Regime 0 (Cheap)
        [0.60, 0.80],  # Regime 1 (Expensive)
    ])
    lcb_2d = np.array([
        [0.00, 0.40],  # Regime 0: Arm 1 costs 0.40
        [0.00, 0.80],  # Regime 1: Arm 1 costs 0.80
    ])
    p_s = np.array([0.5, 0.5])
    rho = 0.20

    policy_probs = solve_policy_knapsack_lp(ucb_2d, lcb_2d, p_s, rho=rho, null_arm=0)

    # Regime 0 plays Arm 1 with prob 1.0; Regime 1 plays Arm 0 (Pass) with prob 1.0
    np.testing.assert_allclose(policy_probs[0], np.array([0.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(policy_probs[1], np.array([1.0, 0.0]), atol=1e-12)


# =====================================================================
# 3. SLIDING WINDOW, CONTEXTUAL & BUDGET DEPLETION TESTS
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

    for _ in range(5):
        sw_agent.update(arm=1, reward=1.0, info={"cost": 0.1})
        stat_agent.update(arm=1, reward=1.0, info={"cost": 0.1})

    assert sw_agent.win_counts[1] == 5
    assert stat_agent.win_counts[1] == 5

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
    agent.update(arm=2, reward=0.8, info={"cost": 0.6})
    assert np.isclose(agent.remaining_budget, 0.4)
    agent.update(arm=3, reward=0.9, info={"cost": 0.6})
    assert np.isclose(agent.remaining_budget, 0.0)

    for _ in range(10):
        assert agent.select_arm() == 0


def test_contextual_ucb_bwk_separates_regime_statistics() -> None:
    """
    ContextualUCBBwK must store observations in separate regime buckets and
    adapt immediately when switching between observed contexts.
    """
    ctx_agent = ContextualUCBBwK(
        n_arms=3,
        n_contexts=2,
        budget=40.0,
        horizon=200,  # rho = 0.20
        conf_scale=0.0,
        null_arm=0,
        null_arm_reward=0.60,
        couple_regimes=True,
        rng=np.random.default_rng(7),
    )

    # Populate Regime 0 (Cheap) and Regime 1 (Expensive)
    for _ in range(10):
        ctx_agent.set_context(0)
        ctx_agent.update(arm=1, reward=0.80, info={"cost": 0.30, "context": 0})
        ctx_agent.update(arm=2, reward=0.75, info={"cost": 0.20, "context": 0})

        ctx_agent.set_context(1)
        ctx_agent.update(arm=1, reward=0.80, info={"cost": 0.90, "context": 1})
        ctx_agent.update(arm=2, reward=0.75, info={"cost": 0.80, "context": 1})

    # In Regime 1 (Expensive), Policy LP saves budget by playing Null Arm (0)
    assert ctx_agent.select_arm(context=1) == 0
    # In Regime 0 (Cheap), Policy LP spends budget aggressively on active arms
    assert ctx_agent.select_arm(context=0) in (1, 2)


def test_discretize_price_causal_no_future_leakage_and_flat_control() -> None:
    """Flat series maps to all zeros; varying series maps causally to 0..M-1."""
    flat = np.ones(100)
    reg_flat = discretize_price_causal(flat, n_contexts=3)
    assert np.all(reg_flat == 0)

    # Changing future values after t=50 must NOT alter causal regime assignments at t <= 50
    trace_a = np.linspace(0.5, 1.5, 100)
    trace_b = trace_a.copy()
    trace_b[51:] = 10.0

    reg_a = discretize_price_causal(trace_a, n_contexts=3)
    reg_b = discretize_price_causal(trace_b, n_contexts=3)
    np.testing.assert_array_equal(reg_a[:51], reg_b[:51])


def test_sw_ucb_bwk_adapts_faster_than_ucb_bwk_under_regime_shift() -> None:
    """
    Simulate an LP-mixing regime shift at round 250 with target rate rho = 0.30.
    SWUCBBwK evicts Regime 1 within `window_size = 40` rounds and commits to Arm 2.
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