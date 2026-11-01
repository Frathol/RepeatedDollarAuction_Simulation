"""
test_nonstationarity.py

Unit tests for src/metrics/nonstationarity.py, verifying:
1. Stationary environment yields V1 = V2 = W1 = W2 = 0.0 and sigma_r = sigma_c = 1.0.
2. Exact analytical calculation of V1, V2, W1, W2, sigma_r, sigma_c on a 2-regime
   piecewise environment.
3. Context-conditioned residual metrics:
   - Purely exogenous cost non-stationarity (Alice + EC2 price regimes) is absorbed
     by context conditioning (residual W2 -> 0).
   - Endogenous opponent non-stationarity (Bob shifting inside the same price
     regime) remains in the residual metrics (residual W1 > 0).
4. Dynamic LP Oracle (OPT_{dyn-LP}) vs. Per-Round LP (OPT_{per-round}) and
   non-negativity of Part A (Budget Allocation Gap).
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.nonstationarity import (
    compute_dynamic_lp_benchmarks,
    compute_nonstationarity_metrics,
    compute_residual_nonstationarity_metrics,
)


def test_stationary_environment_metrics_are_zero_and_unity() -> None:
    """In a completely stationary environment, V=0, W=0, and sigma=1."""
    T, K = 100, 4
    mu = np.tile(np.array([0.60, 0.50, 0.75, 0.70]), (T, 1))
    c = np.tile(np.array([0.00, 0.10, 0.25, 0.40]), (T, 1))

    m = compute_nonstationarity_metrics(mu, c, include_null_arm=False)
    assert np.isclose(m.v1, 0.0)
    assert np.isclose(m.v2, 0.0)
    assert np.isclose(m.w1, 0.0)
    assert np.isclose(m.w2, 0.0)
    assert np.isclose(m.sigma_r, 1.0)
    assert np.isclose(m.sigma_c, 1.0)
    assert np.isclose(m.w_star_theoretical, float(T))


def test_single_switch_analytical_values() -> None:
    """
    Verify exact hand-calculated values for T = 4 rounds, 1 active arm:
    - Rounds 0, 1: mu = 0.80, c = 0.20
    - Rounds 2, 3: mu = 0.40, c = 0.60
    Then:
    - V1 = |0.40 - 0.80| = 0.40;  V2 = |0.60 - 0.20| = 0.40
    - mu_bar = 0.60, c_bar = 0.40
    - W1 = 4 * |0.80 - 0.60| = 0.80;  W2 = 4 * |0.20 - 0.40| = 0.80
    - sigma_r = max(0.80/0.60, 0.60/0.40) = 1.50
    - sigma_c = max(0.60/0.40, 0.40/0.20) = 2.00
    """
    mu = np.array([
        [0.60, 0.80],
        [0.60, 0.80],
        [0.60, 0.40],
        [0.60, 0.40],
    ])
    c = np.array([
        [0.00, 0.20],
        [0.00, 0.20],
        [0.00, 0.60],
        [0.00, 0.60],
    ])

    m = compute_nonstationarity_metrics(mu, c, include_null_arm=False)
    assert np.isclose(m.v1, 0.40)
    assert np.isclose(m.v2, 0.40)
    assert np.isclose(m.w1, 0.80)
    assert np.isclose(m.w2, 0.80)
    assert np.isclose(m.sigma_r, 1.50)
    assert np.isclose(m.sigma_c, 2.00)


def test_residual_metrics_separate_exogenous_from_endogenous() -> None:
    """
    Verify the core RQ2 diagnostic property:
    - Case A (Pure Exogenous Cost Shifts): Cost changes strictly with context
      regime `s_t in {0, 1}`. Conditioning on `s_t` absorbs 100% of W2
      (`residual_w2 == 0.0`).
    - Case B (Endogenous Opponent Shift inside Fixed Context): Even within the
      same price regime `s_t = 0`, Bob switches from Rational (mu = 0.75) to
      Escalating (mu = 0.35), leaving a large `residual_w1 > 0`.
    """
    T = 20
    regimes = np.array([0] * 10 + [1] * 10)

    # Case A: Alice + Exogenous 2-regime price multiplier
    mu_exo = np.tile([0.60, 0.75], (T, 1))
    c_exo = np.vstack([
        np.tile([0.00, 0.20], (10, 1)),  # Regime 0 (Cheap)
        np.tile([0.00, 0.50], (10, 1)),  # Regime 1 (Expensive)
    ])
    res_exo = compute_residual_nonstationarity_metrics(mu_exo, c_exo, regimes)
    assert res_exo["raw_w2"] > 0.0
    assert np.isclose(res_exo["residual_w2"], 0.0)
    assert np.isclose(res_exo["w2_absorbed_by_context_frac"], 1.0)

    # Case B: Bob escalates halfway through Regime 0 AND halfway through Regime 1
    mu_endo = np.vstack([
        np.tile([0.60, 0.75], (5, 1)),   # Regime 0, Bob Rational
        np.tile([0.60, 0.35], (5, 1)),   # Regime 0, Bob Escalating
        np.tile([0.60, 0.75], (5, 1)),   # Regime 1, Bob Rational
        np.tile([0.60, 0.35], (5, 1)),   # Regime 1, Bob Escalating
    ])
    res_endo = compute_residual_nonstationarity_metrics(mu_endo, c_exo, regimes)
    # Exogenous cost is still absorbed, but endogenous reward deviation W1 remains unabsorbed!
    assert np.isclose(res_endo["residual_w2"], 0.0)
    assert np.isclose(res_endo["residual_w1"], res_endo["raw_w1"])
    assert res_endo["residual_sigma_r"] > 1.3


def test_dynamic_lp_benchmarks_part_a_gap_is_positive_under_price_variation() -> None:
    """
    OPT_{dyn-LP} must strictly exceed OPT_{per-round} when costs vary across
    rounds, because the Dynamic Oracle saves budget in expensive rounds to spend
    in cheap rounds.
    """
    T = 10
    # 5 cheap rounds (c=0.40) and 5 expensive rounds (c=0.80); budget B = 2.0 (rho = 0.20)
    mu = np.tile([0.60, 0.80], (T, 1))
    c = np.vstack([
        np.tile([0.00, 0.40], (5, 1)),
        np.tile([0.00, 0.80], (5, 1)),
    ])
    bench = compute_dynamic_lp_benchmarks(mu, c, budget=2.0, null_arm=0)

    # Per-round LP (cap 0.20): 5 * (0.5*0.80 + 0.5*0.60) + 5 * (0.25*0.80 + 0.75*0.60) = 3.5 + 3.25 = 6.75
    # Dynamic LP (total B=2.0): plays Arm 1 100% in all 5 cheap rounds (cost 2.0) and Arm 0 in expensive = 5*0.80 + 5*0.60 = 7.00
    assert np.isclose(bench["opt_per_round"], 6.75)
    assert np.isclose(bench["opt_dyn_lp"], 7.00)
    assert np.isclose(bench["part_a_budget_allocation_gap"], 0.25)