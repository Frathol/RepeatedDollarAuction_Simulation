"""
nonstationarity.py

Quantitative environment non-stationarity measures and dynamic LP benchmarks
for Bandits with Knapsacks (BwK):

1. Local Total Variation (V_1, V_2) and Global Deviation from Horizon Mean
   (W_1, W_2) from Liu, Jiang, and Li (2022), "Non-stationary Bandits with
   Knapsacks" (NeurIPS 2022).
2. Approximate Stationarity Ratios (sigma_r, sigma_c) from Fikioris and Tardos
   (2023), "Approximately Stationary Bandits with Knapsacks" (COLT 2023).
3. Context-Conditioned Residual Non-Stationarity Metrics (isolating endogenous
   opponent non-stationarity after conditioning on exogenous EC2 price regimes).
4. Dynamic LP Oracle (OPT_{dyn-LP}) and Per-Round LP (OPT_{per-round})
   benchmarks and their regret decomposition (Part A: Budget Allocation Gap vs.
   Part B: Online Estimation Regret).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict
import numpy as np

from src.algorithms.bwk import solve_1d_knapsack_lp, solve_policy_knapsack_lp


@dataclass(frozen=True)
class NonStationarityMetrics:
    """Container for Liu et al. (2022) and Fikioris & Tardos (2023) metrics."""

    v1: float
    v2: float
    w1: float
    w2: float
    sigma_r: float
    sigma_c: float
    w_star_theoretical: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


def _select_arm_columns(
    mu_matrix: np.ndarray,
    c_matrix: np.ndarray,
    include_null_arm: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate `(T, K)` matrices and slice active arms `1..K-1` if requested."""
    mu = np.asarray(mu_matrix, dtype=np.float64)
    c = np.asarray(c_matrix, dtype=np.float64)
    if mu.ndim != 2 or c.ndim != 2 or mu.shape != c.shape:
        raise ValueError(
            f"mu_matrix and c_matrix must have matching 2D shape (T, K), "
            f"got {mu.shape} and {c.shape}"
        )
    if mu.shape[0] < 1 or mu.shape[1] < 1:
        raise ValueError("mu_matrix and c_matrix must be non-empty")

    if not include_null_arm and mu.shape[1] > 1:
        return mu[:, 1:], c[:, 1:]
    return mu, c


def compute_nonstationarity_metrics(
    mu_matrix: np.ndarray,
    c_matrix: np.ndarray,
    include_null_arm: bool = False,
    eps: float = 1e-12,
) -> NonStationarityMetrics:
    """
    Compute V_1, V_2, W_1, W_2 (Liu et al., 2022) and sigma_r, sigma_c
    (Fikioris & Tardos, 2023) from per-round expected reward and cost matrices.

    Parameters
    ----------
    mu_matrix : numpy.ndarray
        2D array of shape `(T, K)` where entry `(t, theta)` is the expected
        normalized reward `mu_t(theta)` in `[0, 1]`.
    c_matrix : numpy.ndarray
        2D array of shape `(T, K)` where entry `(t, theta)` is the expected
        normalized cost `c_t(theta)` in `[0, 1]`.
    include_null_arm : bool, default=False
        If False (default) and `K > 1`, evaluates non-stationarity over the
        active bidding arms `theta in {1, ..., K-1}` (since arm 0 has
        identically zero cost, which would otherwise trigger 0/0 in sigma_c).
    eps : float, default=1e-12
        Numerical floor for strictly positive denominator checks in sigma_r/c.

    Returns
    -------
    NonStationarityMetrics
        Dataclass containing `(v1, v2, w1, w2, sigma_r, sigma_c, w_star_theoretical)`.
    """
    mu, c = _select_arm_columns(
        mu_matrix, c_matrix, include_null_arm=include_null_arm
    )
    T, K_eval = mu.shape

    # 1. Local Total Variation (V_1, V_2)
    if T > 1:
        v1 = float(np.sum(np.max(np.abs(np.diff(mu, axis=0)), axis=1)))
        v2 = float(np.sum(np.max(np.abs(np.diff(c, axis=0)), axis=1)))
    else:
        v1 = 0.0
        v2 = 0.0

    # 2. Global Deviation from Horizon Average (W_1, W_2)
    mu_bar = np.mean(mu, axis=0, keepdims=True)  # (1, K_eval)
    c_bar = np.mean(c, axis=0, keepdims=True)    # (1, K_eval)

    w1 = float(np.sum(np.max(np.abs(mu - mu_bar), axis=1)))
    w2 = float(np.sum(np.max(np.abs(c - c_bar), axis=1)))

    # 3. Approximate Stationarity Ratios (sigma_r, sigma_c)
    sigma_r = _compute_max_ratio(mu, mu_bar, eps=eps)
    sigma_c = _compute_max_ratio(c, c_bar, eps=eps)

    # 4. Theoretical sliding-window guide w* = K^{1/3} T^{2/3} V^{-2/3}
    v_total = v1 + v2
    if v_total <= eps:
        w_star = float(T)
    else:
        w_star = float(
            (K_eval ** (1.0 / 3.0))
            * (T ** (2.0 / 3.0))
            * (v_total ** (-2.0 / 3.0))
        )
        w_star = float(np.clip(w_star, 1.0, float(T)))

    return NonStationarityMetrics(
        v1=v1,
        v2=v2,
        w1=w1,
        w2=w2,
        sigma_r=sigma_r,
        sigma_c=sigma_c,
        w_star_theoretical=w_star,
    )


def _compute_max_ratio(
    mat: np.ndarray,
    mat_bar: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute `max_{t, theta} max(mat[t, theta] / mat_bar[theta], mat_bar[theta] / mat[t, theta])`
    over all arms with positive horizon mean `mat_bar[theta] > eps`.
    """
    active_cols = np.flatnonzero(mat_bar.ravel() > eps)
    if active_cols.size == 0:
        return 1.0

    m_sub = np.maximum(mat[:, active_cols], eps)
    b_sub = np.maximum(mat_bar[:, active_cols], eps)
    ratios = np.maximum(m_sub / b_sub, b_sub / m_sub)
    return float(np.max(ratios))


def compute_residual_nonstationarity_metrics(
    mu_matrix: np.ndarray,
    c_matrix: np.ndarray,
    regimes: np.ndarray,
    include_null_arm: bool = False,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Decompose non-stationarity into Raw (unconditioned) vs. Context-Conditioned
    Residual metrics given discrete exogenous context regimes `s_t`.

    For each regime `s`, replaces the global mean `(mu_bar, c_bar)` with the
    within-regime conditional mean `(mu_bar_s, c_bar_s)` to measure how much
    deviation (W1_residual, W2_residual), within-regime sequential variation
    (V1_residual, V2_residual), and fluctuation ratio (sigma_r_residual,
    sigma_c_residual) remain unexplained by the exogenous price context.

    Parameters
    ----------
    mu_matrix : numpy.ndarray
        2D array of shape `(T, K)` with expected rewards.
    c_matrix : numpy.ndarray
        2D array of shape `(T, K)` with expected costs.
    regimes : numpy.ndarray
        1D integer array of shape `(T,)` with context regime IDs `s_t`.
    include_null_arm : bool, default=False
        Whether to include arm 0 in the evaluation.
    eps : float, default=1e-12
        Numerical floor for ratio calculations.

    Returns
    -------
    dict[str, float]
        Dictionary containing raw metrics, residual metrics, and the fraction of
        cost/reward deviation absorbed by the context.
    """
    raw = compute_nonstationarity_metrics(
        mu_matrix, c_matrix, include_null_arm=include_null_arm, eps=eps
    )
    mu, c = _select_arm_columns(
        mu_matrix, c_matrix, include_null_arm=include_null_arm
    )
    reg = np.asarray(regimes, dtype=np.int64)
    if reg.ndim != 1 or reg.shape[0] != mu.shape[0]:
        raise ValueError("regimes must be a 1D array of length T")

    unique_regimes = np.unique(reg)
    v1_res = 0.0
    v2_res = 0.0
    w1_res = 0.0
    w2_res = 0.0
    sigma_r_res = 1.0
    sigma_c_res = 1.0

    for s in unique_regimes:
        idx = np.flatnonzero(reg == s)
        if idx.size == 0:
            continue
        sub_metrics = compute_nonstationarity_metrics(
            mu[idx], c[idx], include_null_arm=True, eps=eps
        )
        v1_res += sub_metrics.v1
        v2_res += sub_metrics.v2
        w1_res += sub_metrics.w1
        w2_res += sub_metrics.w2
        sigma_r_res = max(sigma_r_res, sub_metrics.sigma_r)
        sigma_c_res = max(sigma_c_res, sub_metrics.sigma_c)

    w2_absorbed_frac = (
        float(np.clip(1.0 - (w2_res / raw.w2), 0.0, 1.0))
        if raw.w2 > eps
        else 1.0
    )
    w1_absorbed_frac = (
        float(np.clip(1.0 - (w1_res / raw.w1), 0.0, 1.0))
        if raw.w1 > eps
        else 1.0
    )

    return {
        "raw_v1": raw.v1,
        "raw_v2": raw.v2,
        "raw_w1": raw.w1,
        "raw_w2": raw.w2,
        "raw_sigma_r": raw.sigma_r,
        "raw_sigma_c": raw.sigma_c,
        "residual_v1": float(v1_res),
        "residual_v2": float(v2_res),
        "residual_w1": float(w1_res),
        "residual_w2": float(w2_res),
        "residual_sigma_r": float(sigma_r_res),
        "residual_sigma_c": float(sigma_c_res),
        "w1_absorbed_by_context_frac": w1_absorbed_frac,
        "w2_absorbed_by_context_frac": w2_absorbed_frac,
    }


def compute_dynamic_lp_benchmarks(
    mu_matrix: np.ndarray,
    c_matrix: np.ndarray,
    budget: float,
    null_arm: int = 0,
) -> Dict[str, np.ndarray | float]:
    """
    Compute the Dynamic LP Oracle (`OPT_{dyn-LP}`) and Per-Round LP
    (`OPT_{per-round}`) benchmarks from Liu, Jiang, & Li (2022) (Slide 26 /
    Eq. 2.16-2.17):

        OPT_{dyn-LP}    = max_{x_1..x_T in Delta_K} sum_{t=1}^T mu_t^T x_t
                          s.t. sum_{t=1}^T c_t^T x_t <= B
        OPT_{per-round} = sum_{t=1}^T max_{x_t in Delta_K} { mu_t^T x_t : c_t^T x_t <= B/T }

    Parameters
    ----------
    mu_matrix : numpy.ndarray
        2D array of shape `(T, K)` with expected rewards `mu_t(theta)` (including
        the null arm column at `null_arm`).
    c_matrix : numpy.ndarray
        2D array of shape `(T, K)` with expected costs `c_t(theta)`.
    budget : float
        Total global budget `B > 0`.
    null_arm : int, default=0
        Index of the zero-cost null arm.

    Returns
    -------
    dict
        - `"opt_dyn_lp"`: float, total reward of Dynamic LP Oracle.
        - `"opt_per_round"`: float, total reward of Per-Round LP Oracle.
        - `"part_a_budget_allocation_gap"`: float, `OPT_{dyn-LP} - OPT_{per-round} >= 0`.
        - `"dyn_lp_per_round_rewards"`: 1D array of shape `(T,)` with expected
          reward achieved by the Dynamic LP Oracle at each round `t`.
        - `"per_round_lp_rewards"`: 1D array of shape `(T,)` with expected
          reward achieved by the Per-Round LP Oracle at each round `t`.
    """
    mu = np.asarray(mu_matrix, dtype=np.float64)
    c = np.asarray(c_matrix, dtype=np.float64)
    if mu.ndim != 2 or c.shape != mu.shape:
        raise ValueError("mu_matrix and c_matrix must have matching 2D shape (T, K)")
    T, K = mu.shape
    if budget <= 0.0:
        raise ValueError("budget must be positive")

    rho = float(budget) / float(T)

    # 1. Per-Round LP benchmark: solve 1D LP independently at every round t with cap rho
    per_round_lp_rewards = np.zeros(T, dtype=np.float64)
    for t in range(T):
        x_t = solve_1d_knapsack_lp(mu[t], c[t], rho=rho, null_arm=null_arm)
        per_round_lp_rewards[t] = float(np.dot(x_t, mu[t]))
    opt_per_round = float(np.sum(per_round_lp_rewards))

    # 2. Dynamic LP Oracle: Continuous Multiple-Choice Knapsack across all T rounds
    uniform_round_weights = np.full(T, 1.0 / float(T), dtype=np.float64)
    dyn_x = solve_policy_knapsack_lp(
        ucb_rewards_2d=mu,
        lcb_costs_2d=c,
        context_probs=uniform_round_weights,
        rho=rho,
        null_arm=null_arm,
    )
    dyn_lp_per_round_rewards = np.sum(dyn_x * mu, axis=1)
    opt_dyn_lp = float(np.sum(dyn_lp_per_round_rewards))

    # By LP feasibility, OPT_{dyn-LP} >= OPT_{per-round}
    part_a_gap = max(0.0, opt_dyn_lp - opt_per_round)

    return {
        "opt_dyn_lp": opt_dyn_lp,
        "opt_per_round": opt_per_round,
        "part_a_budget_allocation_gap": part_a_gap,
        "dyn_lp_per_round_rewards": dyn_lp_per_round_rewards,
        "per_round_lp_rewards": per_round_lp_rewards,
    }