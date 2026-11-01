"""
bwk.py

Implementations of Bandits with Knapsacks (BwK) and Contextual Bandits with
Knapsacks (CBwK) algorithms with a single global budget constraint (d = 1)
and a null arm (arm 0):

1. SWUCBBwK         : Sliding-Window UCB for Bandits with Knapsacks, introduced
                      by Liu, Jiang, and Li (2022), "Non-stationary Bandits with
                      Knapsacks" (Algorithm 1).
2. UCBBwK           : Stationary UCB for Bandits with Knapsacks (Agrawal and
                      Devanur, 2014), implemented as the full-history special
                      case of SWUCBBwK with window_size = None (i.e. w = T).
3. ContextualUCBBwK : Regime-discretized Contextual UCB for Bandits with
                      Knapsacks (Badanidiyuru, Langford, & Slivkins, 2014;
                      Slivkins, 2019 Ch. 8), supporting both coupled policy-level
                      LP across price regimes and per-context LP ablation.

All classes strictly adhere to the BanditAlgorithm interface defined in
src/algorithms/base.py. Per-round observed resource consumption is read from
`info["cost"]` (and optional context index from `info["context"]`) in update().
"""

from __future__ import annotations

from collections import deque
import numpy as np

from src.algorithms.base import BanditAlgorithm


def solve_1d_knapsack_lp(
    ucb_rewards: np.ndarray,
    lcb_costs: np.ndarray,
    rho: float,
    null_arm: int = 0,
) -> np.ndarray:
    """
    Solve the single-constraint per-round BwK Linear Program over the
    probability simplex:

        maximize    sum_{i=0}^{K-1} x_i * ucb_rewards[i]
        subject to  sum_{i=0}^{K-1} x_i * lcb_costs[i] <= rho
                    sum_{i=0}^{K-1} x_i == 1,  x_i >= 0

    By the Fundamental Theorem of Linear Programming, an optimal basic
    feasible solution on the simplex with 1 linear inequality constraint
    has support of size at most 2 (either a single affordable arm or a convex
    combination of one arm with cost <= rho and one arm with cost > rho).

    Parameters
    ----------
    ucb_rewards : numpy.ndarray
        1D array of shape `(n_arms,)` with optimistic reward estimates.
    lcb_costs : numpy.ndarray
        1D array of shape `(n_arms,)` with optimistic (lower bound) cost
        estimates.
    rho : float
        Target per-round budget rate (typically B / T).
    null_arm : int, default=0
        Index of the null arm (pass strategy) guaranteed to have zero cost.

    Returns
    -------
    numpy.ndarray
        Optimal probability distribution `x` of shape `(n_arms,)` summing to 1.
    """
    n_arms = ucb_rewards.shape[0]
    probs = np.zeros(n_arms, dtype=np.float64)

    if rho <= 0.0:
        probs[null_arm] = 1.0
        return probs

    tol = 1e-12
    affordable_mask = lcb_costs <= (rho + tol)
    if not np.any(affordable_mask):
        probs[null_arm] = 1.0
        return probs

    affordable_indices = np.flatnonzero(affordable_mask)
    expensive_indices = np.flatnonzero(~affordable_mask)

    # Candidate 1: Best single affordable arm (support size = 1)
    best_single_idx = int(affordable_indices[0])
    best_obj = -np.inf
    for i in affordable_indices:
        # Tie-break in favor of lower cost if rewards are equal
        if ucb_rewards[i] > best_obj + tol or (
            abs(ucb_rewards[i] - best_obj) <= tol
            and lcb_costs[i] < lcb_costs[best_single_idx]
        ):
            best_obj = float(ucb_rewards[i])
            best_single_idx = int(i)

    best_pair: tuple[int, int, float] | None = None  # (cheap_i, exp_k, alpha_k)

    # Candidate 2: Convex combination of affordable arm i and expensive arm k
    if expensive_indices.size > 0:
        for i in affordable_indices:
            u_i = float(ucb_rewards[i])
            c_i = float(lcb_costs[i])
            if abs(rho - c_i) <= tol:
                continue  # Cannot put positive weight on any expensive arm k
            for k in expensive_indices:
                u_k = float(ucb_rewards[k])
                if u_k <= u_i + tol:
                    continue  # Mixing with k cannot improve upon pure arm i
                c_k = float(lcb_costs[k])
                alpha = (rho - c_i) / (c_k - c_i)
                alpha = float(np.clip(alpha, 0.0, 1.0))
                mix_obj = (1.0 - alpha) * u_i + alpha * u_k
                if mix_obj > best_obj + tol:
                    best_obj = mix_obj
                    best_pair = (int(i), int(k), alpha)

    if best_pair is None:
        probs[best_single_idx] = 1.0
    else:
        i_cheap, k_exp, alpha = best_pair
        probs[i_cheap] = 1.0 - alpha
        probs[k_exp] = alpha

    # Guard against tiny floating-point drift
    probs = np.maximum(probs, 0.0)
    probs /= np.sum(probs)
    return probs


def _upper_convex_hull_arms(
    rewards: np.ndarray,
    costs: np.ndarray,
    tol: float = 1e-12,
) -> list[int]:
    """
    Extract the sequence of arm indices forming the strictly increasing upper
    convex hull (Pareto frontier) in the (cost, reward) plane.

    Starts at the zero/minimum-cost arm with highest reward, then iteratively
    picks the next arm with strictly higher cost and reward that maximizes the
    marginal efficiency slope (delta_reward / delta_cost).
    """
    n_arms = rewards.shape[0]
    min_c = float(np.min(costs))
    zero_c_mask = np.abs(costs - min_c) <= tol
    zero_c_candidates = np.flatnonzero(zero_c_mask)
    # Pick the minimum-cost arm with the highest reward (tie-break lowest index)
    best_start = int(
        zero_c_candidates[np.argmax(rewards[zero_c_candidates])]
    )

    hull = [best_start]
    curr = best_start
    while True:
        c_curr = float(costs[curr])
        u_curr = float(rewards[curr])
        best_next = -1
        best_slope = tol
        best_dc = -np.inf

        for k in range(n_arms):
            dc = float(costs[k]) - c_curr
            du = float(rewards[k]) - u_curr
            if dc > tol and du > tol:
                slope = du / dc
                # If tied in slope, pick the farthest point to avoid redundant vertices
                if slope > best_slope + tol or (
                    abs(slope - best_slope) <= tol and dc > best_dc
                ):
                    best_slope = slope
                    best_next = int(k)
                    best_dc = dc

        if best_next == -1:
            break
        hull.append(best_next)
        curr = best_next

    return hull


def solve_policy_knapsack_lp(
    ucb_rewards_2d: np.ndarray,
    lcb_costs_2d: np.ndarray,
    context_probs: np.ndarray,
    rho: float,
    null_arm: int = 0,
) -> np.ndarray:
    """
    Solve the coupled policy-level Contextual BwK Linear Program (Continuous
    Multiple-Choice Knapsack Problem) across M context regimes:

        maximize    sum_{s=0}^{M-1} p_s * sum_{i=0}^{K-1} x_{s,i} * ucb_rewards_2d[s, i]
        subject to  sum_{s=0}^{M-1} p_s * sum_{i=0}^{K-1} x_{s,i} * lcb_costs_2d[s, i] <= rho
                    sum_{i=0}^{K-1} x_{s,i} == 1,  x_{s,i} >= 0   for all s

    Allows budget pacing across regimes (e.g., saving budget in high-price
    regimes so the agent can spend > rho in low-price regimes).

    Parameters
    ----------
    ucb_rewards_2d : numpy.ndarray
        2D array of shape `(n_contexts, n_arms)` with optimistic rewards.
    lcb_costs_2d : numpy.ndarray
        2D array of shape `(n_contexts, n_arms)` with optimistic lower-bound costs.
    context_probs : numpy.ndarray
        1D array of shape `(n_contexts,)` with non-negative regime probabilities
        summing to 1.
    rho : float
        Target average per-round budget rate (B / T).
    null_arm : int, default=0
        Index of the zero-cost pass arm.

    Returns
    -------
    numpy.ndarray
        2D array `probs` of shape `(n_contexts, n_arms)` where each row `probs[s]`
        is the optimal probability distribution for regime `s`.
    """
    ucb_2d = np.asarray(ucb_rewards_2d, dtype=np.float64)
    lcb_2d = np.asarray(lcb_costs_2d, dtype=np.float64)
    p_s = np.asarray(context_probs, dtype=np.float64)

    if ucb_2d.ndim != 2 or lcb_2d.shape != ucb_2d.shape:
        raise ValueError("ucb_rewards_2d and lcb_costs_2d must have matching 2D shape (M, K)")
    n_contexts, n_arms = ucb_2d.shape
    if p_s.shape != (n_contexts,):
        raise ValueError("context_probs must have shape (n_contexts,)")

    probs = np.zeros((n_contexts, n_arms), dtype=np.float64)
    if rho <= 0.0:
        probs[:, null_arm] = 1.0
        return probs

    p_sum = float(np.sum(p_s))
    if p_sum <= 0.0:
        p_s = np.full(n_contexts, 1.0 / n_contexts, dtype=np.float64)
    else:
        p_s = p_s / p_sum

    tol = 1e-12
    # Step 1: Initialize each context s at its cheapest upper-hull vertex
    current_arm = np.zeros(n_contexts, dtype=np.int64)
    segments: list[tuple[float, int, int, int, float]] = []
    base_cost = 0.0

    for s in range(n_contexts):
        hull = _upper_convex_hull_arms(ucb_2d[s], lcb_2d[s], tol=tol)
        current_arm[s] = hull[0]
        probs[s, hull[0]] = 1.0
        base_cost += float(p_s[s] * lcb_2d[s, hull[0]])

        if p_s[s] > tol:
            for j in range(len(hull) - 1):
                i_from = hull[j]
                i_to = hull[j + 1]
                dc = float(lcb_2d[s, i_to] - lcb_2d[s, i_from])
                du = float(ucb_2d[s, i_to] - ucb_2d[s, i_from])
                slope = du / dc
                weighted_dc = float(p_s[s]) * dc
                segments.append((slope, s, i_from, i_to, weighted_dc))

    rem_rho = float(rho) - base_cost
    if rem_rho <= tol:
        return probs

    # Step 2: Sort all upgrade segments globally by decreasing efficiency slope
    # Tie-break by context index s so segments within the same context stay ordered
    segments.sort(key=lambda item: (-item[0], item[1]))

    for slope, s, i_from, i_to, weighted_dc in segments:
        if rem_rho <= tol:
            break
        if rem_rho >= weighted_dc - tol:
            # Full upgrade in context s from i_from to i_to
            probs[s, i_from] = 0.0
            probs[s, i_to] = 1.0
            current_arm[s] = i_to
            rem_rho -= weighted_dc
        else:
            # Partial upgrade (convex combination on the boundary segment)
            alpha = float(np.clip(rem_rho / weighted_dc, 0.0, 1.0))
            probs[s, i_from] = 1.0 - alpha
            probs[s, i_to] = alpha
            rem_rho = 0.0
            break

    # Normalize each row against floating-point drift
    probs = np.maximum(probs, 0.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs /= np.where(row_sums > 0.0, row_sums, 1.0)
    return probs


def discretize_price_causal(
    price_trace: np.ndarray,
    n_contexts: int = 3,
    min_history: int = 30,
) -> np.ndarray:
    """
    Causally discretize an exogenous price multiplier series `m_t` into
    `n_contexts` quantile regimes in `{0, 1, ..., n_contexts - 1}` without
    future leakage.

    At round t, quantile thresholds are computed strictly from the prefix
    `price_trace[:t+1]`. If the prefix has near-zero variance (e.g., flat
    control series), all rounds map cleanly to regime `0`.

    Parameters
    ----------
    price_trace : numpy.ndarray
        1D array of length `T` containing positive price multipliers `m_t`.
    n_contexts : int, default=3
        Number of discrete price regimes M.
    min_history : int, default=30
        Minimum prefix length before empirical quantiles are updated; for
        `t < min_history`, thresholds are computed on `price_trace[:max(1, t+1)]`.

    Returns
    -------
    numpy.ndarray
        1D integer array `regimes` of shape `(T,)` with values in `[0, n_contexts)`.
    """
    trace = np.asarray(price_trace, dtype=np.float64)
    if trace.ndim != 1 or trace.size == 0:
        raise ValueError("price_trace must be a non-empty 1D array")
    if n_contexts < 1:
        raise ValueError("n_contexts must be >= 1")

    T = trace.size
    regimes = np.zeros(T, dtype=np.int64)
    if n_contexts == 1:
        return regimes

    quantiles = np.linspace(0.0, 1.0, n_contexts + 1)[1:-1]
    tol = 1e-12

    for t in range(T):
        prefix = trace[: t + 1]
        if float(np.max(prefix) - np.min(prefix)) <= tol:
            regimes[t] = 0
        else:
            edges = np.quantile(prefix, quantiles)
            # Only use strictly distinct edges if some quantiles collapse
            regimes[t] = int(np.searchsorted(edges, trace[t], side="right"))
            regimes[t] = min(regimes[t], n_contexts - 1)

    return regimes


class SWUCBBwK(BanditAlgorithm):
    """
    Sliding-Window UCB for Bandits with Knapsacks (Liu, Jiang, & Li, 2022).

    Maintains sliding-window empirical estimates over the last `window_size`
    rounds. At each round t, solves the optimistic Linear Program using UCB for
    rewards and LCB for resource consumption costs, then samples an arm from the
    resulting distribution x_t. If remaining budget is exhausted
    (`remaining_budget <= 0`), falls back to the null arm (`null_arm = 0`).

    Parameters
    ----------
    n_arms : int
        Number of arms in the strategy set S0 (including the null arm at 0).
    budget : float
        Total global budget B across the horizon T.
    horizon : int
        Total time horizon T (used to compute target rate rho = B / T).
    window_size : int or None, default=500
        Sliding window length w in rounds. If None or >= horizon, uses full
        history (equivalent to stationary UCB-BwK).
    conf_scale : float, default=0.5
        Confidence radius scaling constant kappa in
        rad_t(a) = conf_scale * sqrt(ln(min(t, w) + 1) / N_{t,w}(a)).
    null_arm : int, default=0
        Index of the zero-cost pass strategy (theta = 0).
    null_arm_reward : float, default=0.0
        Known deterministic normalized reward of the null arm in [0, 1].
        Defaults to 0.0 for standard BwK; set to `b / (b + s)` (e.g., 0.6)
        when rewards are Min-Max shifted from [-b, s] to [0, 1].
    rng : numpy.random.Generator, optional
        Random generator for sampling from the LP distribution x_t.
    """

    def __init__(
        self,
        n_arms: int,
        budget: float = 2500.0,
        horizon: int = 10000,
        window_size: int | None = 500,
        conf_scale: float = 0.5,
        null_arm: int = 0,
        null_arm_reward: float = 0.0,
        rng=None,
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        if budget <= 0.0:
            raise ValueError("budget must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if window_size is not None and window_size <= 0:
            raise ValueError("window_size must be a positive integer or None")
        if conf_scale < 0.0:
            raise ValueError("conf_scale must be non-negative")
        if not (0 <= null_arm < self.n_arms):
            raise ValueError(f"null_arm must be in [0, {self.n_arms})")
        if not (0.0 <= null_arm_reward <= 1.0):
            raise ValueError("null_arm_reward must lie in [0, 1]")

        self.initial_budget: float = float(budget)
        self.remaining_budget: float = float(budget)
        self.horizon: int = int(horizon)
        self.rho: float = self.initial_budget / float(self.horizon)
        self.window_size: int | None = (
            int(window_size) if window_size is not None else None
        )
        self.conf_scale: float = float(conf_scale)
        self.null_arm: int = int(null_arm)
        self.null_arm_reward: float = float(null_arm_reward)

        # Sliding-window history buffer storing tuples (arm, reward, cost)
        self._history: deque[tuple[int, float, float]] = deque()
        self.win_counts: np.ndarray = np.zeros(self.n_arms, dtype=np.int64)
        self.win_reward_sums: np.ndarray = np.zeros(
            self.n_arms, dtype=np.float64
        )
        self.win_cost_sums: np.ndarray = np.zeros(
            self.n_arms, dtype=np.float64
        )
        self.last_lp_distribution: np.ndarray = np.full(
            self.n_arms, 1.0 / self.n_arms, dtype=np.float64
        )

    def get_optimistic_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute UCB for rewards and LCB for costs over the active window.

        Returns
        -------
        tuple[numpy.ndarray, numpy.ndarray]
            `(ucb_rewards, lcb_costs)`, each of shape `(n_arms,)` clipped to
            [0, 1]. The null arm always has exact `(null_arm_reward, 0.0)` bounds.
        """
        ucb_rewards = np.ones(self.n_arms, dtype=np.float64)
        lcb_costs = np.zeros(self.n_arms, dtype=np.float64)

        pulled_mask = self.win_counts > 0
        if np.any(pulled_mask):
            eff_time = (
                min(self.t, self.window_size)
                if self.window_size is not None
                else max(1, self.t)
            )
            log_term = np.log(max(2.0, float(eff_time) + 1.0))
            rad = self.conf_scale * np.sqrt(
                log_term / self.win_counts[pulled_mask]
            )

            mean_rew = (
                self.win_reward_sums[pulled_mask]
                / self.win_counts[pulled_mask]
            )
            mean_cost = (
                self.win_cost_sums[pulled_mask] / self.win_counts[pulled_mask]
            )

            ucb_rewards[pulled_mask] = np.clip(mean_rew + rad, 0.0, 1.0)
            lcb_costs[pulled_mask] = np.clip(mean_cost - rad, 0.0, 1.0)

        # The null arm (theta = 0) has known deterministic reward & zero cost
        ucb_rewards[self.null_arm] = self.null_arm_reward
        lcb_costs[self.null_arm] = 0.0
        return ucb_rewards, lcb_costs

    def select_arm(self) -> int:
        """
        Choose an arm to play in the current round by solving the per-round LP
        and sampling from the resulting probability distribution x_t.

        Returns
        -------
        int
            Index into the strategy set S0, in range [0, n_arms).
        """
        if self.remaining_budget <= 1e-9:
            self.last_lp_distribution = np.zeros(self.n_arms, dtype=np.float64)
            self.last_lp_distribution[self.null_arm] = 1.0
            return self.null_arm

        ucb_rewards, lcb_costs = self.get_optimistic_bounds()
        probs = solve_1d_knapsack_lp(
            ucb_rewards=ucb_rewards,
            lcb_costs=lcb_costs,
            rho=self.rho,
            null_arm=self.null_arm,
        )
        self.last_lp_distribution = probs
        return int(self.rng.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float, info: dict | None = None) -> None:
        """
        Update sliding-window statistics and remaining global budget.

        Parameters
        ----------
        arm : int
            The arm played this round, in [0, n_arms).
        reward : float
            The observed normalized reward in [0, 1].
        info : dict, optional
            Dictionary containing `"cost"` (normalized cost in [0, 1] consumed
            this round). If `info` is None or `"cost"` is absent, defaults to
            0.0.
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        if not (-1e-9 <= reward <= 1.0 + 1e-9):
            raise ValueError(f"reward must lie in [0, 1], got {reward}")

        raw_cost = 0.0
        if info is not None and "cost" in info:
            raw_cost = float(info["cost"])
            if not (-1e-9 <= raw_cost <= 1.0 + 1e-9):
                raise ValueError(f"cost must lie in [0, 1], got {raw_cost}")

        clipped_reward = float(np.clip(reward, 0.0, 1.0))
        clipped_cost = (
            0.0 if arm == self.null_arm else float(np.clip(raw_cost, 0.0, 1.0))
        )

        self.t += 1
        self.remaining_budget = max(0.0, self.remaining_budget - clipped_cost)

        # Push new observation into the active window
        self._history.append((arm, clipped_reward, clipped_cost))
        self.win_counts[arm] += 1
        self.win_reward_sums[arm] += clipped_reward
        self.win_cost_sums[arm] += clipped_cost

        # Evict oldest observation if sliding window capacity is exceeded
        if (
            self.window_size is not None
            and len(self._history) > self.window_size
        ):
            old_arm, old_rew, old_cost = self._history.popleft()
            self.win_counts[old_arm] -= 1
            self.win_reward_sums[old_arm] = max(
                0.0, self.win_reward_sums[old_arm] - old_rew
            )
            self.win_cost_sums[old_arm] = max(
                0.0, self.win_cost_sums[old_arm] - old_cost
            )

    def reset(self) -> None:
        """Reset budget, sliding-window buffer, and round counter."""
        self.t = 0
        self.remaining_budget = self.initial_budget
        self._history.clear()
        self.win_counts.fill(0)
        self.win_reward_sums.fill(0.0)
        self.win_cost_sums.fill(0.0)
        self.last_lp_distribution.fill(1.0 / self.n_arms)


class UCBBwK(SWUCBBwK):
    """
    Stationary UCB for Bandits with Knapsacks (Agrawal and Devanur, 2014).

    Equivalent to SWUCBBwK with an unbounded window (`window_size = None`),
    retaining all observations from round t = 1 through T.
    """

    def __init__(
        self,
        n_arms: int,
        budget: float = 2500.0,
        horizon: int = 10000,
        conf_scale: float = 0.5,
        null_arm: int = 0,
        null_arm_reward: float = 0.0,
        rng=None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            budget=budget,
            horizon=horizon,
            window_size=None,
            conf_scale=conf_scale,
            null_arm=null_arm,
            null_arm_reward=null_arm_reward,
            rng=rng,
        )


class ContextualUCBBwK(BanditAlgorithm):
    """
    Contextual UCB for Bandits with Knapsacks (CTX-UCB-BwK).

    Conditions reward and cost estimates on a discrete exogenous context
    regime `s_t in {0, ..., n_contexts - 1}` (such as causally binned Amazon
    EC2 spot price regimes). Supports two LP formulations:
    - `couple_regimes=True` (default): Solves the policy-level LP
      (`solve_policy_knapsack_lp`) weighted by empirical regime arrival
      frequencies `p_hat(s)`, allowing the learner to save budget in expensive
      regimes and spend more aggressively in cheap regimes (Badanidiyuru et al.,
      2014; Agrawal et al., 2016).
    - `couple_regimes=False`: Solves `solve_1d_knapsack_lp` independently
      within the active regime `s_t` under per-round cap `rho = B / T`.

    Parameters
    ----------
    n_arms : int
        Number of arms in the strategy set S0 (including the null arm at 0).
    n_contexts : int, default=3
        Number of discrete exogenous context regimes M.
    budget : float, default=2500.0
        Total global budget B across the horizon T.
    horizon : int, default=10000
        Total time horizon T (target average rate rho = B / T).
    conf_scale : float, default=0.5
        Confidence radius scaling constant kappa.
    null_arm : int, default=0
        Index of the zero-cost pass strategy (theta = 0).
    null_arm_reward : float, default=0.0
        Deterministic normalized reward of the null arm in [0, 1] (`b / (b + s)`
        when using Min-Max normalized Dollar Auction rewards).
    couple_regimes : bool, default=True
        If True, couples budget allocation across regimes via policy LP; if
        False, enforces `<= rho` separately within each regime.
    rng : numpy.random.Generator, optional
        Random generator for sampling from the LP distribution.
    """

    def __init__(
        self,
        n_arms: int,
        n_contexts: int = 3,
        budget: float = 2500.0,
        horizon: int = 10000,
        conf_scale: float = 0.5,
        null_arm: int = 0,
        null_arm_reward: float = 0.0,
        couple_regimes: bool = True,
        rng=None,
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        if n_contexts < 1:
            raise ValueError("n_contexts must be a positive integer")
        if budget <= 0.0:
            raise ValueError("budget must be positive")
        if horizon <= 0:
            raise ValueError("horizon must be a positive integer")
        if conf_scale < 0.0:
            raise ValueError("conf_scale must be non-negative")
        if not (0 <= null_arm < self.n_arms):
            raise ValueError(f"null_arm must be in [0, {self.n_arms})")
        if not (0.0 <= null_arm_reward <= 1.0):
            raise ValueError("null_arm_reward must lie in [0, 1]")

        self.n_contexts: int = int(n_contexts)
        self.initial_budget: float = float(budget)
        self.remaining_budget: float = float(budget)
        self.horizon: int = int(horizon)
        self.rho: float = self.initial_budget / float(self.horizon)
        self.conf_scale: float = float(conf_scale)
        self.null_arm: int = int(null_arm)
        self.null_arm_reward: float = float(null_arm_reward)
        self.couple_regimes: bool = bool(couple_regimes)

        self.current_context: int = 0
        self.context_visits: np.ndarray = np.zeros(
            self.n_contexts, dtype=np.int64
        )
        self.counts: np.ndarray = np.zeros(
            (self.n_contexts, self.n_arms), dtype=np.int64
        )
        self.reward_sums: np.ndarray = np.zeros(
            (self.n_contexts, self.n_arms), dtype=np.float64
        )
        self.cost_sums: np.ndarray = np.zeros(
            (self.n_contexts, self.n_arms), dtype=np.float64
        )
        self.last_lp_distribution: np.ndarray = np.full(
            self.n_arms, 1.0 / self.n_arms, dtype=np.float64
        )

    def set_context(self, context: int) -> None:
        """Set the observable exogenous context regime `s_t` prior to `select_arm()`."""
        if not (0 <= int(context) < self.n_contexts):
            raise ValueError(
                f"context must be in [0, {self.n_contexts}), got {context}"
            )
        self.current_context = int(context)

    def get_optimistic_bounds_for_context(
        self, context: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute UCB reward and LCB cost vectors for a single context regime `s`."""
        s = int(context)
        ucb_rewards = np.ones(self.n_arms, dtype=np.float64)
        lcb_costs = np.zeros(self.n_arms, dtype=np.float64)

        pulled_mask = self.counts[s] > 0
        if np.any(pulled_mask):
            eff_time = max(1, int(self.context_visits[s]))
            log_term = np.log(max(2.0, float(eff_time) + 1.0))
            rad = self.conf_scale * np.sqrt(
                log_term / self.counts[s, pulled_mask]
            )

            mean_rew = (
                self.reward_sums[s, pulled_mask] / self.counts[s, pulled_mask]
            )
            mean_cost = (
                self.cost_sums[s, pulled_mask] / self.counts[s, pulled_mask]
            )

            ucb_rewards[pulled_mask] = np.clip(mean_rew + rad, 0.0, 1.0)
            lcb_costs[pulled_mask] = np.clip(mean_cost - rad, 0.0, 1.0)

        ucb_rewards[self.null_arm] = self.null_arm_reward
        lcb_costs[self.null_arm] = 0.0
        return ucb_rewards, lcb_costs

    def get_all_optimistic_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute `(ucb_rewards_2d, lcb_costs_2d)` of shape `(n_contexts, n_arms)`."""
        ucb_2d = np.empty((self.n_contexts, self.n_arms), dtype=np.float64)
        lcb_2d = np.empty((self.n_contexts, self.n_arms), dtype=np.float64)
        for s in range(self.n_contexts):
            ucb_2d[s], lcb_2d[s] = self.get_optimistic_bounds_for_context(s)
        return ucb_2d, lcb_2d

    def get_empirical_context_probs(self) -> np.ndarray:
        """
        Return Laplace-smoothed empirical arrival probabilities `p_hat(s)` over
        the `n_contexts` regimes observed so far.
        """
        smoothed = self.context_visits.astype(np.float64) + 1.0
        return smoothed / np.sum(smoothed)

    def select_arm(self, context: int | None = None) -> int:
        """
        Choose an arm for the current round given the active exogenous context.

        Parameters
        ----------
        context : int, optional
            If provided, updates `self.current_context` before solving the LP.

        Returns
        -------
        int
            Selected arm index in `[0, n_arms)`.
        """
        if context is not None:
            self.set_context(context)

        if self.remaining_budget <= 1e-9:
            self.last_lp_distribution = np.zeros(self.n_arms, dtype=np.float64)
            self.last_lp_distribution[self.null_arm] = 1.0
            return self.null_arm

        s = self.current_context
        if self.couple_regimes and self.n_contexts > 1:
            ucb_2d, lcb_2d = self.get_all_optimistic_bounds()
            ctx_probs = self.get_empirical_context_probs()
            policy_probs = solve_policy_knapsack_lp(
                ucb_rewards_2d=ucb_2d,
                lcb_costs_2d=lcb_2d,
                context_probs=ctx_probs,
                rho=self.rho,
                null_arm=self.null_arm,
            )
            probs = policy_probs[s]
        else:
            ucb_r, lcb_c = self.get_optimistic_bounds_for_context(s)
            probs = solve_1d_knapsack_lp(
                ucb_rewards=ucb_r,
                lcb_costs=lcb_c,
                rho=self.rho,
                null_arm=self.null_arm,
            )

        self.last_lp_distribution = probs
        return int(self.rng.choice(self.n_arms, p=probs))

    def update(self, arm: int, reward: float, info: dict | None = None) -> None:
        """
        Update statistics for the active context regime and deduct from the
        shared global budget.
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        if not (-1e-9 <= reward <= 1.0 + 1e-9):
            raise ValueError(f"reward must lie in [0, 1], got {reward}")

        raw_cost = 0.0
        s = self.current_context
        if info is not None:
            if "cost" in info:
                raw_cost = float(info["cost"])
                if not (-1e-9 <= raw_cost <= 1.0 + 1e-9):
                    raise ValueError(f"cost must lie in [0, 1], got {raw_cost}")
            if "context" in info and info["context"] is not None:
                s = int(info["context"])
                if not (0 <= s < self.n_contexts):
                    raise ValueError(
                        f"context must be in [0, {self.n_contexts}), got {s}"
                    )
                self.current_context = s

        clipped_reward = float(np.clip(reward, 0.0, 1.0))
        clipped_cost = (
            0.0 if arm == self.null_arm else float(np.clip(raw_cost, 0.0, 1.0))
        )

        self.t += 1
        self.context_visits[s] += 1
        self.remaining_budget = max(0.0, self.remaining_budget - clipped_cost)

        self.counts[s, arm] += 1
        self.reward_sums[s, arm] += clipped_reward
        self.cost_sums[s, arm] += clipped_cost

    def reset(self) -> None:
        """Reset budget, per-context tables, and round counter."""
        self.t = 0
        self.current_context = 0
        self.remaining_budget = self.initial_budget
        self.context_visits.fill(0)
        self.counts.fill(0)
        self.reward_sums.fill(0.0)
        self.cost_sums.fill(0.0)
        self.last_lp_distribution.fill(1.0 / self.n_arms)