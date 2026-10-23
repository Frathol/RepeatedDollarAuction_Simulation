"""
bwk.py

Implementations of Bandits with Knapsacks (BwK) algorithms with a single
global budget constraint (d = 1) and a null arm (arm 0):

1. SWUCBBwK : Sliding-Window UCB for Bandits with Knapsacks, introduced by
              Liu, Jiang, and Li (2022), "Non-stationary Bandits with
              Knapsacks" (Algorithm 1).
2. UCBBwK   : Stationary UCB for Bandits with Knapsacks (Agrawal and Devanur,
              2014), implemented as the full-history special case of SWUCBBwK
              with window_size = None (i.e. w = T).

Both classes strictly adhere to the BanditAlgorithm interface defined in
src/algorithms/base.py. Per-round observed resource consumption is read from
`info["cost"]` in update().
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
    best_single_idx = affordable_indices[0]
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


class SWUCBBwK(BanditAlgorithm):
    """
    Sliding-Window UCB for Bandits with Knapsacks (Liu, Jiang, & Li, 2022).

    Maintains sliding-window empirical estimates over the last `window_size`
    rounds (or Separate windows `window_reward` and `window_cost`). At each
    round t, solves the optimistic Linear Program using UCB for rewards and
    LCB for resource consumption costs, then samples an arm from the resulting
    distribution x_t. If remaining budget is exhausted (`remaining_budget <= 0`),
    falls back to the null arm (`null_arm = 0`).

    Parameters
    ----------
    n_arms : int
        Number of arms in the strategy set S0 (including the null arm at 0).
    budget : float
        Total global budget B across the horizon T.
    horizon : int
        Total time horizon T (used to compute target rate rho = B / T).
    window_size : int or None, default=None
        Sliding window length w in rounds. If None or >= horizon, uses full
        history (equivalent to stationary UCB-BwK).
    conf_scale : float, default=0.5
        Confidence radius scaling constant c_conf in
        rad_t(a) = c_conf * sqrt(ln(min(t, w) + 1) / N_{t,w}(a)).
    null_arm : int, default=0
        Index of the zero-cost pass strategy (theta = 0).
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

        self.initial_budget: float = float(budget)
        self.remaining_budget: float = float(budget)
        self.horizon: int = int(horizon)
        self.rho: float = self.initial_budget / float(self.horizon)
        self.window_size: int | None = (
            int(window_size) if window_size is not None else None
        )
        self.conf_scale: float = float(conf_scale)
        self.null_arm: int = int(null_arm)

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
            [0, 1]. The null arm always has exact (0.0, 0.0) bounds.
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

        # The null arm (theta = 0) has known deterministic zero reward & cost
        ucb_rewards[self.null_arm] = 0.0
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
        clipped_cost = 0.0 if arm == self.null_arm else float(np.clip(raw_cost, 0.0, 1.0))

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
        rng=None,
    ) -> None:
        super().__init__(
            n_arms=n_arms,
            budget=budget,
            horizon=horizon,
            window_size=None,
            conf_scale=conf_scale,
            null_arm=null_arm,
            rng=rng,
        )