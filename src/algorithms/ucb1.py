"""
ucb1.py

Implementation of the Upper Confidence Bound (UCB1) algorithm introduced by
Auer, Cesa-Bianchi, and Fischer (2002), "Finite-time Analysis of the
Multiarmed Bandit Problem".

UCB1 is a stationary stochastic bandit baseline that relies on the principle
of Optimism in the Face of Uncertainty (OFU) derived from Hoeffding's
inequality. It ignores budget constraints and side-information (`info`).
"""

from __future__ import annotations

import numpy as np

from src.algorithms.base import BanditAlgorithm


class UCB1(BanditAlgorithm):
    """
    Upper Confidence Bound (UCB1) bandit algorithm.

    During the first `n_arms` rounds, each arm is pulled exactly once
    in sequential order. In subsequent rounds, the algorithm selects the
    arm that maximizes:

        UCB_i(t) = mu_hat_i(t - 1) + c * sqrt((2 * ln(t)) / n_i(t - 1))

    Parameters
    ----------
    n_arms : int
        Number of arms (i.e. |S0|, the size of the threshold strategy set).
    c : float, default=1.0
        Exploration confidence scaling constant. Standard UCB1 uses c = 1.0.
    rng : numpy.random.Generator, optional
        Random generator used for random tie-breaking. If not provided, a
        fresh default generator is created.
    """

    def __init__(self, n_arms: int, c: float = 1.0, rng=None) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        if c < 0.0:
            raise ValueError("Exploration constant c must be non-negative")
        self.c: float = float(c)
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.int64)
        self.values: np.ndarray = np.zeros(self.n_arms, dtype=np.float64)

    def get_ucb_scores(self) -> np.ndarray:
        """
        Compute current UCB1 index values for all arms at decision step t + 1.

        Returns
        -------
        numpy.ndarray
            Array of shape `(n_arms,)` containing UCB1 indices. Unpulled arms
            (`counts == 0`) are assigned `+np.inf`.
        """
        scores = np.full(self.n_arms, np.inf, dtype=np.float64)
        pulled_mask = self.counts > 0
        if np.any(pulled_mask):
            t_next = max(1, self.t + 1)
            bonus = self.c * np.sqrt(
                (2.0 * np.log(t_next)) / self.counts[pulled_mask]
            )
            scores[pulled_mask] = self.values[pulled_mask] + bonus
        return scores

    def select_arm(self) -> int:
        """
        Choose an arm (strategy index) to play in the current round.

        Returns
        -------
        int
            Index into the strategy set S0, in range [0, n_arms).
        """
        unpulled = np.flatnonzero(self.counts == 0)
        if unpulled.size > 0:
            return int(unpulled[0])

        scores = self.get_ucb_scores()
        max_score = np.max(scores)
        best_arms = np.flatnonzero(
            np.isclose(scores, max_score, rtol=1e-12, atol=1e-12)
        )
        if best_arms.size == 1:
            return int(best_arms[0])
        return int(self.rng.choice(best_arms))

    def update(self, arm: int, reward: float, info: dict | None = None) -> None:
        """
        Update empirical pull counts and mean rewards for the played arm.

        Parameters
        ----------
        arm : int
            The arm that was actually played this round, in [0, n_arms).
        reward : float
            The observed normalized reward in [0, 1].
        info : dict, optional
            Extra side-information passed by `runner.py`. Ignored by UCB1.
        """
        if not (0 <= arm < self.n_arms):
            raise ValueError(f"arm must be in [0, {self.n_arms}), got {arm}")
        if not (-1e-9 <= reward <= 1.0 + 1e-9):
            raise ValueError(f"reward must lie in [0, 1], got {reward}")

        clipped_reward = float(np.clip(reward, 0.0, 1.0))

        self.t += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        self.values[arm] += (clipped_reward - self.values[arm]) / n

    def reset(self) -> None:
        """Reset pull counts, empirical means, and round counter to zero."""
        self.t = 0
        self.counts.fill(0)
        self.values.fill(0.0)