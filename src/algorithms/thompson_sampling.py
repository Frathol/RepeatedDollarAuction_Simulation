"""
thompson_sampling.py

Implementation of Beta-Bernoulli Thompson Sampling for general bounded
stochastic rewards in [0, 1], following Algorithm 2 of Agrawal and Goyal
(2012), "Analysis of Thompson Sampling for the Multi-armed Bandit Problem".

To handle continuous normalized auction rewards in [0, 1] while preserving
the conjugate Beta(alpha, beta) posterior, each observed reward r_t in [0, 1]
is converted into a binary outcome via an independent Bernoulli trial with
success probability r_t.
"""

from __future__ import annotations

import numpy as np

from src.algorithms.base import BanditAlgorithm


class ThompsonSampling(BanditAlgorithm):
    """
    Beta-Bernoulli Thompson Sampling bandit algorithm.

    Maintains a Beta(alpha_i, beta_i) posterior distribution for each arm i.
    At each round, one independent sample theta_i ~ Beta(alpha_i, beta_i) is
    drawn per arm, and the arm with the largest sample is played.

    Parameters
    ----------
    n_arms : int
        Number of arms (i.e. |S0|, the size of the threshold strategy set).
    alpha_0 : float, default=1.0
        Initial alpha parameter for the Beta prior (uniform prior uses 1.0).
    beta_0 : float, default=1.0
        Initial beta parameter for the Beta prior (uniform prior uses 1.0).
    use_bernoulli_trial : bool, default=True
        If True, applies the Bernoulli trial trick from Agrawal and Goyal
        (2012, Algorithm 2) for continuous rewards in [0, 1]. If False,
        performs direct fractional pseudo-count updates.
    rng : numpy.random.Generator, optional
        Random generator for posterior sampling and Bernoulli trials. If not
        provided, a fresh default generator is created.
    """

    def __init__(
        self,
        n_arms: int,
        alpha_0: float = 1.0,
        beta_0: float = 1.0,
        use_bernoulli_trial: bool = True,
        rng=None,
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        if alpha_0 <= 0.0 or beta_0 <= 0.0:
            raise ValueError("Prior parameters alpha_0 and beta_0 must be > 0")

        self.alpha_0: float = float(alpha_0)
        self.beta_0: float = float(beta_0)
        self.use_bernoulli_trial: bool = bool(use_bernoulli_trial)

        self.alphas: np.ndarray = np.full(
            self.n_arms, self.alpha_0, dtype=np.float64
        )
        self.betas: np.ndarray = np.full(
            self.n_arms, self.beta_0, dtype=np.float64
        )
        self.counts: np.ndarray = np.zeros(self.n_arms, dtype=np.int64)
        self.values: np.ndarray = np.zeros(self.n_arms, dtype=np.float64)

    def sample_posterior(self) -> np.ndarray:
        """
        Draw one independent sample per arm from Beta(alphas, betas).

        Returns
        -------
        numpy.ndarray
            Array of shape `(n_arms,)` with posterior samples in (0, 1).
        """
        return self.rng.beta(self.alphas, self.betas)

    def select_arm(self) -> int:
        """
        Choose an arm by drawing from each arm's Beta posterior and picking
        the maximizer.

        Returns
        -------
        int
            Index into the strategy set S0, in range [0, n_arms).
        """
        theta_samples = self.sample_posterior()
        return int(np.argmax(theta_samples))

    def update(self, arm: int, reward: float, info: dict | None = None) -> None:
        """
        Update the Beta(alpha, beta) posterior parameters for the played arm.

        Parameters
        ----------
        arm : int
            The arm that was actually played this round, in [0, n_arms).
        reward : float
            The observed normalized reward in [0, 1].
        info : dict, optional
            Extra side-information passed by `runner.py`. Ignored by
            ThompsonSampling.
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

        if self.use_bernoulli_trial:
            trial_outcome = (
                1.0 if self.rng.random() < clipped_reward else 0.0
            )
            self.alphas[arm] += trial_outcome
            self.betas[arm] += 1.0 - trial_outcome
        else:
            self.alphas[arm] += clipped_reward
            self.betas[arm] += 1.0 - clipped_reward

    def reset(self) -> None:
        """Reset Beta posterior parameters, pull counts, and round counter."""
        self.t = 0
        self.alphas.fill(self.alpha_0)
        self.betas.fill(self.beta_0)
        self.counts.fill(0)
        self.values.fill(0.0)