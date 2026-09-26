"""
exp3.py

Classic EXP3 (Exponential-weight algorithm for Exploration and
Exploitation), Auer et al. (2002). This is the "vanilla" adversarial
bandit baseline -- no side-information, no prefix tricks, just plain
importance-weighted exponential weighting. Contrast with ELP
(src/algorithms/elp.py), which is EXP3-family too but additionally
exploits the threshold-strategy prefix structure to update multiple
arms per round instead of just the one played.

Update rule (Auer et al. 2002, classic form):
    x_hat_i(t) = r_i(t) / p_i(t)   if i == played arm
                 0                  otherwise
    w_i(t+1) = w_i(t) * exp(gamma * x_hat_i(t) / K)

Selection probability:
    p_i(t) = (1 - gamma) * w_i(t)/sum_j w_j(t)  +  gamma/K

This matches exactly what we derived by hand in the walkthrough: the
importance-weighted estimator (dividing by the arm's own selection
probability) keeps the estimator unbiased despite only observing one
arm per round, and the gamma/K floor guarantees every arm is tried
often enough to keep that estimator numerically stable.
"""

from __future__ import annotations

import math

from src.algorithms.base import BanditAlgorithm


class EXP3(BanditAlgorithm):
    """
    Parameters
    ----------
    n_arms : int
    gamma : float, optional
        Exploration/mixing parameter in (0, 1]. If not given, computed
        via the theoretically optimal formula from Auer et al. (2002),
        Corollary 4.2, using `horizon_T` (which must then be given):

            gamma = min(1, sqrt(K * ln(K) / ((e - 1) * T)))

        This is the classic "you must know T in advance to tune gamma
        optimally" fixed-horizon EXP3 (same finite-horizon spirit as
        the Corollary 7 beta used in ELP -- see elp.py). An "anytime"
        doubling-trick variant exists in the literature but is not
        needed for this thesis's fixed-T experiments.
    horizon_T : int, optional
        Used only to compute the default gamma if not given explicitly.
    rng : numpy.random.Generator, optional
    """

    def __init__(
        self,
        n_arms: int,
        gamma: float | None = None,
        horizon_T: int | None = None,
        rng=None,
    ) -> None:
        super().__init__(n_arms=n_arms, rng=rng)

        if gamma is None:
            if horizon_T is None:
                raise ValueError(
                    "Either gamma or horizon_T must be provided "
                    "(horizon_T is used to compute the theoretically "
                    "optimal gamma via Auer et al. 2002, Corollary 4.2)."
                )
            gamma = self.optimal_gamma(n_arms, horizon_T)
        if not (0 < gamma <= 1):
            raise ValueError("gamma must be in (0, 1]")
        self.gamma = gamma

        self.w = [1.0] * n_arms
        self._last_probs = [1.0 / n_arms] * n_arms

    @staticmethod
    def optimal_gamma(n_arms: int, horizon_T: int) -> float:
        """
        gamma = min(1, sqrt(K * ln(K) / ((e - 1) * T)))

        The classic fixed-horizon-optimal exploration rate for EXP3
        (Auer, Cesa-Bianchi, Freund, & Schapire, 2002, Corollary 4.2),
        which yields the O(sqrt(K T ln K)) regret bound.
        """
        if n_arms <= 1:
            raise ValueError("n_arms must be at least 2")
        e_minus_1 = math.e - 1
        value = math.sqrt(n_arms * math.log(n_arms) / (e_minus_1 * horizon_T))
        return min(1.0, value)

    def select_arm(self) -> int:
        total_w = sum(self.w)
        probs = [
            (1 - self.gamma) * (self.w[k] / total_w) + self.gamma / self.n_arms
            for k in range(self.n_arms)
        ]
        # normalize away any floating point drift
        s = sum(probs)
        probs = [p / s for p in probs]
        self._last_probs = probs

        arm = int(self.rng.choice(self.n_arms, p=probs))
        return arm

    def update(self, arm: int, reward: float, info: dict | None = None) -> None:
        # info is accepted for interface uniformity with ELP, but
        # plain EXP3 does not use any side-information -- it only
        # ever updates the ONE arm actually played.
        p = self._last_probs[arm]
        x_hat = reward / p if p > 0 else 0.0
        self.w[arm] = self.w[arm] * math.exp(self.gamma * x_hat / self.n_arms)
        self.t += 1

    def reset(self) -> None:
        """Reset weights to uniform, as if starting fresh. Used later
        by EXP3.S's Page-Hinkley-triggered reset (see exp3s.py)."""
        self.w = [1.0] * self.n_arms
        self._last_probs = [1.0 / self.n_arms] * self.n_arms
        self.t = 0