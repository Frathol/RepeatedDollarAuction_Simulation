"""
base.py

Common interface for all bandit algorithms (ELP, EXP3, EXP3.S, and any
future variant). Every algorithm implementation in src/algorithms/
MUST subclass BanditAlgorithm and implement select_arm() and update(),
so that src/simulation/runner.py can run any of them interchangeably
without algorithm-specific branching.

This is the interface convention documented in the project README.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BanditAlgorithm(ABC):
    """
    Abstract base class for a multi-armed bandit strategy-selection
    algorithm.

    Parameters
    ----------
    n_arms : int
        Number of arms (i.e. |S0|, the size of the strategy set).
    rng : numpy.random.Generator, optional
        Random generator for reproducibility. If not provided, a
        fresh default generator is created.
    """

    def __init__(self, n_arms: int, rng=None) -> None:
        import numpy as np

        if n_arms <= 0:
            raise ValueError("n_arms must be a positive integer")
        self.n_arms = n_arms
        self.rng = rng if rng is not None else np.random.default_rng()
        self.t = 0  # round counter, incremented on every update()

    @abstractmethod
    def select_arm(self) -> int:
        """
        Choose an arm (strategy index) to play in the current round.

        Returns
        -------
        int
            Index into the strategy set S0, in range [0, n_arms).
        """
        raise NotImplementedError

    @abstractmethod
    def update(self, arm: int, reward: float, info: dict | None = None) -> None:
        """
        Update the algorithm's internal state after observing the
        outcome of playing `arm` and receiving `reward`.

        Parameters
        ----------
        arm : int
            The arm that was actually played this round.
        reward : float
            The observed reward, expected to lie in [0, 1] (see
            DollarAuction.normalize_reward in
            src/environment/dollar_auction.py).
        info : dict, optional
            Extra side-information some algorithms need beyond the
            scalar reward of the arm played. Plain reward-only
            algorithms (EXP3, EXP3.S) ignore this entirely. ELP
            requires it: runner.py must pass
            {"agent_state_trace": AuctionResult.agent_state_trace}
            so ELP can apply Lemma 1 (prefix side-information) to
            infer rewards for every strategy that is a prefix of the
            one actually played, not just the played arm itself. This
            keeps the interface uniform -- runner.py always calls
            update() the same way regardless of which algorithm is
            active; algorithms that don't need `info` simply don't
            read it.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Optional: reset the algorithm to its initial state. Not all
        algorithms need this, but it is required for anything that
        implements an explicit "prior reset" behaviour (e.g. EXP3.S
        combined with a Page-Hinkley changepoint trigger -- see
        src/algorithms/page_hinkley.py and src/algorithms/exp3s.py).

        Default implementation raises NotImplementedError; subclasses
        that support resets should override this.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement reset()."
        )

    @property
    def name(self) -> str:
        """Human-readable identifier used in logged results (the
        'algorithm' column described in the README)."""
        return self.__class__.__name__
