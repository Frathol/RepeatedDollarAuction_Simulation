"""
alice_rational.py

Alice: a rational, non-learning opponent agent. She always follows the
same fixed decision rule, regardless of round index t or auction
history -- she does NOT accumulate experience across rounds (see the
design_decisions.md note on "only the main agent learns").

Her rule (adapted from the earlier draft discussion, simplified to a
deterministic strategy so it fits the paper's formal strategy
definition f(x, y)):

    Continue raising as long as her own next bid would not exceed a
    fraction mu of the stake s; fold as soon as continuing would.

This is intentionally simple and fully deterministic, matching the
"deterministic strategy" assumption in Waniek et al. It also serves as
a natural rational baseline opponent for Stage 0 (baseline replication)
and as part of the population in later stages.
"""

from __future__ import annotations

from typing import Callable

StrategyFn = Callable[[int, int], int]


class Alice:
    """
    Rational agent with a fixed profit-maximizing fold threshold.

    Parameters
    ----------
    stake : float
        Value of the prize (s).
    budget : int
        Alice's maximum bid (b).
    mu : float, default 0.8
        Fraction of the stake beyond which Alice considers continuing
        unprofitable and folds. E.g. mu=0.8 with stake=$10 means Alice
        will not push her own bid past $8.
    increment : int, default 1
        Minimal bid increment (delta).
    """

    def __init__(
        self,
        stake: float,
        budget: int,
        mu: float = 0.8,
        increment: int = 1,
    ) -> None:
        self.stake = stake
        self.budget = budget
        self.mu = mu
        self.increment = increment
        self.fold_threshold = mu * stake

    def get_strategy(self) -> StrategyFn:
        """
        Returns the strategy function f(x, y) for Alice. Since Alice's
        rule does not depend on t (round index) or on any accumulated
        state, this can safely be called once and reused across all
        rounds -- but is exposed as a method (rather than a bare
        function) for interface symmetry with Bob, whose strategy DOES
        depend on t.
        """

        def strategy(x: int, y: int) -> int:
            next_bid = y + self.increment
            if next_bid > self.fold_threshold or next_bid > self.budget:
                return y  # pass / fold
            return next_bid

        return strategy
