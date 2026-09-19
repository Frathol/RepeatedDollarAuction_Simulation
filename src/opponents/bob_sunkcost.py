"""
bob_sunkcost.py

Bob: a rule-based (non-learning) opponent agent that models the
sunk-cost fallacy. Like Alice, Bob does not "learn" in the bandit
sense -- his rule is scripted, but the rule itself changes over time
(t) and reacts to his own accumulated sunk cost within an auction.
This is the "scripted but non-stationary from the agent's point of
view" distinction discussed earlier (see docs/design_decisions.md).

Two modes are implemented, corresponding to Stage 1a and Stage 1b of
the experiment plan:

    - "single_switch": Bob behaves like Alice (rational) for rounds
      t <= switch_round. From switch_round onward, he is PERMANENTLY
      in "escalation mode": once his own bid within an auction crosses
      `sunk_cost_threshold`, he continues bidding with a probability
      that increases with his accumulated sunk cost (an exponential
      escalation curve), up to a hard ceiling `escalation_ceiling`.

    - "recurring_switch": Bob toggles between rational and escalation
      mode multiple times across the T-round horizon, according to a
      list of switch points. This is the harder, more realistic test
      case that demonstrates the practical value of EXP3.S +
      Page-Hinkley over algorithms that only need to adapt once.

IMPORTANT (see our earlier discussion on determinism): because Bob's
escalation decision is PROBABILISTIC (not a deterministic function of
(x, y) alone), his strategy is NOT a valid deterministic strategy in
the sense required by Waniek et al.'s Lemma 1 (the prefix
side-information trick). This is intentional and should be flagged
explicitly in the thesis: it is precisely the relaxation that breaks
ELP's side-information assumption and motivates evaluating ELP under
a weakened/adapted guarantee when facing Bob.
"""

from __future__ import annotations

from typing import Callable, List, Optional

StrategyFn = Callable[[int, int], int]


class Bob:
    """
    Sunk-cost fallacy opponent agent.

    Parameters
    ----------
    stake : float
        Value of the prize (s).
    budget : int
        Bob's nominal maximum bid under rational play (b). Note his
        escalation ceiling may legitimately exceed this once he is in
        escalation mode -- this models a player who is willing to
        "dig into" funds beyond their normal comfort budget once
        sunk-cost reasoning takes over.
    increment : int, default 1
        Minimal bid increment (delta).
    mu : float, default 0.8
        Same meaning as Alice's mu: while rational, Bob folds once his
        next bid would exceed mu * stake.
    sunk_cost_threshold : float, default 5.0
        The amount of Bob's own accumulated bid within ONE auction
        (i.e. his current x) beyond which escalation dynamics can
        trigger, once he is in escalation mode.
    escalation_rate : float, default 0.5
        Rate parameter of the exponential escalation curve: higher
        values mean the probability of continuing to bid rises faster
        as sunk cost accumulates beyond the threshold.
    escalation_ceiling : float, default 20.0
        Hard cap on how high Bob will ever bid, even while escalating.
    mode : {"single_switch", "recurring_switch"}, default "single_switch"
        Which temporal escalation pattern to use (see module docstring).
    switch_round : int, default 1000
        For "single_switch" mode: the round t after which Bob
        permanently enters escalation mode.
    switch_points : list[int], optional
        For "recurring_switch" mode: a sorted list of round indices at
        which Bob's regime flips (rational <-> escalation). E.g.
        [1000, 3000, 4000, 7000] means: rational until 1000, escalation
        from 1000-3000, rational from 3000-4000, escalation from
        4000-7000, rational from 7000 onward.
    rng : numpy.random.Generator, optional
        Random generator for reproducibility.
    """

    def __init__(
        self,
        stake: float,
        budget: int,
        increment: int = 1,
        mu: float = 0.8,
        sunk_cost_threshold: float = 5.0,
        escalation_rate: float = 0.5,
        escalation_ceiling: float = 20.0,
        mode: str = "single_switch",
        switch_round: int = 1000,
        switch_points: Optional[List[int]] = None,
        rng=None,
    ) -> None:
        import numpy as np

        if mode not in ("single_switch", "recurring_switch"):
            raise ValueError("mode must be 'single_switch' or 'recurring_switch'")
        if mode == "recurring_switch" and not switch_points:
            raise ValueError(
                "switch_points must be provided (non-empty) when "
                "mode='recurring_switch'"
            )

        self.stake = stake
        self.budget = budget
        self.increment = increment
        self.mu = mu
        self.fold_threshold = mu * stake
        self.sunk_cost_threshold = sunk_cost_threshold
        self.escalation_rate = escalation_rate
        self.escalation_ceiling = escalation_ceiling
        self.mode = mode
        self.switch_round = switch_round
        self.switch_points = sorted(switch_points) if switch_points else []
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------
    # Regime logic: is Bob "rational" or "escalating" at round t?
    # ------------------------------------------------------------------

    def is_escalating_at(self, t: int) -> bool:
        """
        Ground-truth regime label for round t. This is exactly the
        ground truth used later to compute detection delay for
        Page-Hinkley (src/metrics/detection_delay.py) and to compute
        switching regret per-phase (src/metrics/regret.py) -- so
        keep this function as the single source of truth for "when
        did Bob's regime actually change".
        """
        if self.mode == "single_switch":
            return t > self.switch_round

        # recurring_switch: count how many switch points have passed;
        # an even count means still in the ORIGINAL regime (rational),
        # an odd count means currently flipped (escalating).
        n_switches_passed = sum(1 for sp in self.switch_points if t > sp)
        return n_switches_passed % 2 == 1

    # ------------------------------------------------------------------
    # Strategy construction
    # ------------------------------------------------------------------

    def get_strategy(self, t: int) -> StrategyFn:
        """
        Returns the strategy function f(x, y) for Bob AT ROUND t.

        Unlike Alice, this must be re-obtained (or at least
        re-evaluated) for every round, since which branch of logic
        applies depends on t via `is_escalating_at`.

        Note the returned function is stochastic when Bob is in
        escalation mode (see module docstring on the determinism
        implications).
        """
        escalating = self.is_escalating_at(t)

        def strategy(x: int, y: int) -> int:
            next_bid = y + self.increment

            if not escalating:
                # Rational phase: identical logic to Alice.
                if next_bid > self.fold_threshold or next_bid > self.budget:
                    return y
                return next_bid

            # Escalation phase.
            if x < self.sunk_cost_threshold:
                # Not yet "hooked" -- behave rationally until the sunk
                # cost threshold is crossed within this auction.
                if next_bid > self.fold_threshold or next_bid > self.budget:
                    return y
                return next_bid

            # Past the sunk-cost threshold: probability of continuing
            # increases with accumulated sunk cost x, following an
            # exponential escalation curve, capped at escalation_ceiling.
            if next_bid > self.escalation_ceiling:
                return y  # hard ceiling reached -> fold

            p_continue = 1 - pow(2.71828182845905, -self.escalation_rate * (x - self.sunk_cost_threshold))
            # p_continue in [0, 1); rises toward 1 as x grows past the
            # threshold. (Using pow(e, ...) directly to avoid an extra
            # numpy/math import at module load time; swap for
            # numpy.exp / math.exp freely.)

            if self.rng.random() < p_continue:
                return next_bid
            return y  # folds despite being "hooked", with prob 1 - p_continue

        return strategy
