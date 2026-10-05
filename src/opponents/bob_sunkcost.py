"""
bob_sunkcost.py

Bob: ADAPTIVE ADVERSARY with sunk-cost escalation (design_decisions.md
sec. 13-16).

What "adaptive" means here (and what it does not)
-------------------------------------------------
Bob's regime for round t (Rational / Escalating) depends on the agent's
behaviour in PREVIOUS rounds (a rolling window of her aggressiveness),
never on the round index. Inside one auction his bids are a function of
the live state (x, y) as before; that within-auction reactivity is NOT
what makes him adaptive (sec. 13).

Bob is adaptive but not a learner: his reaction rule is fixed and hand
designed (window mean -> escalation probability); nothing in it optimizes
an objective of Bob's own (sec. 2).

Per-round protocol (the runner MUST follow this order)
------------------------------------------------------
    regime  = bob.begin_round()          # decide regime from history up to t-1
    log.append((t, bob.regime_label))    # realized_regime_log (sec. 15)
    strat   = bob.get_strategy()         # fresh closure every round
    result  = env.run(alice_strategy, strat, ...)
    bob.observe_agent_move(signal_t)     # signal in [0,1], e.g. theta_t / budget

Snapshot / hindsight safety (README rule, sec. 16)
--------------------------------------------------
`snapshot()` returns a deep copy including the private RNG, so the copy
and the live Bob evolve independently: stepping a snapshot never mutates
the live window, dwell counter or RNG.
  * Never cache the closure returned by get_strategy() on the instance:
    deepcopy does not copy functions, so a cached closure in a clone
    would still point at the ORIGINAL Bob and its RNG.
  * Always call get_strategy() on the object (live or snapshot) you are
    actually stepping.
  * The regime log lives in the runner, not in Bob, so snapshots stay
    small (window deque + a few scalars + RNG state).

Design extension flagged for your decision: `min_dwell`
-------------------------------------------------------
With min_dwell=1 the regime is re-drawn EVERY round (exactly sec. 13), so
the regime can flicker round to round and "phases" in realized_regime_log
become 1-round long. Per-phase hindsight regret needs persistent phases,
so `min_dwell>1` holds a drawn regime for at least that many rounds. The
default (1) preserves sec. 13 behaviour.
"""

from __future__ import annotations

import copy
import math
import warnings
from collections import deque
from typing import Callable, Deque, Dict, Optional

import numpy as np

StrategyFn = Callable[[int, int], int]

REGIME_RATIONAL = "Rational"
REGIME_ESCALATING = "Escalating"


class Bob:
    """
    Parameters
    ----------
    stake, budget, increment, mu : as in Alice. While rational, Bob folds
        when his next bid would exceed mu * stake or the budget.
    sunk_cost_threshold : float, optional
        Bob's own bid x at/above which sunk-cost dynamics apply (while
        escalating). Default 0.5 * fold_threshold. Must be <= fold
        threshold, otherwise Bob can never reach it and escalation is
        silently a no-op (e.g. stake=5, mu=0.8 -> fold at 4 < old default
        threshold 5.0).
    prior_sunk_cost : float
        Investment Bob ALREADY has when the auction starts (e.g. compute
        progress of a job that was running before). Added to his own bid
        when deciding whether he is "hooked": effective sunk cost =
        x + prior_sunk_cost. Default 0 = original behaviour. Without it
        Bob only becomes hooked after HE has bid, so when the learner opens
        the auction Bob folds like a rational player and escalation can
        only matter in the ~half of rounds where Bob starts. When > 0, Bob
        still always opens if asked to move first (a hooked Bob never
        "skips" an auction).
    escalation_rate : float
        Rate of the within-auction curve p_continue = 1 - exp(-rate*(x-thr)).
        (Unchanged from the previous version; note p_continue = 0 exactly
        at x == thr.)
    escalation_ceiling : float, optional
        Hard cap on Bob's bid while escalating. Default = budget. The
        environment clips all bids to the shared budget, so a ceiling
        above `budget` is clamped (with a warning).
    window : int
        Size W of the rolling window of the agent's aggressiveness signals.
    escalation_gain : float
        p_escalate = clip(gain * mean(window), 0, 1). With signals
        theta/budget and gain=1 this is sec. 13's min(1, mean_theta/budget).
    min_dwell : int
        Minimum number of rounds a drawn regime is held (see module doc).
    min_history : int, optional
        Bob does not react (p_escalate = 0) until his window holds at least
        this many observations. Default None = `window`, i.e. he waits for
        a FULL window. (With min_history=1 a single early observation can
        already give p_escalate = 1, which is how the first smoke run showed
        P_Escalate=1.0 at round 2.)
    fixed_regime : bool, optional
        None (default) = adaptive Bob. True/False pins the regime to
        Escalating/Rational forever (an "oracle" Bob). Used by calibration
        and sweeps to measure each regime in isolation; never in the real
        experiments.
    rng : numpy.random.Generator, optional
        Bob's PRIVATE generator. Do not share it with the agent, the
        environment, or any hindsight code.
    """

    def __init__(
        self,
        stake: float,
        budget: int,
        increment: int = 1,
        mu: float = 0.8,
        sunk_cost_threshold: Optional[float] = None,
        prior_sunk_cost: float = 0.0,
        escalation_rate: float = 0.5,
        escalation_ceiling: Optional[float] = None,
        window: int = 50,
        escalation_gain: float = 1.0,
        min_dwell: int = 1,
        min_history: Optional[int] = None,
        fixed_regime: Optional[bool] = None,
        rng=None,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if min_history is None:
            min_history = window
        if not (1 <= min_history <= window):
            raise ValueError("min_history must be in [1, window]")
        if min_dwell < 1:
            raise ValueError("min_dwell must be >= 1")

        self.stake = stake
        self.budget = budget
        self.increment = increment
        self.mu = mu
        self.fold_threshold = mu * stake

        if sunk_cost_threshold is None:
            sunk_cost_threshold = 0.5 * self.fold_threshold
        if sunk_cost_threshold > self.fold_threshold:
            raise ValueError(
                f"sunk_cost_threshold={sunk_cost_threshold} exceeds the "
                f"rational fold threshold mu*stake={self.fold_threshold}: "
                "Bob could never reach it and escalation would be a no-op."
            )
        self.sunk_cost_threshold = sunk_cost_threshold

        if escalation_ceiling is None:
            escalation_ceiling = float(budget)
        elif escalation_ceiling > budget:
            warnings.warn(
                f"escalation_ceiling={escalation_ceiling} > budget={budget}; "
                "the environment clips bids to the shared budget, clamping.",
                stacklevel=2,
            )
            escalation_ceiling = float(budget)
        if escalation_ceiling <= self.fold_threshold:
            warnings.warn(
                "escalation_ceiling <= rational fold threshold: escalation "
                "cannot push Bob beyond rational play.",
                stacklevel=2,
            )
        self.escalation_ceiling = escalation_ceiling

        if prior_sunk_cost < 0:
            raise ValueError("prior_sunk_cost must be >= 0")
        self.prior_sunk_cost = prior_sunk_cost
        self.escalation_rate = escalation_rate
        self.window = window
        self.escalation_gain = escalation_gain
        self.min_dwell = min_dwell
        self.min_history = min_history
        self.fixed_regime = fixed_regime
        self.rng = rng if rng is not None else np.random.default_rng()

        self._window: Deque[float] = deque(maxlen=window)
        self._escalating: bool = False
        self._dwell_left: int = 0

    # ------------------------------------------------------------------
    # History-dependent regime logic
    # ------------------------------------------------------------------

    def aggressiveness_score(self) -> float:
        """Mean of the agent's recent aggressiveness signals (0 if none)."""
        if not self._window:
            return 0.0
        return float(sum(self._window) / len(self._window))

    def p_escalate(self) -> float:
        """Probability of entering/holding escalation, from history only."""
        if len(self._window) < self.min_history:
            return 0.0
        return min(1.0, max(0.0, self.escalation_gain * self.aggressiveness_score()))

    def begin_round(self) -> bool:
        """
        Decide this round's regime from history up to the previous round.
        Call exactly once per round, BEFORE get_strategy(). Returns True
        if escalating.
        """
        if self.fixed_regime is not None:
            self._escalating = bool(self.fixed_regime)
            return self._escalating
        if self._dwell_left > 0:
            self._dwell_left -= 1
        else:
            self._escalating = bool(self.rng.random() < self.p_escalate())
            self._dwell_left = self.min_dwell - 1
        return self._escalating

    def observe_agent_move(self, signal: float) -> None:
        """
        Record the agent's aggressiveness for the round just played.
        `signal` should be in [0, 1] (e.g. theta_t / budget; an observable
        alternative is the agent's final bid / budget). Clipped to [0, 1].
        """
        self._window.append(min(1.0, max(0.0, float(signal))))

    @property
    def current_regime(self) -> bool:
        """True if escalating in the current round."""
        return self._escalating

    @property
    def regime_label(self) -> str:
        """Label for the DataCollector's Bob_Regime column."""
        return REGIME_ESCALATING if self._escalating else REGIME_RATIONAL

    # ------------------------------------------------------------------
    # Within-auction strategy
    # ------------------------------------------------------------------

    def get_strategy(self) -> StrategyFn:
        """
        Strategy f(x, y) for the CURRENT round (regime captured now).
        Stochastic while escalating (draws from this Bob's private RNG).
        Build it fresh each round; do not cache it (see module doc).
        """
        escalating = self._escalating

        def strategy(x: int, y: int) -> int:
            next_bid = y + self.increment
            sunk = x + self.prior_sunk_cost
            opening = (x == 0 and y == 0)       # Bob is asked to move first

            if ((not escalating) or sunk < self.sunk_cost_threshold
                    or (opening and self.prior_sunk_cost > 0)):
                if next_bid > self.fold_threshold or next_bid > self.budget:
                    return y
                return next_bid

            if next_bid > self.escalation_ceiling:
                return y

            p_continue = 1.0 - math.exp(
                -self.escalation_rate * (sunk - self.sunk_cost_threshold)
            )
            if self.rng.random() < p_continue:
                return next_bid
            return y

        return strategy

    # ------------------------------------------------------------------
    # Snapshot support (hindsight / counterfactual re-simulation)
    # ------------------------------------------------------------------

    def snapshot(self) -> "Bob":
        """Independent deep copy, including the private RNG state."""
        return copy.deepcopy(self)

    def fingerprint(self) -> Dict:
        """Comparable summary of all mutable state (for tests)."""
        return {
            "window": tuple(self._window),
            "escalating": self._escalating,
            "dwell_left": self._dwell_left,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }