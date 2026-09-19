"""
strategies.py

Defines the strategy set S0 (the "arms" of the bandit problem), built
from O'Neill's (1986) optimal non-spiteful strategy and the "strategy
with threshold" construction from Waniek et al. (2016), Section 5.

Recap of the theory (see our earlier discussion):
    - O'Neill's strategy f_hat is provably optimal against a
      non-spiteful, rational opponent, GIVEN knowledge of both
      players' budgets.
    - A "strategy with threshold theta" follows f_hat up to the point
      where the opponent's bid would exceed theta, then passes.
    - Waniek et al. show (Theorem 6) that restricting S0 to this
      family of threshold strategies gives a much tighter regret bound
      for ELP (O~(sqrt(T)) instead of O~(sqrt(|S0| * T))), because the
      resulting neighbourhood graph is a clique (independence number
      alpha(G) = 1).

Each strategy in this module is a plain Python function with signature
strategy(x, y) -> next_bid, matching the StrategyFn type used by
DollarAuction.run(). This keeps strategies fully interchangeable with
the environment and with the ELP "prefix" side-information logic
(Lemma 1 in the paper), which depends on strategies being deterministic
functions of (x, y).
"""

from __future__ import annotations

from typing import Callable, List

StrategyFn = Callable[[int, int], int]


def oneill_optimal_strategy(budget: int, stake: float, increment: int = 1) -> StrategyFn:
    """
    O'Neill's (1986) optimal pure strategy against a non-spiteful
    opponent, assuming equal budgets `budget` for both players.

    Following the paper's Section 5.1:
        x0 = (b - 1) mod (s - 1) + 1
        x_i = x0 + i * (s - 1),  for i > 0
        x_(-1) = 0

        f_hat(x, y) = x_i   if x == x_{i-1}
                      y     otherwise (i.e. pass)

    for y in [x_{i-1}, x_i - 1].

    In words: the strategy jumps straight to a specific "magic" bid
    that guarantees the opponent can never profitably top it without
    exceeding their own budget, forcing them to fold immediately.
    """
    b = budget
    s = stake

    # Precompute the sequence of "magic" bid thresholds x_i, as far as
    # they can go within the budget.
    x0 = (b - 1) % (s - 1) + 1 if s > 1 else 1
    thresholds: List[int] = [0]  # x_{-1} = 0 as a sentinel
    i = 0
    while True:
        xi = x0 + i * (s - 1)
        if xi > b:
            break
        thresholds.append(xi)
        i += 1

    def strategy(x: int, y: int) -> int:
        # Find which threshold interval y currently falls into and
        # respond with the next threshold, if it's still affordable;
        # otherwise pass.
        for idx in range(1, len(thresholds)):
            lower = thresholds[idx - 1]
            upper = thresholds[idx] - 1
            if lower <= y <= upper:
                candidate = thresholds[idx]
                return candidate if candidate <= b else y
        # y is beyond all known thresholds -> nothing profitable left.
        return y

    return strategy


def threshold_strategy(budget: int, stake: float, theta: int, increment: int = 1) -> StrategyFn:
    """
    A "strategy with threshold theta" (Waniek et al., Section 5.2):
    follow O'Neill's optimal strategy up to the point where the
    opponent's bid y reaches or exceeds theta, then pass.

    This is the family used as the default S0 (arm set) in our
    experiments: each distinct theta in {0, ..., budget} is one arm.
    Lower theta = more risk-averse (folds earlier); theta = budget
    reduces to the unrestricted O'Neill strategy.
    """
    base = oneill_optimal_strategy(budget, stake, increment)

    def strategy(x: int, y: int) -> int:
        if y >= theta:
            return y  # pass
        return base(x, y)

    return strategy


def build_threshold_arm_set(
    budget: int, stake: float, thetas: List[int] | None = None, increment: int = 1
) -> List[StrategyFn]:
    """
    Convenience constructor for S0: a list of threshold strategies,
    one per theta value. If `thetas` is not given, defaults to every
    integer threshold from 0 to `budget`.

    Returns
    -------
    List[StrategyFn]
        Index i in this list corresponds to arm i in the bandit
        algorithms (see src/algorithms/base.py).
    """
    if thetas is None:
        thetas = list(range(0, budget + 1))
    return [threshold_strategy(budget, stake, theta, increment) for theta in thetas]


def fixed_opening_bid_strategy(opening_bid: int, follow_up: StrategyFn | None = None) -> StrategyFn:
    """
    A simpler, "naive" strategy family: open with a fixed bid, then
    optionally delegate subsequent decisions to `follow_up` (defaults
    to "always match the opponent's raise by exactly one increment").

    This is provided mainly for Stage 0 sanity checks against very
    simple baselines -- NOT the primary S0 used for the thesis's main
    claims, which should use the threshold-strategy family above to
    match Waniek et al.'s Theorem 6 setting.
    """

    def default_follow_up(x: int, y: int) -> int:
        return y + 1

    follow_up = follow_up or default_follow_up

    def strategy(x: int, y: int) -> int:
        if x == 0 and y == 0:
            return opening_bid
        return follow_up(x, y)

    return strategy
