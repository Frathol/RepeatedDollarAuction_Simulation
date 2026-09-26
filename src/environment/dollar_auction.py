"""
dollar_auction.py

Core 2-player Dollar Auction environment, following the formalization in:
Waniek, Tran-Thanh, & Michalak (2016), "Repeated Dollar Auctions:
A Multi-Armed Bandit Approach", AAMAS 2016, Section 2.1.

Notation (kept consistent with the paper and with docs/notation.md):
    b        : budget of both players (equal budgets assumed)
    s        : stake / prize value of the auction
    delta    : minimal bid increment (assumed = 1, as in the paper)
    x, y     : state (x, y) in X_b, where x = last bid of the player
               about to move, y = last bid of their opponent
    f(x, y)  : a strategy -- the bid a player makes when the state is (x, y)

Auction rules (all-pay):
    - Both players alternate making bids strictly higher than the
      opponent's last bid, or pass.
    - The auction ends when a player passes, or when a player cannot
      top the opponent's bid without exceeding their budget.
    - The higher bidder wins the stake s.
    - BOTH the winner and the runner-up must pay their final bid
      (this is what makes it an "all-pay" auction and is the source
      of the sunk-cost dynamics we are studying).

This module implements a SINGLE auction (one "round" in the bandit
sense -- see docs/notation.md for the round/state distinction). The
repeated, multi-round bandit loop lives in src/simulation/runner.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple


# A strategy is a function f(x, y) -> next_bid.
# By convention (following the paper):
#   - f(0, 0) > 0 means "I open the bidding with this amount"
#     f(0, 0) == 0 means "I let the opponent move first"
#   - f(x, y) == y means "I pass" (I do not top the opponent's bid)
#   - f(x, y) > y means "I raise my bid to this amount"
StrategyFn = Callable[[int, int], int]


@dataclass
class AuctionResult:
    """Outcome of a single dollar auction round."""

    winner: str                 # "agent" or "opponent"
    agent_final_bid: int
    opponent_final_bid: int
    agent_payoff: float
    opponent_payoff: float
    history: List[Tuple[str, int]] = field(default_factory=list)
    # history is a list of (who_bid, bid_amount) pairs in chronological
    # order, e.g. [("agent", 1), ("opponent", 2), ("agent", 3), ...]

    agent_state_trace: List[Tuple[int, int]] = field(default_factory=list)
    # The ordered sequence of states (x, y) at which the AGENT (not the
    # opponent) was asked to decide a bid this round, in chronological
    # order. This is exactly the "(x1,y1), ..., (xn,yn)" sequence used
    # in Lemma 1 (Waniek et al., Section 3.2) to infer the outcome of
    # any strategy g that is a prefix of the agent's actual strategy,
    # without having to play g directly. Required by ELP
    # (src/algorithms/elp.py); ignored by algorithms that don't need
    # side-information (EXP3, EXP3.S).


class DollarAuction:
    """
    A single 2-player, all-pay Dollar Auction.

    Parameters
    ----------
    stake : float
        The value of the prize being auctioned (s in the paper).
    budget : int
        The maximum total amount either player may bid (b in the paper).
        Equal budgets are assumed for both players, as in Waniek et al.
    bid_increment : int
        Minimal bid increment (delta in the paper). Defaults to 1.
    max_rounds_of_bidding : int, optional
        Safety cap on the number of back-and-forth bids inside ONE
        auction, to guard against infinite loops from a misbehaving
        strategy function. Not part of the original formalization --
        purely a defensive implementation detail.
    """

    def __init__(
        self,
        stake: float,
        budget: int,
        bid_increment: int = 1,
        max_rounds_of_bidding: int = 10_000,
    ) -> None:
        self.stake = stake
        self.budget = budget
        self.bid_increment = bid_increment
        self.max_rounds_of_bidding = max_rounds_of_bidding

    def run(
        self,
        agent_strategy: StrategyFn,
        opponent_strategy: StrategyFn,
        agent_starts: Optional[bool] = None,
        rng=None,
    ) -> AuctionResult:
        """
        Run a single auction between `agent_strategy` and
        `opponent_strategy`.

        Both strategies are called as strategy(x, y), where x is the
        last bid made by the player whose turn it is now, and y is the
        last bid made by their opponent. This mirrors f(x, y) in the
        paper exactly.

        Parameters
        ----------
        agent_strategy, opponent_strategy : StrategyFn
            The two players' bidding strategies.
        agent_starts : bool, optional
            If None, the starting player is chosen uniformly at random
            (as in the paper: "the starting player is determined
            randomly at the beginning of the auction"). Pass True/False
            to force a specific starting player (useful for testing).
        rng : numpy.random.Generator, optional
            Random number generator for reproducibility. If None, a
            fresh default generator is used.

        Returns
        -------
        AuctionResult
        """
        import numpy as np

        if rng is None:
            rng = np.random.default_rng()

        if agent_starts is None:
            agent_starts = bool(rng.integers(0, 2))

        # bids[player] = that player's last bid so far
        agent_bid, opponent_bid = 0, 0
        history: List[Tuple[str, int]] = []
        agent_state_trace: List[Tuple[int, int]] = []

        current, other = ("agent", "opponent") if agent_starts else (
            "opponent",
            "agent",
        )

        while True:
            x = agent_bid if current == "agent" else opponent_bid
            y = opponent_bid if current == "agent" else agent_bid

            if current == "agent":
                # Record the state BEFORE the decision, exactly the
                # (xi, yi) sequence Lemma 1 replays over.
                agent_state_trace.append((x, y))

            strategy = agent_strategy if current == "agent" else opponent_strategy
            proposed_bid = strategy(x, y)

            # A strategy "passes" by returning exactly y (no increase).
            passed = proposed_bid <= y

            if not passed:
                # Enforce budget constraint and minimal increment.
                proposed_bid = min(proposed_bid, self.budget)
                if proposed_bid < y + self.bid_increment:
                    # Cannot legally top the opponent -> forced pass.
                    passed = True

            if passed:
                # `current` player passes/folds -> `other` player wins.
                winner = other
                break

            # Record the bid and switch turns.
            if current == "agent":
                agent_bid = proposed_bid
            else:
                opponent_bid = proposed_bid
            history.append((current, proposed_bid))

            current, other = other, current

            if len(history) >= self.max_rounds_of_bidding:
                # Defensive cutoff -- should not trigger under valid
                # (budget-bounded) strategies, but prevents infinite
                # loops during development/testing.
                winner = "agent" if agent_bid >= opponent_bid else "opponent"
                break

        return self._settle(
            winner, agent_bid, opponent_bid, history, agent_state_trace
        )

    def _settle(
        self,
        winner: str,
        agent_bid: int,
        opponent_bid: int,
        history: List[Tuple[str, int]],
        agent_state_trace: List[Tuple[int, int]],
    ) -> AuctionResult:
        """
        Compute all-pay payoffs given the auction outcome.

        Following Section 2.1 of the paper:
            p_i = s - y_i   if y_i > y_j   (i won)
            p_i = -y_i      if y_i < y_j   (i lost)
        Both winner and loser pay their final bid; only the winner
        additionally receives the stake.
        """
        if winner == "agent":
            agent_payoff = self.stake - agent_bid
            opponent_payoff = -opponent_bid
        else:
            agent_payoff = -agent_bid
            opponent_payoff = self.stake - opponent_bid

        return AuctionResult(
            winner=winner,
            agent_final_bid=agent_bid,
            opponent_final_bid=opponent_bid,
            agent_payoff=agent_payoff,
            opponent_payoff=opponent_payoff,
            history=history,
            agent_state_trace=agent_state_trace,
        )

    @staticmethod
    def normalize_reward(payoff: float, budget: int, stake: float) -> float:
        """
        Map a raw payoff p in {-budget+1, ..., stake} to a normalized
        reward in [0, 1], following the paper's convention that bandit
        rewards must lie in [0, 1] (Section 3.1: "we assume that
        always r_f(t) in [0, 1]").

        This uses a simple linear rescaling:
            reward = (payoff + budget) / (budget + stake)
        """
        return (payoff + budget) / (budget + stake)

    @staticmethod
    def binarize_reward(payoff: float) -> int:
        """
        Alternative reward mapping used by some baselines (e.g. the
        d-BayesMAB sketch we critiqued): a binary "success/failure"
        signal Y_t in {0, 1}, where success means a net-positive round
        (payoff > 0). Kept here for baseline comparisons only -- the
        main experiments should prefer `normalize_reward`, since
        binarizing discards payoff magnitude information.
        """
        return 1 if payoff > 0 else 0
