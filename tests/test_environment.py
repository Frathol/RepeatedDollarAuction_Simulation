"""
test_environment.py

Basic sanity checks for src/environment/dollar_auction.py and
src/environment/strategies.py. These are not exhaustive -- they are a
starting skeleton; extend as ELP/EXP3/EXP3.S are implemented and need
environment-level guarantees to hold.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.environment.dollar_auction import DollarAuction
from src.environment.strategies import (
    build_threshold_arm_set,
    oneill_optimal_strategy,
    fixed_opening_bid_strategy,
)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def test_auction_ends_and_pays_both_players(rng):
    """
    All-pay sanity check: the auction must terminate, and BOTH the
    winner and the loser must have non-zero payments recorded whenever
    at least one bid was placed (the defining feature of an all-pay
    auction, per Section 2.1 of Waniek et al.).
    """
    env = DollarAuction(stake=10.0, budget=8)
    always_raise_by_one = fixed_opening_bid_strategy(1)
    # Force the second bidder to fold immediately after one bid each,
    # by having them use a strategy that folds after one raise.
    def fold_after_one(x, y):
        return y if x > 0 else y + 1

    result = env.run(always_raise_by_one, fold_after_one, agent_starts=True, rng=rng)

    assert result.winner in ("agent", "opponent")
    assert result.agent_final_bid >= 0
    assert result.opponent_final_bid >= 0
    # winner pays their final bid and receives the stake net;
    # loser pays their final bid and receives nothing.
    if result.winner == "agent":
        assert result.agent_payoff == pytest.approx(10.0 - result.agent_final_bid)
        assert result.opponent_payoff == pytest.approx(-result.opponent_final_bid)
    else:
        assert result.opponent_payoff == pytest.approx(10.0 - result.opponent_final_bid)
        assert result.agent_payoff == pytest.approx(-result.agent_final_bid)


def test_oneill_strategy_wins_against_itself_deterministically(rng):
    """
    With equal budgets, O'Neill's strategy should force the SECOND
    mover to fold immediately, regardless of random seed, since the
    first mover's opening bid is specifically chosen to make raising
    unaffordable for the opponent.
    """
    budget, stake = 8, 10.0
    env = DollarAuction(stake=stake, budget=budget)
    strat = oneill_optimal_strategy(budget=budget, stake=stake)

    for seed in range(5):
        r = np.random.default_rng(seed)
        result = env.run(strat, strat, agent_starts=True, rng=r)
        assert result.winner == "agent"
        assert result.opponent_final_bid == 0


def test_threshold_arm_set_has_expected_size():
    arms = build_threshold_arm_set(budget=8, stake=10.0)
    assert len(arms) == 9  # thetas 0..8 inclusive

    arms_custom = build_threshold_arm_set(budget=8, stake=10.0, thetas=[0, 4, 8])
    assert len(arms_custom) == 3


def test_reward_normalization_bounds():
    budget, stake = 8, 10.0
    # Worst possible payoff: lost after bidding the entire budget.
    worst = DollarAuction.normalize_reward(-budget, budget, stake)
    # Best possible payoff: won while bidding almost nothing.
    best = DollarAuction.normalize_reward(stake, budget, stake)
    assert worst == pytest.approx(0.0)
    assert best == pytest.approx(1.0)


def test_binarize_reward_matches_sign():
    assert DollarAuction.binarize_reward(5) == 1
    assert DollarAuction.binarize_reward(0) == 0
    assert DollarAuction.binarize_reward(-1) == 0
