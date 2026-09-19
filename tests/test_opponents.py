"""
test_opponents.py

Sanity checks for src/opponents/alice_rational.py and
src/opponents/bob_sunkcost.py, focused on the REGIME logic (Bob's
is_escalating_at), since that is the ground truth used later by the
switching-regret and detection-delay metrics -- it must be exactly
right.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.opponents.alice_rational import Alice
from src.opponents.bob_sunkcost import Bob


def test_alice_folds_beyond_threshold():
    alice = Alice(stake=10.0, budget=8, mu=0.8)  # fold_threshold = 8.0
    strat = alice.get_strategy()
    # At y=7, next bid would be 8, which is NOT > 8.0 -> should raise.
    assert strat(0, 7) == 8
    # At y=8, next bid would be 9, which IS > 8.0 -> should fold (pass).
    assert strat(0, 8) == 8


def test_bob_single_switch_regime_boundary():
    bob = Bob(stake=10.0, budget=8, mode="single_switch", switch_round=1000)
    assert bob.is_escalating_at(1000) is False
    assert bob.is_escalating_at(1001) is True
    assert bob.is_escalating_at(9999) is True


def test_bob_recurring_switch_regime_toggles():
    bob = Bob(
        stake=10.0,
        budget=8,
        mode="recurring_switch",
        switch_points=[1000, 3000, 4000],
    )
    # regime sequence: rational | [1000] escalating | [3000] rational
    #                  | [4000] escalating
    assert bob.is_escalating_at(500) is False
    assert bob.is_escalating_at(1000) is False  # boundary is exclusive
    assert bob.is_escalating_at(1001) is True
    assert bob.is_escalating_at(2999) is True
    assert bob.is_escalating_at(3001) is False
    assert bob.is_escalating_at(4001) is True


def test_bob_recurring_switch_requires_switch_points():
    with pytest.raises(ValueError):
        Bob(stake=10.0, budget=8, mode="recurring_switch", switch_points=None)


def test_bob_rational_phase_matches_alice_like_behaviour():
    rng = np.random.default_rng(0)
    bob = Bob(stake=10.0, budget=8, mode="single_switch", switch_round=1000, rng=rng)
    strat = bob.get_strategy(t=1)  # rational phase
    assert strat(0, 7) == 8
    assert strat(0, 8) == 8  # folds


def test_bob_escalation_never_exceeds_ceiling():
    rng = np.random.default_rng(1)
    bob = Bob(
        stake=10.0,
        budget=8,
        mode="single_switch",
        switch_round=0,  # escalating from round 1 onward
        sunk_cost_threshold=2.0,
        escalation_rate=1.0,
        escalation_ceiling=15.0,
        rng=rng,
    )
    strat = bob.get_strategy(t=1)
    # Simulate deep into sunk-cost territory -- bid should never
    # propose beyond the ceiling regardless of RNG draws.
    for _ in range(200):
        next_bid = strat(14, 14)
        assert next_bid <= 15
