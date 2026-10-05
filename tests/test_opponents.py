"""
test_opponents.py

Sanity checks for src/opponents/alice_rational.py (oblivious rational
opponent, Stage 0) and src/opponents/bob_sunkcost.py (adaptive adversary).

History: the old Bob tests (mode="single_switch", switch_round,
is_escalating_at, get_strategy(t=...)) were removed together with the
scripted-schedule Bob. Ground truth for Bob's regime is now the runner's
realized regime log (design_decisions sec. 15); regime/history logic is
covered in tests/test_bob_sunkcost.py and tests/test_runner.py. The tests
below only cover within-auction behaviour of the CURRENT Bob API.
"""

import numpy as np
import pytest

from src.opponents.alice_rational import Alice
from src.opponents.bob_sunkcost import Bob


def _escalating_bob(**kw):
    """Bob forced into the escalating regime through the public API:
    a window full of maximal aggressiveness => p_escalate = 1."""
    params = dict(stake=10.0, budget=12, window=5, rng=np.random.default_rng(0))
    params.update(kw)
    bob = Bob(**params)
    for _ in range(5):
        bob.observe_agent_move(1.0)
    assert bob.begin_round() is True
    return bob


def test_alice_folds_beyond_threshold():
    alice = Alice(stake=10.0, budget=8, mu=0.8)  # fold_threshold = 8.0
    strat = alice.get_strategy()
    assert strat(0, 7) == 8      # next bid 8 is not > 8.0 -> raise
    assert strat(0, 8) == 8      # next bid 9 > 8.0 -> fold


def test_bob_fresh_has_no_history_and_is_rational():
    bob = Bob(stake=10.0, budget=8, rng=np.random.default_rng(0))
    assert bob.aggressiveness_score() == 0.0
    assert bob.p_escalate() == 0.0
    assert bob.begin_round() is False
    assert bob.regime_label == "Rational"


def test_bob_rational_regime_matches_alice_like_behaviour():
    bob = Bob(stake=10.0, budget=8, rng=np.random.default_rng(0))
    bob.begin_round()                     # empty history -> rational
    strat = bob.get_strategy()
    assert strat(0, 7) == 8
    assert strat(0, 8) == 8               # folds, same as Alice


def test_bob_escalating_regime_label():
    bob = _escalating_bob()
    assert bob.regime_label == "Escalating"


def test_bob_escalation_never_exceeds_ceiling():
    bob = _escalating_bob(sunk_cost_threshold=2.0, escalation_rate=1.0,
                          escalation_ceiling=10.0)
    strat = bob.get_strategy()
    for _ in range(300):
        assert strat(10, 10) == 10        # next bid 11 > ceiling -> always folds
        assert strat(9, 9) in (9, 10)     # may continue up to the ceiling only


def test_bob_ceiling_cannot_exceed_budget():
    with pytest.warns(UserWarning):
        bob = Bob(stake=10.0, budget=8, escalation_ceiling=15.0,
                  rng=np.random.default_rng(0))
    assert bob.escalation_ceiling == 8.0