import warnings

import numpy as np

from src.opponents.bob_sunkcost import Bob


def _bob(seed=0, **kw):
    params = dict(stake=10.0, budget=12, window=20)
    params.update(kw)
    return Bob(rng=np.random.default_rng(seed), **params)


def _play_rounds(bob, signals):
    """Step a Bob through rounds, exercising regime draw + strategy RNG."""
    regimes = []
    for s in signals:
        bob.begin_round()
        regimes.append(bob.current_regime)
        strat = bob.get_strategy()
        # drive the within-auction stochastic path a bit
        x = 0
        y = 0
        for _ in range(10):
            nb = strat(x, y)
            if nb <= y:
                break
            x = nb
            y = x + 1
        bob.observe_agent_move(s)
    return regimes


def test_regime_depends_on_history_not_round_index():
    low = _bob(seed=1)
    high = _bob(seed=1)
    r_low = _play_rounds(low, [0.0] * 300)
    r_high = _play_rounds(high, [1.0] * 300)
    # after the window fills, aggressive history => escalation prob 1,
    # passive history => 0
    assert sum(r_low) == 0
    assert sum(r_high[20:]) == len(r_high[20:])


def test_snapshot_does_not_mutate_live_bob():
    live = _bob(seed=2)
    _play_rounds(live, [0.5] * 40)
    before = live.fingerprint()

    snap = live.snapshot()
    _play_rounds(snap, [1.0] * 200)       # heavy use of the clone

    assert live.fingerprint() == before


def test_snapshot_is_faithful_and_deterministic():
    live = _bob(seed=3)
    _play_rounds(live, [0.5] * 40)
    a = live.snapshot()
    b = live.snapshot()
    sig = list(np.random.default_rng(7).random(100))
    assert _play_rounds(a, sig) == _play_rounds(b, sig)
    assert a.fingerprint() == b.fingerprint()


def test_clone_strategy_uses_clone_rng_not_original():
    live = _bob(seed=4)
    _play_rounds(live, [1.0] * 30)
    live.begin_round()
    snap = live.snapshot()
    before = live.fingerprint()
    strat = snap.get_strategy()           # built on the clone
    for _ in range(200):
        strat(6, 7)                        # draws from the clone's RNG
    assert live.fingerprint() == before


def test_min_dwell_holds_regime():
    bob = _bob(seed=5, min_dwell=10, window=5)
    regimes = _play_rounds(bob, [0.5] * 400)
    # regime can only change at multiples of the dwell block
    changes = [i for i in range(1, len(regimes)) if regimes[i] != regimes[i - 1]]
    assert all(i % 10 == 0 for i in changes)


def test_unreachable_sunk_cost_threshold_raises():
    # stake=5, mu=0.8 -> fold at 4.0; old default threshold 5.0 would be a no-op
    try:
        _bob(stake=5.0, sunk_cost_threshold=5.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_ceiling_above_budget_is_clamped():
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        bob = _bob(escalation_ceiling=20.0)
    assert bob.escalation_ceiling == 12.0


def test_escalation_bids_beyond_rational_fold():
    bob = _bob(seed=6, stake=10.0, budget=12)
    bob._escalating = True
    strat = bob.get_strategy()
    # x above sunk-cost threshold, next_bid above fold (8) but under ceiling
    raised = [strat(9, 8) for _ in range(500)]
    assert any(v == 9 for v in raised)     # sometimes continues past fold
    assert any(v == 8 for v in raised)     # sometimes folds


# ---- min_history / fixed_regime / prior_sunk_cost ------------------------

def test_default_waits_for_a_full_window():
    bob = _bob(seed=10, window=10)
    for _ in range(9):
        bob.observe_agent_move(1.0)
    assert bob.p_escalate() == 0.0
    assert bob.begin_round() is False
    bob.observe_agent_move(1.0)               # window now full
    assert bob.p_escalate() == 1.0


def test_min_history_one_reacts_immediately():
    bob = _bob(seed=11, window=10, min_history=1)
    bob.observe_agent_move(1.0)
    assert bob.p_escalate() == 1.0


def test_min_history_must_fit_window():
    for bad in (0, 11):
        try:
            _bob(window=10, min_history=bad)
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_fixed_regime_pins_the_regime():
    esc = _bob(seed=12, fixed_regime=True)
    rat = _bob(seed=12, window=5, fixed_regime=False)
    for _ in range(5):
        rat.observe_agent_move(1.0)           # would normally force escalation
    assert all(esc.begin_round() for _ in range(20))
    assert not any(rat.begin_round() for _ in range(20))


def _contest_bob(prior):
    return Bob(stake=8.0, budget=12, mu=0.5, sunk_cost_threshold=0.5,
               prior_sunk_cost=prior, escalation_rate=50.0,
               escalation_ceiling=12.0, fixed_regime=True,
               rng=np.random.default_rng(0))


def test_prior_sunk_cost_makes_bob_contest_an_opening():
    # fold threshold = 0.5 * 8 = 4; the learner opened with 5.
    plain = _contest_bob(prior=0.0)
    plain.begin_round()
    assert plain.get_strategy()(0, 5) == 5     # not yet hooked -> folds
    hooked = _contest_bob(prior=3.0)
    hooked.begin_round()
    assert hooked.get_strategy()(0, 5) == 6    # arrives invested -> raises


def test_prior_sunk_cost_bob_still_opens_when_first():
    bob = _contest_bob(prior=3.0)
    bob.begin_round()
    assert bob.get_strategy()(0, 0) == 1       # never "skips" the auction


def test_negative_prior_sunk_cost_raises():
    try:
        Bob(stake=8.0, budget=12, prior_sunk_cost=-1.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")