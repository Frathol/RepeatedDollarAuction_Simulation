import numpy as np

from src.algorithms.base import BanditAlgorithm
from src.simulation.config import SimConfig
from src.simulation.runner import make_simulation


class FixedArm(BanditAlgorithm):
    """Always plays one arm (to probe Bob's reaction to a fixed behaviour)."""

    def __init__(self, n_arms, arm, rng=None):
        super().__init__(n_arms, rng)
        self.arm = arm

    def select_arm(self):
        return self.arm

    def update(self, arm, reward, info=None):
        self.t += 1

    def reset(self):
        self.t = 0


def _cfg(**kw):
    base = dict(budget=12, stake=10.0, T=300, seed=0, bob_window=20)
    base.update(kw)
    return SimConfig(**base)


def test_records_schema_and_reward_range():
    sim = make_simulation(_cfg(), "exp3")
    recs = sim.run()
    assert len(recs) == 300
    for key in ("round", "arm", "theta", "bob_regime", "agent_starts",
                "winner", "reward", "winning_bid", "ph_detected"):
        assert key in recs[0]
    assert all(0.0 <= r["reward"] <= 1.0 for r in recs)
    assert [r["round"] for r in recs] == list(range(1, 301))


def test_elp_runs_end_to_end():
    recs = make_simulation(_cfg(T=200), "elp").run()
    assert len(recs) == 200


def test_same_seed_is_reproducible_different_seed_differs():
    a = make_simulation(_cfg(seed=5), "exp3").run()
    b = make_simulation(_cfg(seed=5), "exp3").run()
    c = make_simulation(_cfg(seed=6), "exp3").run()
    assert [r["arm"] for r in a] == [r["arm"] for r in b]
    assert [r["bob_regime"] for r in a] == [r["bob_regime"] for r in b]
    assert [r["arm"] for r in a] != [r["arm"] for r in c]


def test_random_streams_are_independent():
    # Changing Bob's parameters must not change the starter sequence
    # (env stream) nor the learner's first draw (learner stream).
    a = make_simulation(_cfg(seed=7, bob_window=5), "exp3").run()
    b = make_simulation(_cfg(seed=7, bob_window=40, bob_gain=0.5), "exp3").run()
    assert [r["agent_starts"] for r in a] == [r["agent_starts"] for r in b]
    assert a[0]["arm"] == b[0]["arm"]


def test_bob_reacts_to_learner_behaviour():
    cfg = _cfg(T=400, seed=1)
    n_arms = cfg.budget + 1
    base = make_simulation(cfg, "exp3")
    passive = make_simulation(cfg, FixedArm(n_arms, arm=1))
    aggressive = make_simulation(cfg, FixedArm(n_arms, arm=cfg.budget))
    for sim in (passive, aggressive):
        sim.run()
    frac = lambda s: np.mean([r["bob_escalating"] for r in s.records[50:]])
    assert frac(passive) < 0.2
    assert frac(aggressive) > 0.8


def test_phase_start_snapshots_reproduce_logged_regime():
    # Strong check that snapshots are faithful: stepping the stored
    # pre-round snapshot's begin_round() must give the logged regime.
    sim = make_simulation(_cfg(T=400, seed=2, bob_min_dwell=15), "exp3")
    sim.run()
    assert sim.phase_starts[0][0] == 1
    for t, snap in sim.phase_starts:
        assert snap.begin_round() == sim.records[t - 1]["bob_escalating"]


def test_stepping_snapshots_does_not_touch_live_bob():
    sim = make_simulation(_cfg(T=200, seed=3), "exp3")
    sim.run()
    before = sim.bob.fingerprint()
    for _, snap in sim.phase_starts:
        for _ in range(50):
            snap.begin_round()
            snap.get_strategy()(5, 6)
            snap.observe_agent_move(1.0)
    assert sim.bob.fingerprint() == before


def test_min_dwell_gives_persistent_phases():
    sim = make_simulation(_cfg(T=600, seed=4, bob_min_dwell=25,
                               bob_window=10), "exp3")
    sim.run()
    log = [r["bob_escalating"] for r in sim.records]
    changes = [i for i in range(1, len(log)) if log[i] != log[i - 1]]
    assert all(c % 25 == 0 for c in changes)


def test_ph_with_exp3_runs_and_flags_match_detector():
    cfg = _cfg(T=600, seed=8, ph_threshold=1.0)
    sim = make_simulation(cfg, "exp3", use_ph=True)
    recs = sim.run()
    flagged = [r["round"] for r in recs if r["ph_detected"]]
    assert flagged == sim.detector.detection_rounds


def test_ph_with_elp_is_rejected_until_elp_has_reset():
    try:
        make_simulation(_cfg(), "elp", use_ph=True)
    except ValueError:
        return
    raise AssertionError("expected ValueError")