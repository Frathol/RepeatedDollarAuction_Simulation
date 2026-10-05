import numpy as np

from src.simulation.calibration import arm_rewards, opening_bid, separation_score


def test_opening_bid_matches_oneill_formula():
    assert opening_bid(12, 5) == 4
    assert opening_bid(12, 8) == 5
    assert opening_bid(14, 10) == 5


def test_separation_score_known_values():
    r_rat = np.array([0.6, 0.8, 0.8])
    r_esc = np.array([0.7, 0.5, 0.5])
    s = separation_score(r_rat, r_esc)
    assert s["best_rational"] == [1, 2]
    assert s["best_escalating"] == [0]
    assert abs(s["A"] - 0.2) < 1e-9 and abs(s["B"] - 0.2) < 1e-9
    assert abs(s["score"] - 0.2) < 1e-9


def test_separation_score_is_zero_when_regimes_agree():
    r = np.array([0.5, 0.7, 0.7])
    assert separation_score(r, r.copy())["score"] == 0.0


def _kw(prior):
    return dict(mu=0.5, sunk_cost_threshold=0.5, prior_sunk_cost=prior,
                escalation_rate=3.0, escalation_ceiling=12.0)


def test_prior_sunk_cost_separates_the_regimes_and_zero_does_not():
    good = separation_score(
        arm_rewards(12, 8.0, _kw(3.0), False, T=150, seeds=2),
        arm_rewards(12, 8.0, _kw(3.0), True, T=150, seeds=2))
    assert good["score"] > 0.10
    assert good["best_escalating"] == [0]      # when Bob escalates: do not engage
    assert 0 not in good["best_rational"]      # when Bob is rational: engage

    old = separation_score(
        arm_rewards(12, 8.0, _kw(0.0), False, T=150, seeds=2),
        arm_rewards(12, 8.0, _kw(0.0), True, T=150, seeds=2))
    # the original Bob (no prior investment) separates the regimes far less
    assert good["score"] > old["score"] + 0.05