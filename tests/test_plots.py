import matplotlib

matplotlib.use("Agg")

import numpy as np

from src.metrics.plots import RunBundle, moving_average, save_standard_report


def _bundle(with_regime=False, with_bound=False, S=3, T=200, K=5, seed=0):
    rng = np.random.default_rng(seed)
    probs = rng.random((S, K))
    probs /= probs.sum(axis=1, keepdims=True)
    return RunBundle(
        label="synthetic", thetas=list(range(K)),
        reward=rng.random((S, T)),
        regret=np.cumsum(rng.random((S, T)) * 0.1, axis=1),
        arm_theta=rng.integers(0, K, (S, T)),
        agent_won=rng.random((S, T)) > 0.5,
        agent_bid=rng.integers(0, 10, (S, T)).astype(float),
        opp_bid=rng.integers(0, 10, (S, T)).astype(float),
        final_probs=probs,
        hindsight_per_arm=rng.random((S, K)) * T,
        bound=(np.arange(1, T + 1) * 0.5 + 5) if with_bound else None,
        regime=(rng.random((S, T)) > 0.5) if with_regime else None,
        alarms=[[50, 120], [60], []] if with_regime else None,
    )


def test_moving_average_matches_naive():
    x = np.random.default_rng(1).random((2, 30))
    w = 5
    got = moving_average(x, w)
    for t in range(30):
        lo = max(0, t - w + 1)
        assert np.allclose(got[:, t], x[:, lo:t + 1].mean(axis=1))


def test_standard_report_baseline_files(tmp_path):
    paths = save_standard_report(_bundle(with_bound=True), tmp_path)
    names = {p.name for p in paths}
    assert names == {"regret_cumulative.png", "regret_normalized.png",
                     "moving_avg_reward.png", "arm_distribution.png",
                     "arm_heatmap.png", "outcomes.png", "regret_vs_bound.png"}
    assert all(p.exists() and p.stat().st_size > 0 for p in paths)


def test_later_stage_layers_reuse_the_same_figure_set(tmp_path):
    # Stage 1+ only fills optional fields; the file set stays the same.
    paths = save_standard_report(_bundle(with_regime=True), tmp_path)
    assert {p.name for p in paths} == {
        "regret_cumulative.png", "regret_normalized.png",
        "moving_avg_reward.png", "arm_distribution.png",
        "arm_heatmap.png", "outcomes.png"}