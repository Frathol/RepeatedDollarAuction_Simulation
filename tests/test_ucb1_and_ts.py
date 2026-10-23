"""
test_ucb1_and_ts.py

Unit tests for UCB1 (src/algorithms/ucb1.py) and ThompsonSampling
(src/algorithms/thompson_sampling.py), verifying compliance with the
BanditAlgorithm interface in src/algorithms/base.py as well as mathematical
correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.algorithms.base import BanditAlgorithm
from src.algorithms.thompson_sampling import ThompsonSampling
from src.algorithms.ucb1 import UCB1


# =====================================================================
# 1. INTERFACE & VALIDATION TESTS
# =====================================================================

@pytest.mark.parametrize("algo_cls", [UCB1, ThompsonSampling])
def test_subclass_of_bandit_algorithm(algo_cls) -> None:
    """Both UCB1 and ThompsonSampling must subclass BanditAlgorithm."""
    rng = np.random.default_rng(42)
    agent = algo_cls(n_arms=5, rng=rng)
    assert isinstance(agent, BanditAlgorithm)
    assert agent.n_arms == 5
    assert agent.t == 0
    assert agent.name == algo_cls.__name__


@pytest.mark.parametrize("algo_cls", [UCB1, ThompsonSampling])
def test_invalid_n_arms_raises(algo_cls) -> None:
    """Non-positive n_arms must raise ValueError from BanditAlgorithm."""
    with pytest.raises(ValueError, match="n_arms must be a positive integer"):
        algo_cls(n_arms=0)
    with pytest.raises(ValueError, match="n_arms must be a positive integer"):
        algo_cls(n_arms=-3)


@pytest.mark.parametrize("algo_cls", [UCB1, ThompsonSampling])
def test_update_validates_arm_and_reward_bounds(algo_cls) -> None:
    """Out-of-range arm indices or rewards outside [0, 1] must raise ValueError."""
    agent = algo_cls(n_arms=3, rng=np.random.default_rng(7))
    with pytest.raises(ValueError, match="arm must be in"):
        agent.update(arm=-1, reward=0.5)
    with pytest.raises(ValueError, match="arm must be in"):
        agent.update(arm=3, reward=0.5)
    with pytest.raises(ValueError, match="reward must lie in"):
        agent.update(arm=0, reward=-0.1)
    with pytest.raises(ValueError, match="reward must lie in"):
        agent.update(arm=0, reward=1.2)


@pytest.mark.parametrize("algo_cls", [UCB1, ThompsonSampling])
def test_update_ignores_runner_info_dict(algo_cls) -> None:
    """
    runner.py passes info={'agent_state_trace': ...} on every update() call.
    Plain reward-only algorithms must accept and ignore this argument cleanly.
    """
    agent = algo_cls(n_arms=4, rng=np.random.default_rng(11))
    arm = agent.select_arm()
    dummy_info = {"agent_state_trace": [(0, 0), (1, 2)], "cost": 0.25}
    agent.update(arm=arm, reward=0.6, info=dummy_info)
    assert agent.t == 1


# =====================================================================
# 2. UCB1 MATHEMATICAL & BEHAVIORAL TESTS
# =====================================================================

def test_ucb1_negative_c_raises() -> None:
    """Negative exploration constant c must raise ValueError."""
    with pytest.raises(ValueError, match="Exploration constant c"):
        UCB1(n_arms=3, c=-0.1)


def test_ucb1_initialization_pulls_each_arm_once() -> None:
    """UCB1 must pull each arm sequentially during the first K rounds."""
    n_arms = 6
    agent = UCB1(n_arms=n_arms, rng=np.random.default_rng(42))

    chosen = []
    for r in range(n_arms):
        arm = agent.select_arm()
        chosen.append(arm)
        agent.update(arm=arm, reward=0.1 * r)

    assert chosen == list(range(n_arms))
    assert np.all(agent.counts == 1)
    assert agent.t == n_arms


def test_ucb1_exact_hoeffding_index_calculation() -> None:
    """
    Verify exact UCB1 index values at rounds t = 4 and t = 5 for K = 3 arms
    with initial rewards [0.80, 0.60, 0.20].
    """
    agent = UCB1(n_arms=3, c=1.0, rng=np.random.default_rng(42))

    # Rounds t = 1, 2, 3 (initialization)
    for arm_idx, rew in enumerate([0.80, 0.60, 0.20]):
        assert agent.select_arm() == arm_idx
        agent.update(arm=arm_idx, reward=rew)

    # Decision at round t = 4: bonus = sqrt(2 * ln(4) / 1)
    bonus_t4 = np.sqrt(2.0 * np.log(4.0) / 1.0)
    expected_t4 = np.array([0.80, 0.60, 0.20]) + bonus_t4
    np.testing.assert_allclose(agent.get_ucb_scores(), expected_t4, rtol=1e-12)

    arm_t4 = agent.select_arm()
    assert arm_t4 == 0
    agent.update(arm=arm_t4, reward=0.60)  # Arm 0 mean becomes (0.80 + 0.60)/2 = 0.70

    # Decision at round t = 5: Arm 0 has n_0 = 2, Arm 1 still has n_1 = 1
    expected_t5 = np.array([
        0.70 + np.sqrt(2.0 * np.log(5.0) / 2.0),
        0.60 + np.sqrt(2.0 * np.log(5.0) / 1.0),
        0.20 + np.sqrt(2.0 * np.log(5.0) / 1.0),
    ])
    np.testing.assert_allclose(agent.get_ucb_scores(), expected_t5, rtol=1e-12)
    # Arm 1 must overtake Arm 0 at t = 5 due to higher uncertainty bonus
    assert agent.select_arm() == 1


def test_ucb1_reset_restores_initial_state() -> None:
    """reset() must clear counts, empirical values, and round counter t."""
    agent = UCB1(n_arms=3, rng=np.random.default_rng(99))
    for _ in range(10):
        arm = agent.select_arm()
        agent.update(arm=arm, reward=0.5)

    agent.reset()
    assert agent.t == 0
    assert np.all(agent.counts == 0)
    assert np.all(agent.values == 0.0)
    assert agent.select_arm() == 0


def test_ucb1_converges_on_stationary_bandit() -> None:
    """UCB1 must concentrate pulls on the optimal arm in a stationary setting."""
    env_rng = np.random.default_rng(2026)
    algo_rng = np.random.default_rng(2027)
    true_means = np.array([0.15, 0.45, 0.85])
    agent = UCB1(n_arms=len(true_means), rng=algo_rng)

    n_rounds = 2000
    for _ in range(n_rounds):
        arm = agent.select_arm()
        reward = 1.0 if env_rng.random() < true_means[arm] else 0.0
        agent.update(arm=arm, reward=reward)

    assert int(np.argmax(agent.counts)) == 2
    assert agent.counts[2] > int(0.80 * n_rounds)


# =====================================================================
# 3. THOMPSON SAMPLING MATHEMATICAL & BEHAVIORAL TESTS
# =====================================================================

def test_ts_invalid_prior_raises() -> None:
    """Non-positive alpha_0 or beta_0 must raise ValueError."""
    with pytest.raises(ValueError, match="Prior parameters"):
        ThompsonSampling(n_arms=3, alpha_0=0.0, beta_0=1.0)
    with pytest.raises(ValueError, match="Prior parameters"):
        ThompsonSampling(n_arms=3, alpha_0=1.0, beta_0=-1.0)


def test_ts_binary_reward_updates_deterministically() -> None:
    """Rewards of exactly 1.0 or 0.0 must increment alpha or beta by 1.0."""
    agent = ThompsonSampling(n_arms=2, rng=np.random.default_rng(123))

    agent.update(arm=0, reward=1.0)
    assert agent.alphas[0] == 2.0
    assert agent.betas[0] == 1.0

    agent.update(arm=1, reward=0.0)
    assert agent.alphas[1] == 1.0
    assert agent.betas[1] == 2.0


def test_ts_bernoulli_trial_trick_unbiased_on_continuous_rewards() -> None:
    """
    Verify Agrawal & Goyal (2012) Algorithm 2: for continuous reward r in (0, 1),
    the Bernoulli trial update keeps (alpha + beta) integer-valued and converges
    to E[trial] = r.
    """
    agent = ThompsonSampling(
        n_arms=1, use_bernoulli_trial=True, rng=np.random.default_rng(555)
    )
    continuous_reward = 0.72
    n_trials = 5000

    for _ in range(n_trials):
        agent.update(arm=0, reward=continuous_reward)

    # Total pseudo-counts must increase by exactly 1 per round
    assert np.isclose(agent.alphas[0] + agent.betas[0], 2.0 + n_trials)
    # Empirical success rate of Bernoulli trials must match 0.72 within tolerance
    realized_rate = (agent.alphas[0] - 1.0) / n_trials
    assert abs(realized_rate - continuous_reward) < 0.02
    assert np.isclose(agent.values[0], continuous_reward)


def test_ts_reproducibility_with_seeded_rng_and_reset() -> None:
    """Two ThompsonSampling instances given identically seeded RNGs must match."""
    agent_a = ThompsonSampling(n_arms=4, rng=np.random.default_rng(888))
    agent_b = ThompsonSampling(n_arms=4, rng=np.random.default_rng(888))

    arms_a, arms_b = [], []
    for step in range(40):
        a = agent_a.select_arm()
        b = agent_b.select_arm()
        rew = (step % 5) / 4.0
        agent_a.update(arm=a, reward=rew)
        agent_b.update(arm=b, reward=rew)
        arms_a.append(a)
        arms_b.append(b)

    assert arms_a == arms_b
    np.testing.assert_array_equal(agent_a.alphas, agent_b.alphas)
    np.testing.assert_array_equal(agent_a.betas, agent_b.betas)

    agent_a.reset()
    assert agent_a.t == 0
    np.testing.assert_array_equal(agent_a.alphas, np.ones(4))
    np.testing.assert_array_equal(agent_a.betas, np.ones(4))


def test_ts_converges_on_continuous_rewards() -> None:
    """ThompsonSampling must concentrate pulls on the optimal arm."""
    env_rng = np.random.default_rng(301)
    algo_rng = np.random.default_rng(302)
    true_means = np.array([0.10, 0.40, 0.85])
    agent = ThompsonSampling(n_arms=len(true_means), rng=algo_rng)

    n_rounds = 2000
    for _ in range(n_rounds):
        arm = agent.select_arm()
        noisy_reward = float(
            np.clip(env_rng.normal(true_means[arm], 0.05), 0.0, 1.0)
        )
        agent.update(arm=arm, reward=noisy_reward)

    assert int(np.argmax(agent.counts)) == 2
    assert agent.counts[2] > int(0.85 * n_rounds)