"""
run_stage1.py

Stage 1 -- Adaptive Opponent (Bob) Evaluation via Mesa Orchestration:
Compares all four non-BwK bandit algorithms (ELP, EXP3, UCB1, ThompsonSampling)
against the memory-driven, reactive opponent Bob (`bob_sunkcost.py`).

Evaluates both:
1. Counterfactual Policy Regret (Arora et al., 2012): benchmarked against the
   best fixed threshold arm theta* when Bob reacts counterfactually to theta*
   from round 1 to T.
2. Weak / Realized-Trajectory Regret: benchmarked against the best fixed arm
   evaluated passively on the learner's realized sequence of Bob regimes.

Outputs per configuration (results/stage1/<label>/):
    tables/algorithm_summary.csv
    tables/checkpoint_policy_regret.csv
    tables/per_theta_selection_prob.csv
    figures/plot_1_policy_vs_weak_regret_<label>.png
    figures/plot_2_reward_and_bob_regime_<label>.png
    figures/plot_3_arm_distribution_vs_regimes_<label>.png
    figures/plot_4_stage1_summary_bars_<label>.png

Run from the project root:
    python -m experiments.stage1_adaptive_bob.run_stage1
    python -m experiments.stage1_adaptive_bob.run_stage1 --quick
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.base import BanditAlgorithm
from src.environment.dollar_auction import DollarAuction
from src.environment.strategies import build_threshold_arm_set
from src.opponents.bob_sunkcost import Bob
from src.simulation.calibration import arm_rewards
from src.simulation.config import SimConfig, stage1_config
from src.simulation.runner import make_simulation, run_with_mesa

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it=None, **kwargs):
        return it
    tqdm.write = print


STAGE1_CONFIGS = [
    {
        "label": "recommended_b12_s8_mem25",
        "overrides": dict(
            budget=12,
            stake=8.0,
            T=10000,
            bob_mu=0.5,
            bob_sunk_cost_threshold=0.5,
            bob_prior_sunk_cost=3.0,
            bob_escalation_rate=3.0,
            bob_window=25,
            bob_min_dwell=25,
            bob_gain=1.0,
        ),
        "n_seeds": 8,
    },
    {
        "label": "control_memoryless_b12_s8",
        "overrides": dict(
            budget=12,
            stake=8.0,
            T=10000,
            bob_mu=0.5,
            bob_sunk_cost_threshold=0.5,
            bob_prior_sunk_cost=3.0,
            bob_escalation_rate=3.0,
            bob_window=1,
            bob_min_dwell=1,
            bob_gain=0.0,  # gain=0 => Bob does not react to learner history
        ),
        "n_seeds": 8,
    },
]

ALGO_ORDER = ["ELP", "EXP3", "UCB1", "TS"]

ALGO_STYLE: dict[str, dict[str, Any]] = {
    "ELP": {
        "runner_key": "elp",
        "label": "ELP (Waniek et al., 2016)",
        "color": "tab:blue",
        "linestyle": "-",
        "marker": "|",
    },
    "EXP3": {
        "runner_key": "exp3",
        "label": "EXP3 (Adversarial Baseline)",
        "color": "tab:orange",
        "linestyle": "--",
        "marker": "|",
    },
    "UCB1": {
        "runner_key": "ucb1",
        "label": "UCB1 (Stochastic Hoeffding)",
        "color": "tab:green",
        "linestyle": "-.",
        "marker": "|",
    },
    "TS": {
        "runner_key": "ts",
        "label": "Thompson Sampling (Beta-Bernoulli)",
        "color": "tab:red",
        "linestyle": ":",
        "marker": "|",
    },
}

CHECKPOINT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


# =====================================================================
# 1. HELPER CLASSES & COUNTERFACTUAL / WEAK REGRET ENGINE
# =====================================================================

class FixedArmAgent(BanditAlgorithm):
    """
    Deterministic agent that always pulls `fixed_arm`.
    Used to evaluate exact counterfactual trajectories against reactive Bob
    for Policy Regret calculation (Arora et al., 2012).
    """

    def __init__(self, n_arms: int, fixed_arm: int, rng=None) -> None:
        super().__init__(n_arms=n_arms, rng=rng)
        self.fixed_arm = int(fixed_arm)

    def select_arm(self) -> int:
        return self.fixed_arm

    def update(self, arm: int, reward: float, info: dict | None = None) -> None:
        self.t += 1

    def reset(self) -> None:
        self.t = 0


def get_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute O(T) rolling mean of a 1D array using cumulative sums."""
    window = max(1, int(window))
    cum_sum = np.cumsum(arr, dtype=np.float64)
    roll_mean = np.empty_like(cum_sum)
    for i in range(min(window, len(arr))):
        roll_mean[i] = cum_sum[i] / (i + 1)
    if len(arr) > window:
        roll_mean[window:] = (cum_sum[window:] - cum_sum[:-window]) / window
    return roll_mean


def compute_counterfactual_arm_rewards(cfg: SimConfig) -> np.ndarray:
    """
    Simulate playing each fixed threshold arm theta in {0, ..., budget} from
    round 1 to T against a fresh reactive Bob initialized with `cfg.seed`.

    Returns
    -------
    numpy.ndarray
        Array of shape `(n_arms, T)` containing per-round normalized rewards
        earned by each fixed arm along its own counterfactual trajectory.
    """
    n_arms = cfg.budget + 1
    cf_rewards = np.zeros((n_arms, cfg.T), dtype=np.float64)

    for arm_idx in range(n_arms):
        fixed_cfg = stage1_config(**cfg.__dict__)
        fixed_cfg.keep_snapshots = False
        fixed_agent = FixedArmAgent(n_arms=n_arms, fixed_arm=arm_idx)
        sim = make_simulation(fixed_cfg, algorithm=fixed_agent, use_ph=False)
        recs = sim.run(progress=False)
        cf_rewards[arm_idx, :] = [r["reward"] for r in recs]

    return cf_rewards


def compute_weak_hindsight_rewards(
    cfg: SimConfig, records: list[dict[str, Any]]
) -> np.ndarray:
    """
    Compute realized-regime (Weak Regret) per-round rewards for all candidate
    arms given the sequence of Bob's realized regimes (`bob_escalating`) and
    starting players (`agent_starts`) recorded during a learner's run.

    Returns
    -------
    numpy.ndarray
        Array of shape `(n_arms, T)` with per-round rewards under the learner's
        realized regime trajectory.
    """
    thetas = list(range(0, cfg.budget + 1))
    arms = build_threshold_arm_set(cfg.budget, cfg.stake, thetas=thetas)
    n_arms = len(arms)
    T = len(records)

    env = DollarAuction(stake=cfg.stake, budget=cfg.budget)

    # Pre-build Bob's two fixed-regime strategies (Rational and Escalating)
    bob_rat = Bob(
        stake=cfg.stake,
        budget=cfg.budget,
        mu=cfg.bob_mu,
        sunk_cost_threshold=cfg.bob_sunk_cost_threshold,
        prior_sunk_cost=cfg.bob_prior_sunk_cost,
        escalation_rate=cfg.bob_escalation_rate,
        fixed_regime=False,
        escalation_ceiling=cfg.bob_escalation_ceiling,
        rng=np.random.default_rng(cfg.seed + 99991),
    )
    bob_esc = Bob(
        stake=cfg.stake,
        budget=cfg.budget,
        mu=cfg.bob_mu,
        sunk_cost_threshold=cfg.bob_sunk_cost_threshold,
        prior_sunk_cost=cfg.bob_prior_sunk_cost,
        escalation_rate=cfg.bob_escalation_rate,
        fixed_regime=True,
        escalation_ceiling=cfg.bob_escalation_ceiling,
        rng=np.random.default_rng(cfg.seed + 99992),
    )
    bob_rat.begin_round()
    bob_esc.begin_round()
    strat_rat = bob_rat.get_strategy()
    strat_esc = bob_esc.get_strategy()

    # Pre-compute lookup table: (is_escalating, agent_starts, arm_idx) -> norm_reward
    lookup = np.zeros((2, 2, n_arms), dtype=np.float64)
    dummy_rng = np.random.default_rng(0)
    for esc_flag, b_strat in [(0, strat_rat), (1, strat_esc)]:
        for start_flag in (0, 1):
            for i, a_strat in enumerate(arms):
                res = env.run(
                    a_strat,
                    b_strat,
                    agent_starts=bool(start_flag),
                    rng=dummy_rng,
                )
                lookup[esc_flag, start_flag, i] = DollarAuction.normalize_reward(
                    res.agent_payoff, cfg.budget, cfg.stake
                )

    weak_rewards = np.zeros((n_arms, T), dtype=np.float64)
    for t_idx, rec in enumerate(records):
        esc_i = 1 if rec["bob_escalating"] else 0
        start_i = 1 if rec["agent_starts"] else 0
        weak_rewards[:, t_idx] = lookup[esc_i, start_i, :]

    return weak_rewards


# =====================================================================
# 2. SINGLE-SEED RUNNER VIA MESA
# =====================================================================

def run_single_seed_stage1(base_cfg: SimConfig, seed: int) -> dict[str, Any]:
    """
    Run all 4 non-BwK algorithms via `run_with_mesa` on a single seed against
    adaptive Bob, and compute both Policy Regret and Weak Regret.
    """
    cfg = stage1_config(**base_cfg.__dict__)
    cfg.seed = seed
    T = cfg.T
    n_arms = cfg.budget + 1
    thetas = list(range(0, n_arms))

    # 1. Counterfactual benchmark for Policy Regret on this seed
    cf_rewards = compute_counterfactual_arm_rewards(cfg)  # (n_arms, T)
    cf_cum = np.cumsum(cf_rewards, axis=1)                # (n_arms, T)
    best_cf_cum_so_far = cf_cum.max(axis=0)               # (T,)
    cf_total_per_arm = cf_cum[:, -1]                      # (n_arms,)

    algo_results: dict[str, dict[str, Any]] = {}

    for algo_key in ALGO_ORDER:
        runner_key = ALGO_STYLE[algo_key]["runner_key"]
        algo_cfg = stage1_config(**cfg.__dict__)

        # Execute through Mesa orchestration layer
        model, model_df, _ = run_with_mesa(
            algo_cfg, algorithm=runner_key, use_ph=False, progress=False
        )
        recs = model.sim.records

        rew = np.array([r["reward"] for r in recs], dtype=np.float64)
        cum_rew = np.cumsum(rew)
        theta_arr = np.array([r["theta"] for r in recs], dtype=np.int64)
        esc_arr = np.array([r["bob_escalating"] for r in recs], dtype=bool)
        p_esc_arr = np.array([r["p_escalate"] for r in recs], dtype=np.float64)
        won_arr = np.array([r["winner"] == "agent" for r in recs], dtype=bool)
        abid_arr = np.array([r["agent_final_bid"] for r in recs], dtype=np.float64)
        obid_arr = np.array([r["opponent_final_bid"] for r in recs], dtype=np.float64)

        # Weak Regret against realized Bob regime trajectory
        weak_arm_rews = compute_weak_hindsight_rewards(algo_cfg, recs)
        weak_cum = np.cumsum(weak_arm_rews, axis=1)
        best_weak_cum_so_far = weak_cum.max(axis=0)

        policy_regret = best_cf_cum_so_far - cum_rew
        weak_regret = best_weak_cum_so_far - cum_rew

        pull_counts = np.bincount(theta_arr, minlength=n_arms)
        emp_probs = pull_counts.astype(np.float64) / float(T)

        algo_results[algo_key] = {
            "reward": rew,
            "policy_regret": policy_regret,
            "weak_regret": weak_regret,
            "arm_theta": theta_arr,
            "bob_escalating": esc_arr,
            "p_escalate": p_esc_arr,
            "agent_won": won_arr,
            "agent_bid": abid_arr,
            "opp_bid": obid_arr,
            "empirical_probs": emp_probs,
            "mesa_rows": len(model_df),
        }

    return {
        "algos": algo_results,
        "cf_total_per_arm": cf_total_per_arm,
        "thetas": thetas,
    }


# =====================================================================
# 3. PUBLICATION-GRADE PLOTTING SUITE FOR STAGE 1
# =====================================================================

def save_stage1_publication_plots(
    label: str,
    base_cfg: SimConfig,
    agg: dict[str, dict[str, np.ndarray]],
    cf_totals: np.ndarray,
    r_rat: np.ndarray,
    r_esc: np.ndarray,
    thetas: list[int],
    fig_dir: Path,
) -> list[Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    T = base_cfg.T
    budget = base_cfg.budget
    stake = base_cfg.stake
    t_steps = np.arange(1, T + 1)
    mark_interval = max(1, T // 10)
    window_size = max(20, min(200, T // 15))

    mean_cf_per_round = cf_totals.mean(axis=0) / float(T)
    opt_cf_idx = int(np.argmax(mean_cf_per_round))
    opt_cf_theta = thetas[opt_cf_idx]

    # -----------------------------------------------------------------
    # PLOT 1: Counterfactual Policy Regret vs. Weak Regret (2 Panels)
    # -----------------------------------------------------------------
    fig, (ax_pol, ax_weak) = plt.subplots(1, 2, figsize=(11.5, 5.0), dpi=180)

    for algo_key in ALGO_ORDER:
        style = ALGO_STYLE[algo_key]
        pol_mat = agg[algo_key]["policy_regret"]
        weak_mat = agg[algo_key]["weak_regret"]

        m_pol, s_pol = pol_mat.mean(axis=0), pol_mat.std(axis=0)
        m_wk, s_wk = weak_mat.mean(axis=0), weak_mat.std(axis=0)

        ax_pol.plot(
            t_steps, m_pol,
            label=f"{algo_key} ($R_T^{{\\mathrm{{pol}}}}={m_pol[-1]:.1f}$)",
            color=style["color"], linestyle=style["linestyle"], linewidth=2.0,
            marker=style["marker"], markevery=mark_interval, markersize=8,
        )
        ax_pol.fill_between(
            t_steps, m_pol - s_pol, m_pol + s_pol, color=style["color"], alpha=0.14
        )

        ax_weak.plot(
            t_steps, m_wk,
            label=f"{algo_key} ($R_T^{{\\mathrm{{weak}}}}={m_wk[-1]:.1f}$)",
            color=style["color"], linestyle=style["linestyle"], linewidth=2.0,
            marker=style["marker"], markevery=mark_interval, markersize=8,
        )
        ax_weak.fill_between(
            t_steps, m_wk - s_wk, m_wk + s_wk, color=style["color"], alpha=0.14
        )

    ax_pol.set_title("Counterfactual Policy Regret (Arora et al., 2012)", fontsize=11, fontweight="bold")
    ax_pol.set_xlabel(r"Time Horizon ($T$)", fontsize=10.5)
    ax_pol.set_ylabel(r"Cumulative Policy Regret ($R_T^{\mathrm{pol}}$)", fontsize=10.5)
    ax_pol.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax_pol.grid(True, linestyle=":", alpha=0.6)
    ax_pol.legend(loc="upper left", frameon=True, fontsize=9.0)

    ax_weak.set_title("Weak Regret (Realized-Regime Hindsight)", fontsize=11, fontweight="bold")
    ax_weak.set_xlabel(r"Time Horizon ($T$)", fontsize=10.5)
    ax_weak.set_ylabel(r"Cumulative Weak Regret ($R_T^{\mathrm{weak}}$)", fontsize=10.5)
    ax_weak.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax_weak.grid(True, linestyle=":", alpha=0.6)
    ax_weak.legend(loc="upper left", frameon=True, fontsize=9.0)

    fig.suptitle(
        f"Stage 1 Regret Comparison vs. Adaptive Bob — {label} ($b={budget}, s={stake:g}$)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    p1 = fig_dir / f"plot_1_policy_vs_weak_regret_{label}.png"
    fig.savefig(p1, bbox_inches="tight")
    plt.close(fig)
    saved.append(p1)

    # -----------------------------------------------------------------
    # PLOT 2: Rolling Reward (Top) + Bob Escalation Rate (Bottom)
    # -----------------------------------------------------------------
    fig, (ax_rew, ax_esc) = plt.subplots(
        2, 1, figsize=(9.5, 6.8), dpi=180, sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
    )

    ax_rew.axhline(
        float(mean_cf_per_round[opt_cf_idx]),
        color="black", linestyle="--", linewidth=1.5,
        label=f"Counterfactual Optimal Arm ($\\theta^*={opt_cf_theta}$, $\\mu^*={mean_cf_per_round[opt_cf_idx]:.3f}$)",
    )

    for algo_key in ALGO_ORDER:
        style = ALGO_STYLE[algo_key]
        rew_mat = agg[algo_key]["reward"]
        esc_mat = agg[algo_key]["bob_escalating"].astype(np.float64)

        roll_rew = np.vstack([get_rolling_mean(rew_mat[s], window_size) for s in range(rew_mat.shape[0])])
        roll_esc = np.vstack([get_rolling_mean(esc_mat[s], window_size) for s in range(esc_mat.shape[0])])

        ax_rew.plot(
            t_steps, roll_rew.mean(axis=0),
            label=style["label"], color=style["color"],
            linestyle=style["linestyle"], linewidth=1.9,
        )
        ax_rew.fill_between(
            t_steps,
            roll_rew.mean(axis=0) - roll_rew.std(axis=0),
            roll_rew.mean(axis=0) + roll_rew.std(axis=0),
            color=style["color"], alpha=0.12,
        )

        ax_esc.plot(
            t_steps, roll_esc.mean(axis=0) * 100.0,
            label=f"{algo_key} (Mean Esc: {esc_mat.mean()*100:.1f}%)",
            color=style["color"], linestyle=style["linestyle"], linewidth=1.9,
        )

    ax_rew.set_title(
        f"Stage 1: {window_size}-Round Rolling Reward & Bob Escalation Dynamics ({label})",
        fontsize=11.5, fontweight="bold",
    )
    ax_rew.set_ylabel(r"Normalized Reward $\tilde{r}_t \in [0, 1]$", fontsize=10.5)
    ax_rew.grid(True, linestyle=":", alpha=0.6)
    ax_rew.legend(loc="lower right", frameon=True, fontsize=8.8)

    ax_esc.set_ylabel("Bob Escalating (%)", fontsize=10.5)
    ax_esc.set_xlabel(r"Time Horizon ($T$)", fontsize=10.5)
    ax_esc.set_ylim(-2.0, 105.0)
    ax_esc.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax_esc.grid(True, linestyle=":", alpha=0.6)
    ax_esc.legend(loc="upper right", frameon=True, fontsize=8.8)

    fig.tight_layout()
    p2 = fig_dir / f"plot_2_reward_and_bob_regime_{label}.png"
    fig.savefig(p2)
    plt.close(fig)
    saved.append(p2)

    # -----------------------------------------------------------------
    # PLOT 3: Arm Selection Distribution vs. Rational/Escalating Payoffs
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10.2, 5.5), dpi=180)
    x = np.arange(len(thetas))
    n_algos = len(ALGO_ORDER)
    bar_width = 0.76 / n_algos

    for idx, algo_key in enumerate(ALGO_ORDER):
        style = ALGO_STYLE[algo_key]
        emp_probs = agg[algo_key]["empirical_probs"]
        mean_p = emp_probs.mean(axis=0)
        std_p = emp_probs.std(axis=0)
        offset = (idx - (n_algos - 1) / 2.0) * bar_width

        ax1.bar(
            x + offset, mean_p, width=bar_width, yerr=std_p, capsize=2.5,
            label=algo_key, color=style["color"], alpha=0.85,
            edgecolor="black", linewidth=0.5,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\theta={th}$" for th in thetas], fontsize=9.5)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(r"Threshold Strategy Arm ($\theta \in \{0, \dots, b\}$)", fontsize=11)
    ax1.set_ylabel("Empirical Selection Frequency ($N_T(\\theta)/T$)", fontsize=11)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        x, r_rat, color="darkgreen", marker="o", linestyle="-", linewidth=1.9,
        label="Payoff vs. Rational Bob $\\mu_{\\mathrm{rat}}(\\theta)$",
    )
    ax2.plot(
        x, r_esc, color="crimson", marker="s", linestyle="--", linewidth=1.9,
        label="Payoff vs. Escalating Bob $\\mu_{\\mathrm{esc}}(\\theta)$",
    )
    ax2.set_ylabel("Expected Normalized Reward", fontsize=11)
    ax2.set_ylim(0.0, 1.05)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", frameon=True, fontsize=8.8)
    ax1.set_title(
        f"Stage 1 Threshold Selection vs. Bob Regime Payoff Profiles ({label})",
        fontsize=11.5, fontweight="bold",
    )
    fig.tight_layout()
    p3 = fig_dir / f"plot_3_arm_distribution_vs_regimes_{label}.png"
    fig.savefig(p3)
    plt.close(fig)
    saved.append(p3)

    # -----------------------------------------------------------------
    # PLOT 4: 3-Panel Summary Bars (Policy Regret, Escalation %, Reward)
    # -----------------------------------------------------------------
    fig, (ax_p, ax_e, ax_r) = plt.subplots(1, 3, figsize=(12.2, 4.4), dpi=180)
    x_pos = np.arange(len(ALGO_ORDER))
    colors = [ALGO_STYLE[k]["color"] for k in ALGO_ORDER]

    pol_means = [float(agg[k]["policy_regret"][:, -1].mean()) for k in ALGO_ORDER]
    pol_stds = [float(agg[k]["policy_regret"][:, -1].std()) for k in ALGO_ORDER]

    esc_means = [float(agg[k]["bob_escalating"].mean(axis=1).mean()) * 100.0 for k in ALGO_ORDER]
    esc_stds = [float(agg[k]["bob_escalating"].mean(axis=1).std()) * 100.0 for k in ALGO_ORDER]

    rew_means = [float(agg[k]["reward"].mean(axis=1).mean()) for k in ALGO_ORDER]
    rew_stds = [float(agg[k]["reward"].mean(axis=1).std()) for k in ALGO_ORDER]

    b1 = ax_p.bar(x_pos, pol_means, yerr=pol_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax_p.set_xticks(x_pos)
    ax_p.set_xticklabels(ALGO_ORDER, fontweight="bold")
    ax_p.set_title("Final Policy Regret (Lower = Better)", fontsize=10.5, fontweight="bold")
    ax_p.set_ylabel(r"$R_T^{\mathrm{pol}}$")
    ax_p.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, v in zip(b1, pol_means):
        ax_p.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"{v:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    b2 = ax_e.bar(x_pos, esc_means, yerr=esc_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax_e.set_xticks(x_pos)
    ax_e.set_xticklabels(ALGO_ORDER, fontweight="bold")
    ax_e.set_ylim(0.0, 110.0)
    ax_e.set_title("Bob Escalation Triggered (%)", fontsize=10.5, fontweight="bold")
    ax_e.set_ylabel("% of Rounds in Escalating Regime")
    ax_e.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, v in zip(b2, esc_means):
        ax_e.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2.0, f"{v:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    b3 = ax_r.bar(x_pos, rew_means, yerr=rew_stds, capsize=4, color=colors, alpha=0.85, edgecolor="black", width=0.55)
    ax_r.set_xticks(x_pos)
    ax_r.set_xticklabels(ALGO_ORDER, fontweight="bold")
    ax_r.set_ylim(0.0, 1.0)
    ax_r.set_title("Mean Normalized Reward (Higher = Better)", fontsize=10.5, fontweight="bold")
    ax_r.set_ylabel(r"Mean $\tilde{r}_t$")
    ax_r.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, v in zip(b3, rew_means):
        ax_r.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.suptitle(f"Stage 1 Performance Summary vs. Bob — {label}", fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    p4 = fig_dir / f"plot_4_stage1_summary_bars_{label}.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved.append(p4)

    return saved


# =====================================================================
# 4. CONFIGURATION RUNNER & CSV EXPORTER
# =====================================================================

def run_stage1_scenario(scen: dict, out_root: Path) -> None:
    label = scen["label"]
    n_seeds = int(scen["n_seeds"])
    base_cfg = stage1_config(**scen["overrides"])
    T = base_cfg.T

    out_dir = out_root / label
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"\n======================================================================\n"
        f"Stage 1 Scenario '{label}' (Mesa Orchestration)\n"
        f"budget={base_cfg.budget}, stake={base_cfg.stake}, T={T}, seeds={n_seeds}, "
        f"window={base_cfg.bob_window}, dwell={base_cfg.bob_min_dwell}, gain={base_cfg.bob_gain}\n"
        f"======================================================================"
    )

    # Measure Bob's static Rational vs. Escalating payoff profile for reference
    bob_kw = dict(
        mu=base_cfg.bob_mu,
        sunk_cost_threshold=base_cfg.bob_sunk_cost_threshold,
        prior_sunk_cost=base_cfg.bob_prior_sunk_cost,
        escalation_rate=base_cfg.bob_escalation_rate,
        escalation_ceiling=float(base_cfg.budget),
    )
    r_rat = arm_rewards(base_cfg.budget, base_cfg.stake, bob_kw, escalating=False, T=200, seeds=2)
    r_esc = arm_rewards(base_cfg.budget, base_cfg.stake, bob_kw, escalating=True, T=200, seeds=2)

    seed_runs = []
    for seed in tqdm(range(n_seeds), desc=label):
        seed_runs.append(run_single_seed_stage1(base_cfg, seed))

    thetas = seed_runs[0]["thetas"]
    cf_totals = np.stack([r["cf_total_per_arm"] for r in seed_runs])

    agg: dict[str, dict[str, np.ndarray]] = {}
    for algo_key in ALGO_ORDER:
        agg[algo_key] = {}
        for m_key in [
            "reward",
            "policy_regret",
            "weak_regret",
            "arm_theta",
            "bob_escalating",
            "p_escalate",
            "agent_won",
            "agent_bid",
            "opp_bid",
            "empirical_probs",
        ]:
            agg[algo_key][m_key] = np.stack([r["algos"][algo_key][m_key] for r in seed_runs])

    # ---- TABLE 1: Algorithm Summary CSV --------------------------------
    summary_csv = table_dir / "algorithm_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "algorithm",
            "mean_policy_regret",
            "std_policy_regret",
            "mean_weak_regret",
            "std_weak_regret",
            "mean_reward_per_round",
            "bob_escalation_rate",
            "mean_win_rate",
            "mean_agent_bid",
        ])

        tqdm.write(
            f"\n{'Algo':<6} | {'Policy Regret':<20} | {'Weak Regret':<20} | "
            f"{'Esc %':<7} | {'Mean Rew'}"
        )
        tqdm.write("-" * 74)

        for algo_key in ALGO_ORDER:
            pol_r = agg[algo_key]["policy_regret"][:, -1]
            wk_r = agg[algo_key]["weak_regret"][:, -1]
            rew_m = agg[algo_key]["reward"].mean(axis=1)
            esc_m = agg[algo_key]["bob_escalating"].mean(axis=1)
            win_m = agg[algo_key]["agent_won"].mean(axis=1)
            bid_m = agg[algo_key]["agent_bid"].mean(axis=1)

            w.writerow([
                algo_key,
                f"{pol_r.mean():.4f}",
                f"{pol_r.std():.4f}",
                f"{wk_r.mean():.4f}",
                f"{wk_r.std():.4f}",
                f"{rew_m.mean():.4f}",
                f"{esc_m.mean():.4f}",
                f"{win_m.mean():.4f}",
                f"{bid_m.mean():.4f}",
            ])

            tqdm.write(
                f"{algo_key:<6} | {pol_r.mean():8.2f} +- {pol_r.std():<7.2f} | "
                f"{wk_r.mean():8.2f} +- {wk_r.std():<7.2f} | "
                f"{esc_m.mean()*100:5.1f}% | {rew_m.mean():.4f}"
            )

    # ---- TABLE 2: Checkpoint Policy Regret -----------------------------
    checkpoints = sorted({max(1, int(round(fr * T))) for fr in CHECKPOINT_FRACTIONS})
    ckpt_csv = table_dir / "checkpoint_policy_regret.csv"
    with open(ckpt_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["round_t"]
        for algo_key in ALGO_ORDER:
            header.extend([f"{algo_key}_policy_regret_mean", f"{algo_key}_policy_regret_std"])
        w.writerow(header)
        for c in checkpoints:
            row: list[Any] = [c]
            for algo_key in ALGO_ORDER:
                vals = agg[algo_key]["policy_regret"][:, c - 1]
                row.extend([f"{vals.mean():.4f}", f"{vals.std():.4f}"])
            w.writerow(row)

    # ---- TABLE 3: Per-Theta Selection Probabilities --------------------
    theta_csv = table_dir / "per_theta_selection_prob.csv"
    with open(theta_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["theta", "reward_bob_rational", "reward_bob_escalating", "counterfactual_mean_reward"]
        for algo_key in ALGO_ORDER:
            header.extend([f"{algo_key}_emp_freq_mean", f"{algo_key}_emp_freq_std"])
        w.writerow(header)

        cf_per_round = cf_totals.mean(axis=0) / float(T)
        for i, th in enumerate(thetas):
            row = [th, f"{r_rat[i]:.4f}", f"{r_esc[i]:.4f}", f"{cf_per_round[i]:.4f}"]
            for algo_key in ALGO_ORDER:
                p_vals = agg[algo_key]["empirical_probs"][:, i]
                row.extend([f"{p_vals.mean():.4f}", f"{p_vals.std():.4f}"])
            w.writerow(row)

    saved_figs = save_stage1_publication_plots(
        label=label,
        base_cfg=base_cfg,
        agg=agg,
        cf_totals=cf_totals,
        r_rat=r_rat,
        r_esc=r_esc,
        thetas=thetas,
        fig_dir=fig_dir,
    )
    for p in [summary_csv, ckpt_csv, theta_csv, *saved_figs]:
        tqdm.write(f"  saved {p.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 1 Multi-Algorithm Evaluation vs. Adaptive Bob (via Mesa)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Quick verification run (T=500, 2 seeds) saved to results/stage1_quick",
    )
    args = ap.parse_args()

    scenarios = [
        {"label": s["label"], "overrides": dict(s["overrides"]), "n_seeds": s["n_seeds"]}
        for s in STAGE1_CONFIGS
    ]
    out_root = PROJECT_ROOT / "results" / "stage1"
    if args.quick:
        out_root = PROJECT_ROOT / "results" / "stage1_quick"
        for s in scenarios:
            s["overrides"]["T"] = 500
            s["n_seeds"] = 2

    for scen in scenarios:
        run_stage1_scenario(scen, out_root)

    tqdm.write(f"\nDone! Stage 1 outputs saved in {out_root.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()