"""
run_stage0.py

Stage 0 -- Baseline replication: ELP vs. the rational, oblivious opponent
(Alice) in Waniek et al.'s setting, evaluating static regret against the best
fixed threshold arm in hindsight.

This script is STANDALONE (own loop, no runner.py / Mesa): the opponent is
oblivious, so per-round counterfactuals ("what would arm g have earned this
round") are valid and cheap.

Outputs per configuration (results/stage0/<label>/):
    tables/raw_results.csv                      one row per (seed, round)
    tables/checkpoint_regret.csv                regret at fractions of T (mean/std)
    tables/summary.csv                          final selection prob + hindsight reward
    figures/plot_1_cumulative_regret_<label>.png
    figures/plot_2_rolling_reward_<label>.png
    figures/plot_3_arm_distribution_<label>.png
    figures/plot_4_regret_vs_bound_<label>.png
Cross-configuration outputs (results/stage0/):
    comparison_regret.png
    comparison_reward.png

Run from the project root:
    python -m experiments.stage0_baseline_replication.run_stage0
    python -m experiments.stage0_baseline_replication.run_stage0 --quick
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.elp import ELP
from src.environment.dollar_auction import DollarAuction
from src.environment.strategies import build_threshold_arm_set
from src.metrics.plots import RunBundle, moving_average
from src.opponents.alice_rational import Alice

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    def tqdm(it=None, **kwargs):
        return it
    tqdm.write = print


CONFIGS = [
    {
        "label": "budget12_stake5",
        "budget": 12,
        "stake": 5.0,
        "alice_mu": 0.8,
        "T": 10000,
        "n_seeds": 8,
    },
    {
        "label": "budget8_stake10",
        "budget": 8,
        "stake": 10.0,
        "alice_mu": 0.8,
        "T": 10000,
        "n_seeds": 8,
    },
]

MOVING_AVG_WINDOW = 50
CHECKPOINT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]

CONFIG_STYLES = [
    {"color": "tab:blue", "linestyle": "-", "marker": "|"},
    {"color": "tab:red", "linestyle": "--", "marker": "|"},
    {"color": "tab:green", "linestyle": "-.", "marker": "|"},
    {"color": "tab:purple", "linestyle": ":", "marker": "|"},
]


# =====================================================================
# 1. HELPER FUNCTIONS & THEORETICAL BOUND
# =====================================================================

def get_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the O(T) rolling mean of a 1D array using cumulative sums.
    """
    window = max(1, int(window))
    cum_sum = np.cumsum(arr, dtype=np.float64)
    roll_mean = np.empty_like(cum_sum)
    for i in range(min(window, len(arr))):
        roll_mean[i] = cum_sum[i] / (i + 1)
    if len(arr) > window:
        roll_mean[window:] = (cum_sum[window:] - cum_sum[:-window]) / window
    return roll_mean


def theorem6_bound(
    T: int, n_arms: int, beta: float, epsilon: float
) -> np.ndarray:
    """
    Upper bound on ELP's regret for the threshold family (alpha(G)=1), for
    the beta actually used (Theorem 6 / Thm 2 with alpha = 1):

        U(t) <= 9 * beta * t * log(6|S0| / eps) + log|S0| / beta

    Valid for every t simultaneously because beta is fixed. Normalized
    rewards lie in [0, 1], so the units match our cumulative regret.
    """
    t = np.arange(1, T + 1, dtype=np.float64)
    return 9.0 * beta * t * math.log(6 * n_arms / epsilon) + math.log(n_arms) / beta


# =====================================================================
# 2. SINGLE-SEED SIMULATION ENGINE
# =====================================================================

def run_single_seed(cfg: dict, seed: int) -> dict[str, Any]:
    budget = int(cfg["budget"])
    stake = float(cfg["stake"])
    alice_mu = float(cfg["alice_mu"])
    T = int(cfg["T"])
    rng = np.random.default_rng(seed)

    thetas = list(range(0, budget + 1))
    arms = build_threshold_arm_set(budget=budget, stake=stake, thetas=thetas)
    n_arms = len(arms)

    env = DollarAuction(stake=stake, budget=budget)
    alice_strategy = Alice(stake=stake, budget=budget, mu=alice_mu).get_strategy()
    elp = ELP(
        strategies=arms,
        thetas=thetas,
        stake=stake,
        budget=budget,
        horizon_T=T,
        rng=rng,
    )

    hindsight_sums = np.zeros(n_arms, dtype=np.float64)
    realized_sum = 0.0

    reward = np.zeros(T, dtype=np.float64)
    regret = np.zeros(T, dtype=np.float64)
    arm_theta = np.zeros(T, dtype=int)
    agent_won = np.zeros(T, dtype=bool)
    agent_bid = np.zeros(T, dtype=np.float64)
    opp_bid = np.zeros(T, dtype=np.float64)
    pull_counts = np.zeros(n_arms, dtype=np.int64)

    for t in tqdm(range(T), desc=f"  seed {seed} rounds", leave=False):
        agent_starts = bool(rng.integers(0, 2))

        arm = elp.select_arm()
        result = env.run(
            arms[arm], alice_strategy, agent_starts=agent_starts, rng=rng
        )
        r = DollarAuction.normalize_reward(result.agent_payoff, budget, stake)
        elp.update(arm, r, info={"agent_state_trace": result.agent_state_trace})
        realized_sum += r

        # Hindsight: valid here because Alice is oblivious (her behaviour
        # does not depend on the learner's past play).
        for i, s in enumerate(arms):
            r_i = env.run(s, alice_strategy, agent_starts=agent_starts, rng=rng)
            hindsight_sums[i] += DollarAuction.normalize_reward(
                r_i.agent_payoff, budget, stake
            )

        reward[t] = r
        regret[t] = hindsight_sums.max() - realized_sum
        arm_theta[t] = thetas[arm]
        agent_won[t] = result.winner == "agent"
        agent_bid[t] = result.agent_final_bid
        opp_bid[t] = result.opponent_final_bid
        pull_counts[arm] += 1

    return {
        "reward": reward,
        "regret": regret,
        "arm_theta": arm_theta,
        "agent_won": agent_won,
        "agent_bid": agent_bid,
        "opp_bid": opp_bid,
        "final_probs": np.array(elp._last_probs, dtype=np.float64),
        "empirical_probs": pull_counts.astype(np.float64) / float(T),
        "hindsight": hindsight_sums,
        "elp_beta": elp.beta,
        "elp_epsilon": getattr(elp, "epsilon", None),
        "n_arms": n_arms,
    }


# =====================================================================
# 3. PUBLICATION-GRADE PLOTTING FUNCTIONS
# =====================================================================

def save_stage0_publication_plots(
    bundle: RunBundle,
    empirical_probs: np.ndarray,
    cfg: dict,
    fig_dir: Path,
) -> list[Path]:
    """
    Generate 4 clean publication-grade figures for a single Stage 0 config.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    label = bundle.label
    thetas = list(bundle.thetas)
    T = int(cfg["T"])
    budget = int(cfg["budget"])
    stake = float(cfg["stake"])
    t_steps = np.arange(1, T + 1)
    mark_interval = max(1, T // 10)
    window_size = max(20, min(500, T // 20))

    mean_hindsight_per_round = bundle.hindsight_per_arm.mean(axis=0) / float(T)
    opt_arm_idx = int(np.argmax(mean_hindsight_per_round))
    opt_theta = thetas[opt_arm_idx]
    opt_reward_val = float(mean_hindsight_per_round[opt_arm_idx])

    mean_reg = bundle.regret.mean(axis=0)
    std_reg = bundle.regret.std(axis=0)

    # -----------------------------------------------------------------
    # PLOT 1: Empirical Cumulative Static Regret (Zoomed Clean View)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=180)
    ax.plot(
        t_steps,
        mean_reg,
        label=f"ELP (Final $R_T = {mean_reg[-1]:.2f} \\pm {std_reg[-1]:.2f}$)",
        color="tab:blue",
        linestyle="-",
        linewidth=2.1,
        marker="|",
        markevery=mark_interval,
        markersize=9,
    )
    ax.fill_between(
        t_steps,
        np.maximum(0.0, mean_reg - std_reg),
        mean_reg + std_reg,
        color="tab:blue",
        alpha=0.18,
        label=r"$\pm 1$ Std. Dev. across seeds",
    )

    ax.set_title(
        f"Stage 0 Empirical Static Regret — ELP vs. Alice "
        f"(budget $b={budget}$, stake $s={stake:g}$, $K={len(thetas)}$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"Time Horizon ($T$)", fontsize=11)
    ax.set_ylabel(r"Cumulative Static Regret ($R_T$)", fontsize=11)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()

    p1 = fig_dir / f"plot_1_cumulative_regret_{label}.png"
    fig.savefig(p1)
    plt.close(fig)
    saved_paths.append(p1)

    # -----------------------------------------------------------------
    # PLOT 2: Rolling-Mean Normalized Reward vs. Optimal Threshold
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=180)
    ax.axhline(
        opt_reward_val,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=f"Optimal Fixed Strategy ($\\theta^*={opt_theta}$, $\\mu^*={opt_reward_val:.3f}$)",
    )

    roll_matrix = np.vstack(
        [
            get_rolling_mean(bundle.reward[s], window_size)
            for s in range(bundle.reward.shape[0])
        ]
    )
    mean_roll = roll_matrix.mean(axis=0)
    std_roll = roll_matrix.std(axis=0)

    ax.plot(
        t_steps,
        mean_roll,
        label=f"ELP ({window_size}-Round Rolling Mean)",
        color="tab:blue",
        linestyle="-",
        linewidth=2.0,
        marker="|",
        markevery=mark_interval,
        markersize=9,
    )
    ax.fill_between(
        t_steps,
        mean_roll - std_roll,
        mean_roll + std_roll,
        color="tab:blue",
        alpha=0.16,
    )

    ax.set_title(
        f"{window_size}-Round Rolling Mean Reward — ELP vs. Alice "
        f"(budget $b={budget}$, stake $s={stake:g}$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"Time Horizon ($T$)", fontsize=11)
    ax.set_ylabel(r"Normalized Reward $\tilde{r}_t \in [0, 1]$", fontsize=11)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()

    p2 = fig_dir / f"plot_2_rolling_reward_{label}.png"
    fig.savefig(p2)
    plt.close(fig)
    saved_paths.append(p2)

    # -----------------------------------------------------------------
    # PLOT 3: Threshold Arm Distribution (Empirical vs. Final Weights)
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(9.6, 5.4), dpi=180)
    x = np.arange(len(thetas))
    bar_width = 0.36

    emp_mean = empirical_probs.mean(axis=0)
    emp_std = empirical_probs.std(axis=0)
    fin_mean = bundle.final_probs.mean(axis=0)
    fin_std = bundle.final_probs.std(axis=0)

    ax1.bar(
        x - bar_width / 2.0,
        emp_mean,
        width=bar_width,
        yerr=emp_std,
        capsize=3,
        label="Empirical Pull Frequency ($N_T(\\theta)/T$)",
        color="tab:blue",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )
    ax1.bar(
        x + bar_width / 2.0,
        fin_mean,
        width=bar_width,
        yerr=fin_std,
        capsize=3,
        label="Final ELP Distribution ($p_T(\\theta)$)",
        color="tab:cyan",
        alpha=0.85,
        edgecolor="black",
        linewidth=0.6,
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\theta={th}$" for th in thetas], fontsize=9.5)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(r"Threshold Strategy Arm ($\theta \in \{0, \dots, b\}$)", fontsize=11)
    ax1.set_ylabel("Selection Probability", fontsize=11)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    ax2.plot(
        x,
        mean_hindsight_per_round,
        color="tab:red",
        marker="o",
        linestyle="-",
        linewidth=2.0,
        markersize=6,
        label="Hindsight Mean Reward $\\mu(\\theta)$",
    )
    ax2.set_ylabel(r"Hindsight Mean Reward per Round", fontsize=11, color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax2.set_ylim(
        max(0.0, float(mean_hindsight_per_round.min()) - 0.1),
        min(1.05, float(mean_hindsight_per_round.max()) + 0.1),
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
        frameon=True,
        fontsize=9.5,
    )
    ax1.set_title(
        f"ELP Strategy Convergence & Payoff Profile "
        f"(budget $b={budget}$, stake $s={stake:g}$, optimal $\\theta^*={opt_theta}$)",
        fontsize=11.5,
        fontweight="bold",
    )
    fig.tight_layout()

    p3 = fig_dir / f"plot_3_arm_distribution_{label}.png"
    fig.savefig(p3)
    plt.close(fig)
    saved_paths.append(p3)

    # -----------------------------------------------------------------
    # PLOT 4: Empirical Regret vs. Theorem 6 Theoretical Bound
    # -----------------------------------------------------------------
    if bundle.bound is not None:
        fig, (ax_lin, ax_log) = plt.subplots(1, 2, figsize=(10.5, 4.6), dpi=180)

        # Left Panel: Linear scale showing both Bound and Empirical Regret
        ax_lin.plot(
            t_steps,
            bundle.bound,
            label="Theorem 6 Upper Bound $U(t)$",
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
        )
        ax_lin.plot(
            t_steps,
            mean_reg,
            label="Empirical ELP Regret $R(t)$",
            color="tab:blue",
            linestyle="-",
            linewidth=2.0,
        )
        ax_lin.set_title("Linear Scale ($R(t) \\ll U(t)$)", fontsize=11, fontweight="bold")
        ax_lin.set_xlabel(r"Time Horizon ($T$)", fontsize=10.5)
        ax_lin.set_ylabel("Regret", fontsize=10.5)
        ax_lin.ticklabel_format(style="sci", axis="both", scilimits=(0, 0))
        ax_lin.grid(True, linestyle=":", alpha=0.6)
        ax_lin.legend(loc="upper left", frameon=True, fontsize=9.0)

        # Right Panel: Log-scale Y-axis so both curves are clearly visible
        ax_log.plot(
            t_steps,
            bundle.bound,
            label=f"Theorem 6 Bound ($U(T)={bundle.bound[-1]:.0f}$)",
            color="tab:red",
            linestyle="--",
            linewidth=2.0,
        )
        ax_log.plot(
            t_steps,
            np.maximum(1e-2, mean_reg),
            label=f"Empirical ELP ($R(T)={mean_reg[-1]:.2f}$)",
            color="tab:blue",
            linestyle="-",
            linewidth=2.0,
        )
        ax_log.set_yscale("log")
        ax_log.set_title("Log-Scale Comparison", fontsize=11, fontweight="bold")
        ax_log.set_xlabel(r"Time Horizon ($T$)", fontsize=10.5)
        ax_log.set_ylabel("Regret (log scale)", fontsize=10.5)
        ax_log.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
        ax_log.grid(True, which="both", linestyle=":", alpha=0.6)
        ax_log.legend(loc="lower right", frameon=True, fontsize=9.0)

        fig.suptitle(
            f"Empirical Regret vs. Waniek et al. Theorem 6 Bound — {label}",
            fontsize=12,
            fontweight="bold",
            y=1.02,
        )
        fig.tight_layout()

        p4 = fig_dir / f"plot_4_regret_vs_bound_{label}.png"
        fig.savefig(p4, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(p4)

    return saved_paths


def save_cross_config_comparisons(
    bundles: list[RunBundle], out_root: Path
) -> list[Path]:
    """
    Generate cross-configuration comparison plots for cumulative regret and
    rolling-mean reward in the publication visual style.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # 1. Cross-Config Cumulative Regret Comparison
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=180)
    for idx, b in enumerate(bundles):
        style = CONFIG_STYLES[idx % len(CONFIG_STYLES)]
        T = b.regret.shape[1]
        t_steps = np.arange(1, T + 1)
        mark_interval = max(1, T // 10)
        mean_reg = b.regret.mean(axis=0)
        std_reg = b.regret.std(axis=0)

        ax.plot(
            t_steps,
            mean_reg,
            label=f"{b.label} ($R_T = {mean_reg[-1]:.2f} \\pm {std_reg[-1]:.2f}$)",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker=style["marker"],
            markevery=mark_interval,
            markersize=9,
        )
        ax.fill_between(
            t_steps,
            np.maximum(0.0, mean_reg - std_reg),
            mean_reg + std_reg,
            color=style["color"],
            alpha=0.15,
        )

    ax.set_title(
        "Stage 0: Cumulative Static Regret Across Auction Configurations (ELP vs. Alice)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"Time Horizon ($T$)", fontsize=11)
    ax.set_ylabel(r"Cumulative Static Regret ($R_T$)", fontsize=11)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=10)
    fig.tight_layout()

    p_reg = out_root / "comparison_regret.png"
    fig.savefig(p_reg)
    plt.close(fig)
    saved.append(p_reg)

    # 2. Cross-Config Rolling Mean Reward Comparison
    fig, ax = plt.subplots(figsize=(9.0, 5.4), dpi=180)
    for idx, b in enumerate(bundles):
        style = CONFIG_STYLES[idx % len(CONFIG_STYLES)]
        T = b.reward.shape[1]
        t_steps = np.arange(1, T + 1)
        mark_interval = max(1, T // 10)
        window_size = max(20, min(500, T // 20))

        roll_matrix = np.vstack(
            [get_rolling_mean(b.reward[s], window_size) for s in range(b.reward.shape[0])]
        )
        mean_roll = roll_matrix.mean(axis=0)
        std_roll = roll_matrix.std(axis=0)

        ax.plot(
            t_steps,
            mean_roll,
            label=f"{b.label} (ELP)",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.0,
            marker=style["marker"],
            markevery=mark_interval,
            markersize=9,
        )
        ax.fill_between(
            t_steps,
            mean_roll - std_roll,
            mean_roll + std_roll,
            color=style["color"],
            alpha=0.14,
        )

    ax.set_title(
        "Stage 0: Rolling-Mean Normalized Reward Across Configurations (ELP vs. Alice)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"Time Horizon ($T$)", fontsize=11)
    ax.set_ylabel(r"Normalized Reward $\tilde{r}_t \in [0, 1]$", fontsize=11)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=10)
    fig.tight_layout()

    p_rew = out_root / "comparison_reward.png"
    fig.savefig(p_rew)
    plt.close(fig)
    saved.append(p_rew)

    return saved


# =====================================================================
# 4. CONFIGURATION RUNNER & TABLE EXPORTER
# =====================================================================

def run_config(cfg: dict, out_root: Path) -> RunBundle:
    label, n_seeds, T = cfg["label"], cfg["n_seeds"], cfg["T"]
    out_dir = out_root / label
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"\n======================================================================\n"
        f"Stage 0 Config '{label}' | budget={cfg['budget']}, stake={cfg['stake']}, "
        f"T={T}, seeds={n_seeds}\n"
        f"======================================================================"
    )

    runs = []
    for seed in tqdm(range(n_seeds), desc=label):
        res = run_single_seed(cfg, seed)
        runs.append(res)
        tqdm.write(
            f"  seed={seed:2d} | final regret={res['regret'][-1]:8.3f} | "
            f"mean reward={res['reward'].mean():.4f} | "
            f"win_rate={res['agent_won'].mean()*100:6.2f}%"
        )

    stack = lambda k: np.stack([r[k] for r in runs])
    thetas = list(range(0, cfg["budget"] + 1))

    bound = None
    eps = runs[0]["elp_epsilon"]
    if eps:
        bound = theorem6_bound(T, runs[0]["n_arms"], runs[0]["elp_beta"], eps)

    bundle = RunBundle(
        label=label,
        thetas=thetas,
        reward=stack("reward"),
        regret=stack("regret"),
        arm_theta=stack("arm_theta"),
        agent_won=stack("agent_won"),
        agent_bid=stack("agent_bid"),
        opp_bid=stack("opp_bid"),
        final_probs=stack("final_probs"),
        hindsight_per_arm=stack("hindsight"),
        regret_name="static regret",
        bound=bound,
    )
    empirical_probs = stack("empirical_probs")

    # ---- 1. raw per-round CSV ------------------------------------------
    ma = moving_average(bundle.reward, MOVING_AVG_WINDOW)
    cum_reward = np.cumsum(bundle.reward, axis=1)
    raw_path = table_dir / "raw_results.csv"
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "seed",
            "round",
            "arm_theta",
            "winner",
            "agent_final_bid",
            "opponent_final_bid",
            "reward",
            "moving_avg_reward",
            "cumulative_reward",
            "static_regret",
        ])
        for s in range(n_seeds):
            for t in range(T):
                w.writerow([
                    s,
                    t + 1,
                    bundle.arm_theta[s, t],
                    "agent" if bundle.agent_won[s, t] else "opponent",
                    bundle.agent_bid[s, t],
                    bundle.opp_bid[s, t],
                    bundle.reward[s, t],
                    ma[s, t],
                    cum_reward[s, t],
                    bundle.regret[s, t],
                ])

    # ---- 2. checkpoint regret CSV --------------------------------------
    checkpoints = sorted({max(1, int(round(fr * T))) for fr in CHECKPOINT_FRACTIONS})
    ckpt_path = table_dir / "checkpoint_regret.csv"
    with open(ckpt_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "round_t",
            "mean_static_regret",
            "std_static_regret",
            "theoretical_bound",
        ])
        for c in checkpoints:
            vals = bundle.regret[:, c - 1]
            w.writerow([
                c,
                float(vals.mean()),
                float(vals.std()),
                float(bound[c - 1]) if bound is not None else "",
            ])

    # ---- 3. per-theta summary CSV --------------------------------------
    summary_path = table_dir / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "theta",
            "mean_final_selection_prob",
            "std_final_selection_prob",
            "mean_empirical_pull_freq",
            "std_empirical_pull_freq",
            "mean_hindsight_reward_per_round",
        ])
        for i, th in enumerate(thetas):
            w.writerow([
                th,
                float(bundle.final_probs[:, i].mean()),
                float(bundle.final_probs[:, i].std()),
                float(empirical_probs[:, i].mean()),
                float(empirical_probs[:, i].std()),
                float(bundle.hindsight_per_arm[:, i].mean() / T),
            ])

    mean_hindsight = bundle.hindsight_per_arm.mean(axis=0) / float(T)
    opt_idx = int(np.argmax(mean_hindsight))
    tqdm.write("-" * 70)
    tqdm.write(
        f"  Summary '{label}': Optimal theta* = {thetas[opt_idx]} "
        f"(mu* = {mean_hindsight[opt_idx]:.4f})"
    )
    tqdm.write(
        f"  Final Regret = {bundle.regret[:, -1].mean():.3f} +- "
        f"{bundle.regret[:, -1].std():.3f} | "
        f"Win Rate = {bundle.agent_won.mean()*100:.2f}% | "
        f"P(theta*) = {empirical_probs[:, opt_idx].mean()*100:.2f}%"
    )
    if bound is not None:
        tqdm.write(f"  Theorem 6 Bound at T={T}: {bound[-1]:.1f}")
    for c in checkpoints:
        tqdm.write(f"    t={c:5d}: regret = {bundle.regret[:, c - 1].mean():.3f}")

    # ---- 4. Publication Figures ----------------------------------------
    saved_figs = save_stage0_publication_plots(
        bundle=bundle,
        empirical_probs=empirical_probs,
        cfg=cfg,
        fig_dir=fig_dir,
    )
    for p in [raw_path, ckpt_path, summary_path, *saved_figs]:
        tqdm.write(f"  saved {p.relative_to(PROJECT_ROOT)}")

    return bundle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--quick",
        action="store_true",
        help="tiny run (T=500, 2 seeds) into results/stage0_quick",
    )
    args = ap.parse_args()

    configs = [dict(c) for c in CONFIGS]
    out_root = PROJECT_ROOT / "results" / "stage0"
    if args.quick:
        out_root = PROJECT_ROOT / "results" / "stage0_quick"
        for c in configs:
            c["T"], c["n_seeds"] = 500, 2

    bundles = [run_config(c, out_root) for c in configs]

    comp_figs = save_cross_config_comparisons(bundles, out_root)
    for p in comp_figs:
        tqdm.write(f"  saved {p.relative_to(PROJECT_ROOT)}")
    tqdm.write(f"\nDone. Outputs in {out_root.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()