"""
run_stage0_test.py

Stage 0 Multi-Algorithm Benchmark: Comparing all non-BwK bandit algorithms
(ELP, EXP3, UCB1, ThompsonSampling) against the rational, oblivious opponent
(Alice) in Waniek et al.'s repeated Dollar Auction setting.

All algorithms are evaluated on the exact same sequence of auctions per seed
to guarantee a fair apple-to-apple comparison of static regret, rolling-mean
reward, win rate, and threshold arm convergence.

Outputs per configuration (results/stage0_test/<label>/):
    tables/algorithm_summary.csv             overall metrics per algorithm
    tables/checkpoint_regret_comparison.csv  regret at fractions of T across algos
    tables/per_theta_selection_prob.csv      selection frequency per theta
    figures/plot_1_cumulative_regret_<label>.png
    figures/plot_2_rolling_reward_<label>.png
    figures/plot_3_arm_distribution_<label>.png
    figures/plot_4_final_metrics_bar_<label>.png

Run from the project root:
    python -m experiments.stage0_baseline_replication.run_stage0_test
    python -m experiments.stage0_baseline_replication.run_stage0_test --quick
"""

from __future__ import annotations

import argparse
import csv
import inspect
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
from src.algorithms.exp3 import EXP3
from src.algorithms.thompson_sampling import ThompsonSampling
from src.algorithms.ucb1 import UCB1
from src.environment.dollar_auction import DollarAuction
from src.environment.strategies import build_threshold_arm_set
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

ALGO_ORDER = ["ELP", "EXP3", "UCB1", "TS"]

# Publication-grade visual styling per algorithm
ALGO_STYLE: dict[str, dict[str, Any]] = {
    "ELP": {
        "label": "ELP (Waniek et al., 2016)",
        "color": "tab:blue",
        "linestyle": "-",
        "marker": "|",
    },
    "EXP3": {
        "label": "EXP3 (Adversarial Baseline)",
        "color": "tab:orange",
        "linestyle": "--",
        "marker": "|",
    },
    "UCB1": {
        "label": "UCB1 (Stochastic Hoeffding)",
        "color": "tab:green",
        "linestyle": "-.",
        "marker": "|",
    },
    "TS": {
        "label": "Thompson Sampling (Beta-Bernoulli)",
        "color": "tab:red",
        "linestyle": ":",
        "marker": "|",
    },
}

CHECKPOINT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


# =====================================================================
# 1. HELPER FUNCTIONS & ALGORITHM FACTORY
# =====================================================================

def get_rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """
    Compute the O(T) rolling mean of a 1D array using cumulative sums.
    Matches the rolling-window smoothing used in publication figures.
    """
    window = max(1, int(window))
    cum_sum = np.cumsum(arr, dtype=np.float64)
    roll_mean = np.empty_like(cum_sum)
    for i in range(min(window, len(arr))):
        roll_mean[i] = cum_sum[i] / (i + 1)
    if len(arr) > window:
        roll_mean[window:] = (cum_sum[window:] - cum_sum[:-window]) / window
    return roll_mean


def instantiate_algorithm(
    algo_key: str,
    arms: list[Any],
    thetas: list[int],
    stake: float,
    budget: int,
    horizon_T: int,
    rng: np.random.Generator,
):
    """
    Instantiate one of the four non-BwK algorithms with a dedicated RNG.
    Uses signature inspection for EXP3 so any local constructor convention
    (whether expecting horizon_T, gamma, or n_arms) works seamlessly.
    """
    n_arms = len(arms)

    if algo_key == "ELP":
        return ELP(
            strategies=arms,
            thetas=thetas,
            stake=stake,
            budget=budget,
            horizon_T=horizon_T,
            rng=rng,
        )

    if algo_key == "EXP3":
        sig = inspect.signature(EXP3.__init__)
        candidate_kwargs = {
            "n_arms": n_arms,
            "strategies": arms,
            "thetas": thetas,
            "stake": stake,
            "budget": budget,
            "horizon_T": horizon_T,
            "T": horizon_T,
            "rng": rng,
        }
        accepted_kwargs = {
            k: v for k, v in candidate_kwargs.items() if k in sig.parameters
        }
        return EXP3(**accepted_kwargs)

    if algo_key == "UCB1":
        return UCB1(n_arms=n_arms, c=1.0, rng=rng)

    if algo_key == "TS":
        return ThompsonSampling(n_arms=n_arms, rng=rng)

    raise ValueError(f"Unknown algorithm key: {algo_key}")


# =====================================================================
# 2. SIMULATION ENGINE (COMMON RANDOM NUMBERS ACROSS ALGORITHMS)
# =====================================================================

def run_single_seed_all_algos(cfg: dict, seed: int) -> dict[str, Any]:
    """
    Run all 4 non-BwK algorithms on a single seed against Alice.

    Because Alice is oblivious and deterministic, for each round t we draw
    `agent_starts` once and evaluate all threshold arms `0..budget` under the
    exact same auction condition. Each algorithm then picks its arm, observes
    that arm's outcome (plus state trace for ELP), and updates its state.
    """
    budget = int(cfg["budget"])
    stake = float(cfg["stake"])
    alice_mu = float(cfg["alice_mu"])
    T = int(cfg["T"])

    thetas = list(range(0, budget + 1))
    arms = build_threshold_arm_set(budget=budget, stake=stake, thetas=thetas)
    n_arms = len(arms)

    env = DollarAuction(stake=stake, budget=budget)
    alice_strategy = Alice(stake=stake, budget=budget, mu=alice_mu).get_strategy()

    # Master RNG for the environment sequence (who starts each round)
    env_rng = np.random.default_rng(seed * 10007 + 1)

    # Separate identically-seeded RNGs for each algorithm's internal choices
    agents = {
        algo_key: instantiate_algorithm(
            algo_key=algo_key,
            arms=arms,
            thetas=thetas,
            stake=stake,
            budget=budget,
            horizon_T=T,
            rng=np.random.default_rng(seed * 10007 + 100 + idx),
        )
        for idx, algo_key in enumerate(ALGO_ORDER)
    }

    # Storage per algorithm
    hindsight_sums = np.zeros(n_arms, dtype=np.float64)
    algo_data: dict[str, dict[str, Any]] = {}
    for algo_key in ALGO_ORDER:
        algo_data[algo_key] = {
            "realized_sum": 0.0,
            "reward": np.zeros(T, dtype=np.float64),
            "regret": np.zeros(T, dtype=np.float64),
            "arm_theta": np.zeros(T, dtype=np.int64),
            "agent_won": np.zeros(T, dtype=bool),
            "agent_bid": np.zeros(T, dtype=np.float64),
            "opp_bid": np.zeros(T, dtype=np.float64),
            "pull_counts": np.zeros(n_arms, dtype=np.int64),
        }

    for t in tqdm(range(T), desc=f"  seed {seed} rounds", leave=False):
        agent_starts = bool(env_rng.integers(0, 2))

        # Evaluate all arms for this round's (agent_starts, alice_strategy)
        round_results = []
        round_norm_rewards = np.zeros(n_arms, dtype=np.float64)
        for i, s_arm in enumerate(arms):
            res_i = env.run(
                s_arm, alice_strategy, agent_starts=agent_starts, rng=env_rng
            )
            r_norm = DollarAuction.normalize_reward(
                res_i.agent_payoff, budget, stake
            )
            round_results.append(res_i)
            round_norm_rewards[i] = r_norm
            hindsight_sums[i] += r_norm

        best_hindsight_so_far = float(hindsight_sums.max())

        # Step each algorithm on the exact same round
        for algo_key, agent in agents.items():
            chosen_arm = agent.select_arm()
            res = round_results[chosen_arm]
            r = float(round_norm_rewards[chosen_arm])

            agent.update(
                chosen_arm,
                r,
                info={
                    "agent_state_trace": res.agent_state_trace,
                    "cost": float(res.agent_final_bid) / max(1, budget),
                },
            )

            d = algo_data[algo_key]
            d["realized_sum"] += r
            d["reward"][t] = r
            d["regret"][t] = best_hindsight_so_far - d["realized_sum"]
            d["arm_theta"][t] = thetas[chosen_arm]
            d["agent_won"][t] = res.winner == "agent"
            d["agent_bid"][t] = res.agent_final_bid
            d["opp_bid"][t] = res.opponent_final_bid
            d["pull_counts"][chosen_arm] += 1

    # Record empirical selection probabilities (and internal probs if available)
    for algo_key, agent in agents.items():
        d = algo_data[algo_key]
        emp_probs = d["pull_counts"].astype(np.float64) / float(T)
        d["empirical_probs"] = emp_probs
        if hasattr(agent, "_last_probs") and getattr(agent, "_last_probs") is not None:
            d["final_probs"] = np.asarray(getattr(agent, "_last_probs"), dtype=np.float64)
        else:
            d["final_probs"] = emp_probs

    return {
        "algos": algo_data,
        "hindsight": hindsight_sums,
        "thetas": thetas,
        "n_arms": n_arms,
    }


# =====================================================================
# 3. PUBLICATION-GRADE PLOTTING SUITE
# =====================================================================

def generate_publication_plots(
    label: str,
    cfg: dict,
    agg: dict[str, dict[str, np.ndarray]],
    hindsight_per_arm: np.ndarray,
    thetas: list[int],
    fig_dir: Path,
) -> list[Path]:
    """
    Generate 4 clean, high-resolution figures comparing ELP, EXP3, UCB1, and TS.
    """
    fig_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    T = int(cfg["T"])
    budget = int(cfg["budget"])
    stake = float(cfg["stake"])
    t_steps = np.arange(1, T + 1)
    mark_interval = max(1, T // 10)
    window_size = max(20, min(500, T // 20))

    mean_hindsight_reward_per_round = hindsight_per_arm.mean(axis=0) / float(T)
    opt_arm_idx = int(np.argmax(mean_hindsight_reward_per_round))
    opt_theta = thetas[opt_arm_idx]
    opt_reward_val = float(mean_hindsight_reward_per_round[opt_arm_idx])

    # -----------------------------------------------------------------
    # PLOT 1: Cumulative Static Regret (with +- 1 SD shaded bands)
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=180)
    for algo_key in ALGO_ORDER:
        style = ALGO_STYLE[algo_key]
        reg_matrix = agg[algo_key]["regret"]  # shape: (n_seeds, T)
        mean_reg = reg_matrix.mean(axis=0)
        std_reg = reg_matrix.std(axis=0)

        ax.plot(
            t_steps,
            mean_reg,
            label=style["label"],
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
            alpha=0.14,
        )

    ax.set_title(
        f"Stage 0 Cumulative Static Regret vs. Alice "
        f"(budget $b={budget}$, stake $s={stake:g}$, $K={len(thetas)}$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"Time Horizon ($T$)", fontsize=11)
    ax.set_ylabel(r"Cumulative Static Regret ($R_T$)", fontsize=11)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper left", frameon=True, fontsize=9.5)
    fig.tight_layout()

    p1 = fig_dir / f"plot_1_cumulative_regret_{label}.png"
    fig.savefig(p1)
    plt.close(fig)
    saved_paths.append(p1)

    # -----------------------------------------------------------------
    # PLOT 2: Rolling-Mean Normalized Reward vs. Optimal Threshold
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9.2, 5.5), dpi=180)
    ax.axhline(
        opt_reward_val,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Optimal Fixed Strategy ($\\theta^*={opt_theta}$, $\\mu^*={opt_reward_val:.3f}$)",
    )

    for algo_key in ALGO_ORDER:
        style = ALGO_STYLE[algo_key]
        rew_matrix = agg[algo_key]["reward"]  # shape: (n_seeds, T)
        roll_matrix = np.vstack(
            [get_rolling_mean(rew_matrix[s], window_size) for s in range(rew_matrix.shape[0])]
        )
        mean_roll = roll_matrix.mean(axis=0)
        std_roll = roll_matrix.std(axis=0)

        ax.plot(
            t_steps,
            mean_roll,
            label=style["label"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.9,
        )
        ax.fill_between(
            t_steps,
            mean_roll - std_roll,
            mean_roll + std_roll,
            color=style["color"],
            alpha=0.12,
        )

    ax.set_title(
        f"{window_size}-Round Rolling Mean Reward vs. Alice "
        f"(budget $b={budget}$, stake $s={stake:g}$)",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.set_xlabel(r"Time Horizon ($T$)", fontsize=11)
    ax.set_ylabel(r"Normalized Reward $\tilde{r}_t \in [0, 1]$", fontsize=11)
    ax.ticklabel_format(style="sci", axis="x", scilimits=(0, 0))
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", frameon=True, fontsize=9.5)
    fig.tight_layout()

    p2 = fig_dir / f"plot_2_rolling_reward_{label}.png"
    fig.savefig(p2)
    plt.close(fig)
    saved_paths.append(p2)

    # -----------------------------------------------------------------
    # PLOT 3: Empirical Arm Selection Distribution + Hindsight Profile
    # -----------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(10.0, 5.5), dpi=180)
    x = np.arange(len(thetas))
    n_algos = len(ALGO_ORDER)
    bar_width = 0.76 / n_algos

    for idx, algo_key in enumerate(ALGO_ORDER):
        style = ALGO_STYLE[algo_key]
        emp_probs = agg[algo_key]["empirical_probs"]  # (n_seeds, n_arms)
        mean_p = emp_probs.mean(axis=0)
        std_p = emp_probs.std(axis=0)
        offset = (idx - (n_algos - 1) / 2.0) * bar_width

        ax1.bar(
            x + offset,
            mean_p,
            width=bar_width,
            yerr=std_p,
            capsize=2.5,
            label=algo_key,
            color=style["color"],
            alpha=0.85,
            edgecolor="black",
            linewidth=0.5,
        )

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"$\\theta={th}$" for th in thetas], fontsize=9.5)
    ax1.set_ylim(0.0, 1.05)
    ax1.set_xlabel(r"Threshold Strategy Arm ($\theta \in \{0, \dots, b\}$)", fontsize=11)
    ax1.set_ylabel("Empirical Selection Frequency ($N_T(\\theta) / T$)", fontsize=11)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Secondary axis: True hindsight expected reward per arm
    ax2 = ax1.twinx()
    ax2.plot(
        x,
        mean_hindsight_reward_per_round,
        color="black",
        marker="o",
        linestyle="-",
        linewidth=1.8,
        markersize=6,
        label="Hindsight Mean Reward $\\mu(\\theta)$",
    )
    ax2.set_ylabel(r"Hindsight Mean Reward per Round", fontsize=11, color="black")
    ax2.set_ylim(
        max(0.0, float(mean_hindsight_reward_per_round.min()) - 0.1),
        min(1.05, float(mean_hindsight_reward_per_round.max()) + 0.1),
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper right",
        frameon=True,
        fontsize=9.0,
    )
    ax1.set_title(
        f"Threshold Arm Selection Distribution & Payoff Profile "
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
    # PLOT 4: Two-Panel Final Regret & Win Rate Bar Comparison
    # -----------------------------------------------------------------
    fig, (ax_reg, ax_win) = plt.subplots(1, 2, figsize=(9.8, 4.6), dpi=180)
    x_pos = np.arange(len(ALGO_ORDER))
    colors = [ALGO_STYLE[k]["color"] for k in ALGO_ORDER]

    final_reg_means = [float(agg[k]["regret"][:, -1].mean()) for k in ALGO_ORDER]
    final_reg_stds = [float(agg[k]["regret"][:, -1].std()) for k in ALGO_ORDER]

    win_means = [float(agg[k]["agent_won"].mean(axis=1).mean()) * 100.0 for k in ALGO_ORDER]
    win_stds = [float(agg[k]["agent_won"].mean(axis=1).std()) * 100.0 for k in ALGO_ORDER]

    bars1 = ax_reg.bar(
        x_pos, final_reg_means, yerr=final_reg_stds, capsize=5,
        color=colors, alpha=0.85, edgecolor="black", width=0.55
    )
    ax_reg.set_xticks(x_pos)
    ax_reg.set_xticklabels(ALGO_ORDER, fontsize=10.5, fontweight="bold")
    ax_reg.set_ylabel("Final Cumulative Static Regret ($R_T$)", fontsize=10.5)
    ax_reg.set_title(f"Final Regret at $T={T}$ (Lower is Better)", fontsize=11, fontweight="bold")
    ax_reg.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, m_val in zip(bars1, final_reg_means):
        ax_reg.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(final_reg_stds) * 0.15 + 0.5,
            f"{m_val:.1f}",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    bars2 = ax_win.bar(
        x_pos, win_means, yerr=win_stds, capsize=5,
        color=colors, alpha=0.85, edgecolor="black", width=0.55
    )
    ax_win.set_xticks(x_pos)
    ax_win.set_xticklabels(ALGO_ORDER, fontsize=10.5, fontweight="bold")
    ax_win.set_ylabel("Auction Win Rate (%)", fontsize=10.5)
    ax_win.set_ylim(0.0, 110.0)
    ax_win.set_title("Average Auction Win Rate (%)", fontsize=11, fontweight="bold")
    ax_win.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar, w_val in zip(bars2, win_means):
        ax_win.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 2.0,
            f"{w_val:.1f}%",
            ha="center", va="bottom", fontsize=9.5, fontweight="bold",
        )

    fig.suptitle(
        f"Stage 0 Performance Summary — {label} ($b={budget}, s={stake:g}$)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    p4 = fig_dir / f"plot_4_final_metrics_bar_{label}.png"
    fig.savefig(p4, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(p4)

    return saved_paths


# =====================================================================
# 4. CONFIG RUNNER & CSV REPORT GENERATOR
# =====================================================================

def run_config(cfg: dict, out_root: Path) -> None:
    label = cfg["label"]
    n_seeds = int(cfg["n_seeds"])
    T = int(cfg["T"])

    out_dir = out_root / label
    table_dir = out_dir / "tables"
    fig_dir = out_dir / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tqdm.write(
        f"\n======================================================================\n"
        f"Config '{label}' | budget={cfg['budget']}, stake={cfg['stake']}, "
        f"T={T}, seeds={n_seeds}\n"
        f"Algorithms: {', '.join(ALGO_ORDER)}\n"
        f"======================================================================"
    )

    seed_runs = []
    for seed in tqdm(range(n_seeds), desc=label):
        res = run_single_seed_all_algos(cfg, seed)
        seed_runs.append(res)

    thetas = seed_runs[0]["thetas"]
    hindsight_per_arm = np.stack([r["hindsight"] for r in seed_runs])  # (n_seeds, n_arms)
    mean_hindsight_per_round = hindsight_per_arm.mean(axis=0) / float(T)
    opt_arm_idx = int(np.argmax(mean_hindsight_per_round))
    opt_theta = thetas[opt_arm_idx]

    # Stack metrics per algorithm across seeds
    agg: dict[str, dict[str, np.ndarray]] = {}
    for algo_key in ALGO_ORDER:
        agg[algo_key] = {}
        for metric_key in [
            "reward",
            "regret",
            "arm_theta",
            "agent_won",
            "agent_bid",
            "opp_bid",
            "empirical_probs",
            "final_probs",
        ]:
            agg[algo_key][metric_key] = np.stack(
                [r["algos"][algo_key][metric_key] for r in seed_runs]
            )

    # ---- TABLE 1: Overall Algorithm Comparison Summary -----------------
    summary_csv = table_dir / "algorithm_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "algorithm",
            "mean_final_regret",
            "std_final_regret",
            "mean_reward_per_round",
            "std_reward_per_round",
            "mean_win_rate",
            "mean_agent_bid",
            "optimal_theta",
            "optimal_theta_pull_freq",
        ])

        tqdm.write(f"\n--- Summary Table for '{label}' (Optimal theta* = {opt_theta}) ---")
        tqdm.write(
            f"{'Algo':<6} | {'Final Regret (mean +- std)':<26} | "
            f"{'Mean Rew':<8} | {'Win Rate':<8} | {'P(theta*)'}"
        )
        tqdm.write("-" * 70)

        for algo_key in ALGO_ORDER:
            final_regs = agg[algo_key]["regret"][:, -1]
            rew_per_round = agg[algo_key]["reward"].mean(axis=1)
            win_rates = agg[algo_key]["agent_won"].mean(axis=1)
            bids = agg[algo_key]["agent_bid"].mean(axis=1)
            opt_freq = agg[algo_key]["empirical_probs"][:, opt_arm_idx].mean()

            w.writerow([
                algo_key,
                f"{final_regs.mean():.4f}",
                f"{final_regs.std():.4f}",
                f"{rew_per_round.mean():.4f}",
                f"{rew_per_round.std():.4f}",
                f"{win_rates.mean():.4f}",
                f"{bids.mean():.4f}",
                opt_theta,
                f"{opt_freq:.4f}",
            ])

            tqdm.write(
                f"{algo_key:<6} | {final_regs.mean():10.3f} +- {final_regs.std():<10.3f} | "
                f"{rew_per_round.mean():8.4f} | {win_rates.mean()*100:6.2f}%  | "
                f"{opt_freq*100:6.2f}%"
            )

    # ---- TABLE 2: Checkpoint Regret Across All Algorithms --------------
    checkpoints = sorted({max(1, int(round(fr * T))) for fr in CHECKPOINT_FRACTIONS})
    ckpt_csv = table_dir / "checkpoint_regret_comparison.csv"
    with open(ckpt_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["round_t"]
        for algo_key in ALGO_ORDER:
            header.extend([f"{algo_key}_mean_regret", f"{algo_key}_std_regret"])
        w.writerow(header)

        for c in checkpoints:
            row: list[Any] = [c]
            for algo_key in ALGO_ORDER:
                vals = agg[algo_key]["regret"][:, c - 1]
                row.extend([f"{vals.mean():.4f}", f"{vals.std():.4f}"])
            w.writerow(row)

    # ---- TABLE 3: Per-Theta Selection Frequency per Algorithm ----------
    theta_csv = table_dir / "per_theta_selection_prob.csv"
    with open(theta_csv, "w", newline="") as f:
        w = csv.writer(f)
        header = ["theta", "hindsight_mean_reward"]
        for algo_key in ALGO_ORDER:
            header.extend([f"{algo_key}_emp_freq_mean", f"{algo_key}_emp_freq_std"])
        w.writerow(header)

        for i, th in enumerate(thetas):
            row = [th, f"{mean_hindsight_per_round[i]:.4f}"]
            for algo_key in ALGO_ORDER:
                p_vals = agg[algo_key]["empirical_probs"][:, i]
                row.extend([f"{p_vals.mean():.4f}", f"{p_vals.std():.4f}"])
            w.writerow(row)

    # ---- FIGURES: Save 4 Publication Plots -----------------------------
    saved_figs = generate_publication_plots(
        label=label,
        cfg=cfg,
        agg=agg,
        hindsight_per_arm=hindsight_per_arm,
        thetas=thetas,
        fig_dir=fig_dir,
    )
    tqdm.write("\nSaved tables and figures:")
    for p in [summary_csv, ckpt_csv, theta_csv, *saved_figs]:
        tqdm.write(f"  -> {p.relative_to(PROJECT_ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Stage 0 Multi-Algorithm Comparison (ELP, EXP3, UCB1, TS vs. Alice)"
    )
    ap.add_argument(
        "--quick",
        action="store_true",
        help="Quick verification run (T=500, 2 seeds) saved to results/stage0_test_quick",
    )
    args = ap.parse_args()

    configs = [dict(c) for c in CONFIGS]
    out_root = PROJECT_ROOT / "results" / "stage0_test"
    if args.quick:
        out_root = PROJECT_ROOT / "results" / "stage0_test_quick"
        for c in configs:
            c["T"] = 500
            c["n_seeds"] = 2

    for cfg in configs:
        run_config(cfg, out_root)

    tqdm.write(f"\nAll done! Outputs saved in {out_root.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()