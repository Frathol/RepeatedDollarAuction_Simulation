"""
run_stage0.py

Stage 0 -- Baseline Replication (detailed version).

Adds, on top of the first version:
    - `winner` recorded every round (who actually won that auction)
    - a rolling MOVING AVERAGE of reward (see explanation below)
    - regret reported at several horizon CHECKPOINTS (not just the
      final T), so you can see how fast it grows early vs late
    - support for running SEVERAL PARAMETER CONFIGURATIONS in one
      go, each saved to its OWN subfolder (so results never get mixed
      together), plus one combined comparison plot overlaying all
      configs' mean regret curves side by side.

WHY A MOVING AVERAGE, SPECIFICALLY:
    Cumulative regret (and cumulative reward) can only ever go up (or
    plateau) -- it accumulates the ENTIRE history, so it can never
    "recover" or show you the algorithm's CURRENT, local behavior. A
    moving average of the raw per-round reward, in contrast, only
    looks at the last W rounds (e.g. W=50) and re-averages them at
    every step. This answers a different, complementary question:
    "how well is the algorithm doing RIGHT NOW, ignoring old history?"
    It's the difference between "what's my bank balance" (cumulative)
    and "what's my average daily spending this month" (moving
    average). For Stage 0 specifically, the moving average should
    settle into a fairly flat, stable band once ELP has mostly learned
    to avoid theta=0 -- and later, in Stage 1 (Bob), it is THE key
    diagnostic for visually spotting the exact round where performance
    drops after a regime switch and how many rounds it takes to
    recover, which cumulative regret alone hides.

Run from the project root:
    python3 -m experiments.stage0_baseline_replication.run_stage0
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.environment.dollar_auction import DollarAuction
from src.environment.strategies import build_threshold_arm_set
from src.opponents.alice_rational import Alice
from src.algorithms.elp import ELP


# ---------------------------------------------------------------------
# Experiment configurations -- add/remove entries here to compare
# different parameter regimes. Each gets its own output subfolder.
# ---------------------------------------------------------------------
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
CHECKPOINTS_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]  # as fraction of T

OUTPUT_ROOT = PROJECT_ROOT / "results" / "stage0"


def run_single_seed(cfg: dict, seed: int) -> dict:
    budget, stake, alice_mu, T = cfg["budget"], cfg["stake"], cfg["alice_mu"], cfg["T"]
    rng = np.random.default_rng(seed)

    thetas = list(range(0, budget + 1))
    arms = build_threshold_arm_set(budget=budget, stake=stake, thetas=thetas)
    n_arms = len(arms)

    env = DollarAuction(stake=stake, budget=budget)
    alice_strategy = Alice(stake=stake, budget=budget, mu=alice_mu).get_strategy()

    elp = ELP(
        strategies=arms, thetas=thetas, stake=stake, budget=budget,
        horizon_T=T, rng=rng,
    )

    hindsight_sums = [0.0] * n_arms
    realized_sum = 0.0
    rows = []
    recent_rewards: list = []

    for t in range(1, T + 1):
        agent_starts = bool(rng.integers(0, 2))

        arm = elp.select_arm()
        result = env.run(arms[arm], alice_strategy, agent_starts=agent_starts, rng=rng)
        reward = DollarAuction.normalize_reward(result.agent_payoff, budget, stake)
        elp.update(arm, reward, info={"agent_state_trace": result.agent_state_trace})
        realized_sum += reward

        for i, s in enumerate(arms):
            r_i = env.run(s, alice_strategy, agent_starts=agent_starts, rng=rng)
            hindsight_sums[i] += DollarAuction.normalize_reward(r_i.agent_payoff, budget, stake)

        best_hindsight = max(hindsight_sums)
        static_regret = best_hindsight - realized_sum

        recent_rewards.append(reward)
        if len(recent_rewards) > MOVING_AVG_WINDOW:
            recent_rewards.pop(0)
        moving_avg_reward = float(np.mean(recent_rewards))

        rows.append({
            "seed": seed,
            "round": t,
            "arm_theta": thetas[arm],
            "winner": result.winner,
            "agent_final_bid": result.agent_final_bid,
            "opponent_final_bid": result.opponent_final_bid,
            "reward": reward,
            "moving_avg_reward": moving_avg_reward,
            "cumulative_reward": realized_sum,
            "static_regret": static_regret,
        })

    checkpoints = sorted(set(max(1, int(round(f * T))) for f in CHECKPOINTS_FRACTIONS))
    checkpoint_regret = {c: rows[c - 1]["static_regret"] for c in checkpoints}

    win_count = sum(1 for r in rows if r["winner"] == "agent")

    return {
        "rows": rows,
        "final_probs": dict(zip(thetas, elp._last_probs)),
        "final_regret": rows[-1]["static_regret"],
        "checkpoint_regret": checkpoint_regret,
        "win_rate": win_count / T,
    }


def run_config(cfg: dict) -> dict:
    label = cfg["label"]
    n_seeds = cfg["n_seeds"]
    T = cfg["T"]

    out_dir = OUTPUT_ROOT / label
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    print(f"\n=== Config '{label}' (budget={cfg['budget']}, stake={cfg['stake']}) ===")

    all_rows = []
    final_probs_per_seed = []
    final_regrets = []
    win_rates = []
    checkpoint_regrets_per_seed = []

    for seed in range(n_seeds):
        result = run_single_seed(cfg, seed)
        all_rows.extend(result["rows"])
        final_probs_per_seed.append(result["final_probs"])
        final_regrets.append(result["final_regret"])
        win_rates.append(result["win_rate"])
        checkpoint_regrets_per_seed.append(result["checkpoint_regret"])
        print(f"  seed={seed:2d}  final regret={result['final_regret']:.3f}  "
              f"win_rate={result['win_rate']:.3f}")

    # --- Raw per-round CSV ---
    raw_path = out_dir / "tables" / "raw_results.csv"
    with open(raw_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    # --- Checkpoint regret table (mean +/- std across seeds) ---
    checkpoints = sorted(checkpoint_regrets_per_seed[0].keys())
    checkpoint_path = out_dir / "tables" / "checkpoint_regret.csv"
    with open(checkpoint_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["round_t", "mean_static_regret", "std_static_regret"])
        for c in checkpoints:
            vals = [d[c] for d in checkpoint_regrets_per_seed]
            writer.writerow([c, float(np.mean(vals)), float(np.std(vals))])

    # --- Final selection probability summary ---
    thetas = sorted(final_probs_per_seed[0].keys())
    summary_path = out_dir / "tables" / "summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta", "mean_final_selection_prob", "std_final_selection_prob"])
        for th in thetas:
            vals = [p[th] for p in final_probs_per_seed]
            writer.writerow([th, float(np.mean(vals)), float(np.std(vals))])

    mean_regret = float(np.mean(final_regrets))
    std_regret = float(np.std(final_regrets))
    mean_win_rate = float(np.mean(win_rates))

    print(f"  --> final regret: mean={mean_regret:.3f} std={std_regret:.3f}  "
          f"mean win_rate={mean_win_rate:.3f}")
    print(f"  Checkpoint regret (mean across seeds):")
    for c in checkpoints:
        vals = [d[c] for d in checkpoint_regrets_per_seed]
        print(f"    t={c:5d}: {np.mean(vals):.3f}")

    # --- Build regret matrix + moving-avg-reward matrix for plotting ---
    regret_matrix = np.zeros((n_seeds, T))
    moving_avg_matrix = np.zeros((n_seeds, T))
    for seed in range(n_seeds):
        seed_rows = [r for r in all_rows if r["seed"] == seed]
        regret_matrix[seed, :] = [r["static_regret"] for r in seed_rows]
        moving_avg_matrix[seed, :] = [r["moving_avg_reward"] for r in seed_rows]

    return {
        "label": label,
        "out_dir": out_dir,
        "regret_matrix": regret_matrix,
        "moving_avg_matrix": moving_avg_matrix,
        "T": T,
        "thetas": thetas,
        "final_probs_per_seed": final_probs_per_seed,
    }


def make_individual_plot(config_result: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    label = config_result["label"]
    T = config_result["T"]
    rounds = np.arange(1, T + 1)
    regret_matrix = config_result["regret_matrix"]
    moving_avg_matrix = config_result["moving_avg_matrix"]

    marker_spacing = max(1, T // 10) 

    # --- Figure 1: cumulative static regret ---
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    mean_regret = regret_matrix.mean(axis=0)
    std_regret = regret_matrix.std(axis=0)
    
    ax1.plot(rounds, mean_regret, color="#2A6F77", label="mean static regret", 
             marker='|', markevery=marker_spacing, markersize=8)
    
    ax1.fill_between(rounds, mean_regret - std_regret, mean_regret + std_regret,
                      alpha=0.2, color="#2A6F77", label="±1 std across seeds")
    
    ax1.yaxis.set_major_locator(MaxNLocator(10))
    
    ax1.set_xlabel("Round (t)  ->  1 to T, one full auction per round")
    ax1.set_ylabel("Cumulative static regret  U_A(T)\n(higher = worse; accumulates over all rounds so far)")
    ax1.set_title(f"Stage 0 [{label}]: cumulative static regret")
    ax1.legend()
    fig1.tight_layout()
    regret_path = config_result["out_dir"] / "figures" / "regret_cumulative.png"
    fig1.savefig(regret_path, dpi=150)
    plt.close(fig1)
    print(f"  Regret plot saved to: {regret_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    mean_ma = moving_avg_matrix.mean(axis=0)
    std_ma = moving_avg_matrix.std(axis=0)
    ax2.plot(rounds, mean_ma, color="#C1583A",
              label=f"moving avg reward (window={MOVING_AVG_WINDOW} rounds)",
              marker='|', markevery=marker_spacing, markersize=8)
    ax2.fill_between(rounds, mean_ma - std_ma, mean_ma + std_ma,
                      alpha=0.2, color="#C1583A")
    ax2.yaxis.set_major_locator(MaxNLocator(10))
    ax2.set_xlabel("Round (t)  ->  1 to T, one full auction per round")
    ax2.set_ylabel(f"Reward averaged over the last {MOVING_AVG_WINDOW} rounds\n"
                     f"(local/recent performance, NOT cumulative)")
    ax2.set_title(f"Stage 0 [{label}]: moving average reward (local trend)")
    ax2.legend()
    fig2.tight_layout()
    ma_path = config_result["out_dir"] / "figures" / "moving_avg_reward.png"
    fig2.savefig(ma_path, dpi=150)
    plt.close(fig2)
    print(f"  Moving average plot saved to: {ma_path}")


def make_comparison_plot(config_results: list):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#2A6F77", "#C1583A", "#6A5ACD", "#4C9A2A"]
    
    linestyles = ['-', '--', '-.', ':']

    for i, cfg_result in enumerate(config_results):
        T = cfg_result["T"]
        rounds = np.arange(1, T + 1)
        mean_regret = cfg_result["regret_matrix"].mean(axis=0)
        marker_spacing = max(1, T // 10)
        
        ax.plot(rounds, mean_regret, label=cfg_result["label"],
                 color=colors[i % len(colors)],
                 linestyle=linestyles[i % len(linestyles)], # Variasi garis
                 marker='|', markevery=marker_spacing, markersize=8)

    ax.yaxis.set_major_locator(MaxNLocator(10)) # Detail sumbu Y dinamis
    ax.set_xlabel("Round (t)")
    ax.set_ylabel("Mean cumulative static regret")
    ax.set_title("Stage 0: regret comparison across parameter configurations")
    ax.legend()
    fig.tight_layout()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    fig_path = OUTPUT_ROOT / "comparison_all_configs.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nComparison plot (all configs) saved to: {fig_path}")


def main():
    config_results = []
    for cfg in CONFIGS:
        result = run_config(cfg)
        make_individual_plot(result)
        config_results.append(result)

    make_comparison_plot(config_results)

    print("\n=== All configs done. Output layout: ===")
    for r in config_results:
        print(f"  results/stage0/{r['label']}/tables/{{raw_results,checkpoint_regret,summary}}.csv")
        print(f"  results/stage0/{r['label']}/figures/regret_cumulative.png")
        print(f"  results/stage0/{r['label']}/figures/moving_avg_reward.png")
    print("  results/stage0/comparison_all_configs.png")


if __name__ == "__main__":
    main()