"""
run_stage0.py

Stage 0 -- Baseline Replication.

Purpose (see docs/design_decisions.md and README): before extending
anything, verify our ELP implementation behaves as the theory in
Waniek et al. (2016) predicts, in the SIMPLEST possible setting:
2 players, a rational (Alice) opponent, static regret.

What "success" looks like here is specific and worth stating before
running: against a bounded-rational opponent like Alice, O'Neill's
strategy (theta = budget, i.e. no early fold) resolves the auction in
ONE bid, and because every threshold strategy with theta >= 1 makes
that exact same opening bid before any threshold can matter, ALL of
them tie for the best possible outcome against Alice. Only theta = 0
(auto-fold, never engage) is worse. So a CORRECT ELP implementation
should:
    (a) quickly push weight away from theta = 0,
    (b) NOT be able to meaningfully distinguish among theta = 1..budget
        (they are genuinely tied -- there is no "wrong" choice among
        them here), and
    (c) accumulate only a small, bounded amount of regret overall,
        coming almost entirely from residual exploration into theta=0.

This is not a weak or boring test -- it is a direct empirical
confirmation of the paper's central theoretical claim ("escalation
should never occur" against a rational opponent when using an
O'Neill-based strategy). The genuinely interesting, differentiated
learning dynamics (where different thresholds are NOT tied, and being
slow to adapt actually costs you) is deliberately left for Stage 1
(Bob), where the opponent's behavior changes over time.

Run this from the project root:
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
# Experiment configuration
# ---------------------------------------------------------------------
BUDGET = 12
STAKE = 5.0
ALICE_MU = 0.8
T = 10000             # rounds per run
N_SEEDS = 10    # independent replications (increase later for
                        # thesis-grade results; kept modest here for a
                        # first correctness check)
OUTPUT_DIR = PROJECT_ROOT / "results" / "stage0"


def run_single_seed(seed: int) -> dict:
    """
    Run one full T-round Stage 0 replication with a given seed.
    Returns a dict of per-round records plus the final regret trace.
    """
    rng = np.random.default_rng(seed)

    thetas = list(range(0, BUDGET + 1))
    arms = build_threshold_arm_set(budget=BUDGET, stake=STAKE, thetas=thetas)
    n_arms = len(arms)

    env = DollarAuction(stake=STAKE, budget=BUDGET)
    alice_strategy = Alice(stake=STAKE, budget=BUDGET, mu=ALICE_MU).get_strategy()

    elp = ELP(
        strategies=arms,
        thetas=thetas,
        stake=STAKE,
        budget=BUDGET,
        horizon_T=T,
        rng=rng,
    )

    hindsight_sums = [0.0] * n_arms
    realized_sum = 0.0
    rows = []

    for t in range(1, T + 1):
        # Draw the starting player ONCE for this round, and reuse it
        # both for the arm actually played and for every hindsight
        # evaluation below, so the hindsight comparison is apples-to-
        # apples with what really happened this round (same coin flip).
        agent_starts = bool(rng.integers(0, 2))

        arm = elp.select_arm()
        result = env.run(
            arms[arm], alice_strategy, agent_starts=agent_starts, rng=rng
        )
        reward = DollarAuction.normalize_reward(result.agent_payoff, BUDGET, STAKE)
        elp.update(arm, reward, info={"agent_state_trace": result.agent_state_trace})
        realized_sum += reward

        for i, s in enumerate(arms):
            r_i = env.run(
                s, alice_strategy, agent_starts=agent_starts, rng=rng
            )
            hindsight_sums[i] += DollarAuction.normalize_reward(
                r_i.agent_payoff, BUDGET, STAKE
            )

        best_hindsight = max(hindsight_sums)
        static_regret = best_hindsight - realized_sum

        rows.append(
            {
                "seed": seed,
                "round": t,
                "arm_theta": thetas[arm],
                "reward": reward,
                "cumulative_reward": realized_sum,
                "static_regret": static_regret,
            }
        )

    return {
        "rows": rows,
        "final_probs": dict(zip(thetas, elp._last_probs)),
        "final_regret": rows[-1]["static_regret"],
    }


def main():
    OUTPUT_DIR.joinpath("figures").mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.joinpath("tables").mkdir(parents=True, exist_ok=True)

    all_rows = []
    final_probs_per_seed = []
    final_regrets = []

    print(f"Running Stage 0: ELP vs Alice, T={T}, {N_SEEDS} seeds...")
    for seed in range(N_SEEDS):
        result = run_single_seed(seed)
        all_rows.extend(result["rows"])
        final_probs_per_seed.append(result["final_probs"])
        final_regrets.append(result["final_regret"])
        print(
            f"  seed={seed:2d}  final static regret = {result['final_regret']:.3f}"
        )

    # --- Save raw per-round results ---
    csv_path = OUTPUT_DIR / "tables" / "stage0_raw_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nRaw per-round results saved to: {csv_path}")

    # --- Save summary: mean final selection probability per theta ---
    thetas = sorted(final_probs_per_seed[0].keys())
    summary_path = OUTPUT_DIR / "tables" / "stage0_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["theta", "mean_final_selection_prob", "std_final_selection_prob"])
        for th in thetas:
            vals = [p[th] for p in final_probs_per_seed]
            writer.writerow([th, float(np.mean(vals)), float(np.std(vals))])
    print(f"Summary (per-theta final selection probability) saved to: {summary_path}")

    mean_regret = float(np.mean(final_regrets))
    std_regret = float(np.std(final_regrets))
    print(f"\nFinal static regret across {N_SEEDS} seeds: "
          f"mean={mean_regret:.3f}  std={std_regret:.3f}")

    # --- Plot: mean cumulative regret over time, averaged across seeds ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        regret_matrix = np.zeros((N_SEEDS, T))
        for seed in range(N_SEEDS):
            seed_rows = [r for r in all_rows if r["seed"] == seed]
            regret_matrix[seed, :] = [r["static_regret"] for r in seed_rows]

        mean_trace = regret_matrix.mean(axis=0)
        std_trace = regret_matrix.std(axis=0)
        rounds = np.arange(1, T + 1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(rounds, mean_trace, label="mean static regret", color="#2A6F77")
        ax.fill_between(
            rounds,
            mean_trace - std_trace,
            mean_trace + std_trace,
            alpha=0.2,
            color="#2A6F77",
            label="±1 std across seeds",
        )
        ax.set_xlabel("Round (t)")
        ax.set_ylabel("Cumulative static regret")
        ax.set_title(f"Stage 0: ELP vs Alice (static regret, {N_SEEDS} seeds)")
        ax.legend()
        fig.tight_layout()

        fig_path = OUTPUT_DIR / "figures" / "stage0_static_regret.png"
        fig.savefig(fig_path, dpi=150)
        print(f"Regret plot saved to: {fig_path}")
    except ImportError:
        print("matplotlib not available -- skipped plot generation.")

    # --- Print final selection probabilities (sanity check vs theory) ---
    print("\nMean final selection probability per theta (across seeds):")
    for th in thetas:
        vals = [p[th] for p in final_probs_per_seed]
        marker = "  <-- should be LOWEST (theta=0 is the only bad arm)" if th == 0 else ""
        print(f"  theta={th:2d}: {np.mean(vals):.4f}{marker}")


if __name__ == "__main__":
    main()