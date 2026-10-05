"""
run_stage0.py

Stage 0 -- Baseline replication: ELP vs. the rational, oblivious opponent
(Alice) in Waniek et al.'s setting, static regret against the best fixed
threshold arm in hindsight.

This script is STANDALONE (own loop, no runner.py / Mesa): the opponent is
oblivious, so per-round counterfactuals ("what would arm g have earned this
round") are valid and cheap. It is the BASELINE for all later stages:
figures are produced by the shared `src/metrics/plots.py`, so later stages
produce the same set and only add layers on top.

Outputs per configuration (results/stage0/<label>/):
    tables/raw_results.csv         one row per (seed, round)
    tables/checkpoint_regret.csv   regret at fractions of T (mean/std over seeds)
    tables/summary.csv             final selection prob + hindsight reward per theta
    figures/*.png                  standard set (see plots.py) incl. regret_vs_bound
results/stage0/comparison_regret.png, comparison_reward.png

Run from the project root (folder name = whatever yours is called):
    python -m experiments.stage0_baseline_replication.run_stage0
    python -m experiments.stage0_baseline_replication.run_stage0 --quick
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.algorithms.elp import ELP
from src.environment.dollar_auction import DollarAuction
from src.environment.strategies import build_threshold_arm_set
from src.metrics.plots import (RunBundle, moving_average, plot_comparison,
                               save_fig, save_standard_report)
from src.opponents.alice_rational import Alice

try:                                    # progress bars are optional
    from tqdm import tqdm
except ImportError:                     # pragma: no cover
    def tqdm(it=None, **kwargs):
        return it
    tqdm.write = print


CONFIGS = [
    {"label": "budget12_stake5", "budget": 12, "stake": 5.0,
     "alice_mu": 0.8, "T": 10000, "n_seeds": 8},
    {"label": "budget8_stake10", "budget": 8, "stake": 10.0,
     "alice_mu": 0.8, "T": 10000, "n_seeds": 8},
]

MOVING_AVG_WINDOW = 50
CHECKPOINT_FRACTIONS = [0.1, 0.25, 0.5, 0.75, 1.0]


def run_single_seed(cfg: dict, seed: int) -> dict:
    budget, stake, alice_mu, T = cfg["budget"], cfg["stake"], cfg["alice_mu"], cfg["T"]
    rng = np.random.default_rng(seed)

    thetas = list(range(0, budget + 1))
    arms = build_threshold_arm_set(budget=budget, stake=stake, thetas=thetas)
    n_arms = len(arms)

    env = DollarAuction(stake=stake, budget=budget)
    alice_strategy = Alice(stake=stake, budget=budget, mu=alice_mu).get_strategy()
    elp = ELP(strategies=arms, thetas=thetas, stake=stake, budget=budget,
              horizon_T=T, rng=rng)

    hindsight_sums = np.zeros(n_arms)
    realized_sum = 0.0

    reward = np.zeros(T)
    regret = np.zeros(T)
    arm_theta = np.zeros(T, dtype=int)
    agent_won = np.zeros(T, dtype=bool)
    agent_bid = np.zeros(T)
    opp_bid = np.zeros(T)

    for t in tqdm(range(T), desc=f"  seed {seed} rounds", leave=False):
        agent_starts = bool(rng.integers(0, 2))

        arm = elp.select_arm()
        result = env.run(arms[arm], alice_strategy,
                         agent_starts=agent_starts, rng=rng)
        r = DollarAuction.normalize_reward(result.agent_payoff, budget, stake)
        elp.update(arm, r, info={"agent_state_trace": result.agent_state_trace})
        realized_sum += r

        # Hindsight: valid here because Alice is oblivious (her behaviour
        # does not depend on the learner's past play).
        for i, s in enumerate(arms):
            r_i = env.run(s, alice_strategy, agent_starts=agent_starts, rng=rng)
            hindsight_sums[i] += DollarAuction.normalize_reward(
                r_i.agent_payoff, budget, stake)

        reward[t] = r
        regret[t] = hindsight_sums.max() - realized_sum
        arm_theta[t] = thetas[arm]
        agent_won[t] = result.winner == "agent"
        agent_bid[t] = result.agent_final_bid
        opp_bid[t] = result.opponent_final_bid

    return {
        "reward": reward, "regret": regret, "arm_theta": arm_theta,
        "agent_won": agent_won, "agent_bid": agent_bid, "opp_bid": opp_bid,
        "final_probs": np.array(elp._last_probs),
        "hindsight": hindsight_sums,
        "elp_beta": elp.beta, "elp_epsilon": getattr(elp, "epsilon", None),
        "n_arms": n_arms,
    }


def theorem6_bound(T: int, n_arms: int, beta: float, epsilon: float) -> np.ndarray:
    """
    Upper bound on ELP's regret for the threshold family (alpha(G)=1), for
    the beta actually used (Theorem 6 / Thm 2 with alpha = 1):

        U(t) <= 9 * beta * t * log(6|S0| / eps) + log|S0| / beta

    Valid for every t simultaneously because beta is fixed. Normalized
    rewards in [0,1], so the units match our regret.
    """
    t = np.arange(1, T + 1)
    return 9.0 * beta * t * math.log(6 * n_arms / epsilon) + math.log(n_arms) / beta


def run_config(cfg: dict, out_root: Path) -> RunBundle:
    label, n_seeds, T = cfg["label"], cfg["n_seeds"], cfg["T"]
    out_dir = out_root / label
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)

    tqdm.write(f"\n=== Config '{label}' (budget={cfg['budget']}, "
               f"stake={cfg['stake']}, T={T}, seeds={n_seeds}) ===")

    runs = []
    for seed in tqdm(range(n_seeds), desc=label):
        res = run_single_seed(cfg, seed)
        runs.append(res)
        tqdm.write(f"  seed={seed:2d}  final regret={res['regret'][-1]:.3f}  "
                   f"win_rate={res['agent_won'].mean():.3f}")

    stack = lambda k: np.stack([r[k] for r in runs])
    thetas = list(range(0, cfg["budget"] + 1))

    bound = None
    eps = runs[0]["elp_epsilon"]
    if eps:
        bound = theorem6_bound(T, runs[0]["n_arms"], runs[0]["elp_beta"], eps)

    bundle = RunBundle(
        label=label, thetas=thetas,
        reward=stack("reward"), regret=stack("regret"),
        arm_theta=stack("arm_theta"), agent_won=stack("agent_won"),
        agent_bid=stack("agent_bid"), opp_bid=stack("opp_bid"),
        final_probs=stack("final_probs"), hindsight_per_arm=stack("hindsight"),
        regret_name="static regret", bound=bound,
    )

    # ---- raw per-round CSV (same columns as before) --------------------
    ma = moving_average(bundle.reward, MOVING_AVG_WINDOW)
    cum_reward = np.cumsum(bundle.reward, axis=1)
    raw_path = out_dir / "tables" / "raw_results.csv"
    with open(raw_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["seed", "round", "arm_theta", "winner", "agent_final_bid",
                    "opponent_final_bid", "reward", "moving_avg_reward",
                    "cumulative_reward", "static_regret"])
        for s in range(n_seeds):
            for t in range(T):
                w.writerow([s, t + 1, bundle.arm_theta[s, t],
                            "agent" if bundle.agent_won[s, t] else "opponent",
                            bundle.agent_bid[s, t], bundle.opp_bid[s, t],
                            bundle.reward[s, t], ma[s, t], cum_reward[s, t],
                            bundle.regret[s, t]])

    # ---- checkpoint regret ---------------------------------------------
    checkpoints = sorted({max(1, int(round(fr * T))) for fr in CHECKPOINT_FRACTIONS})
    with open(out_dir / "tables" / "checkpoint_regret.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round_t", "mean_static_regret", "std_static_regret",
                    "theoretical_bound"])
        for c in checkpoints:
            vals = bundle.regret[:, c - 1]
            w.writerow([c, float(vals.mean()), float(vals.std()),
                        float(bound[c - 1]) if bound is not None else ""])

    # ---- per-theta summary ---------------------------------------------
    with open(out_dir / "tables" / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["theta", "mean_final_selection_prob", "std_final_selection_prob",
                    "mean_hindsight_reward_per_round"])
        for i, th in enumerate(thetas):
            w.writerow([th, float(bundle.final_probs[:, i].mean()),
                        float(bundle.final_probs[:, i].std()),
                        float(bundle.hindsight_per_arm[:, i].mean() / T)])

    tqdm.write(f"  --> final regret: mean={bundle.regret[:, -1].mean():.3f} "
               f"std={bundle.regret[:, -1].std():.3f}  "
               f"win_rate={bundle.agent_won.mean():.3f}")
    if bound is not None:
        tqdm.write(f"      theoretical bound at T: {bound[-1]:.1f}")
    for c in checkpoints:
        tqdm.write(f"    t={c:5d}: regret={bundle.regret[:, c - 1].mean():.3f}")

    for p in save_standard_report(bundle, out_dir, window=MOVING_AVG_WINDOW):
        tqdm.write(f"  saved {p.relative_to(PROJECT_ROOT)}")
    return bundle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="tiny run (T=500, 2 seeds) into results/stage0_quick")
    args = ap.parse_args()

    configs = [dict(c) for c in CONFIGS]
    out_root = PROJECT_ROOT / "results" / "stage0"
    if args.quick:
        out_root = PROJECT_ROOT / "results" / "stage0_quick"
        for c in configs:
            c["T"], c["n_seeds"] = 500, 2

    bundles = [run_config(c, out_root) for c in configs]

    out_root.mkdir(parents=True, exist_ok=True)
    save_fig(plot_comparison(bundles, "regret",
                             title="Stage 0: regret across configurations"),
             out_root / "comparison_regret.png")
    save_fig(plot_comparison(bundles, "reward", window=MOVING_AVG_WINDOW,
                             title="Stage 0: moving-average reward across configurations"),
             out_root / "comparison_reward.png")
    tqdm.write(f"\nDone. Outputs in {out_root.relative_to(PROJECT_ROOT)}/")


if __name__ == "__main__":
    main()