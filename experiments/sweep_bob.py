"""
sweep_bob.py  --  choose Bob / auction parameters for Stage 1.

PART A  (regime separation)   `--part A`
    Grid over (budget, stake, Bob mu, prior_sunk_cost, escalation_rate).
    For every config: pin Bob to Rational / Escalating, play every fixed
    theta, compute separation_score (see src/simulation/calibration.py).
    Coarse pass over the whole grid, then a finer re-measurement of the
    top configs. Keeps only configs where Rational Bob folds at O'Neill's
    opening bid (x0 >= mu*stake), because otherwise the rational regime is
    already a war and the regimes cannot be told apart by what theta wins.
    -> results/sweeps/regime_separation.csv

PART B  (adaptive dynamics)   `--part B`
    Take ONE config (CLI flags) and sweep the ADAPTIVE parameters of Bob
    (window W, min_dwell, gain) against a real learner (EXP3 or ELP).
    Reports how many regime phases appear, how long they last, how often
    Bob escalates, and whether the learner plays differently in the two
    regimes. -> results/sweeps/bob_dynamics.csv

Run from the project root:
    python -m experiments.sweep_bob --part A
    python -m experiments.sweep_bob --part B --budget 12 --stake 8 --mu 0.5 --prior 3 --rate 3
    python -m experiments.sweep_bob --part all --quick
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
import warnings
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation.calibration import arm_rewards, opening_bid, separation_score
from src.simulation.config import SimConfig
from src.simulation.runner import make_simulation

try:
    from tqdm import tqdm
except ImportError:                      # pragma: no cover
    def tqdm(it=None, **kw):
        return it
    tqdm.write = print

OUT_DIR = PROJECT_ROOT / "results" / "sweeps"
THR = 0.5   # Bob's sunk-cost threshold; with prior_sunk_cost > 0 it is always exceeded


# ----------------------------------------------------------------------
# Part A
# ----------------------------------------------------------------------

def part_a(quick: bool, top_k: int = 12):
    budgets = (12, 14) if quick else (10, 12, 14, 16, 20)
    stakes = (8.0, 10.0) if quick else (5.0, 8.0, 10.0, 12.0)
    mus = (0.5,) if quick else (0.4, 0.5, 0.6, 0.8)
    priors = (1.0, 3.0)
    rates = (1.0, 3.0)
    T0, S0 = (150, 2) if quick else (250, 3)

    grid = []
    for b, s, mu, prior, rate in itertools.product(budgets, stakes, mus, priors, rates):
        if opening_bid(b, s) < mu * s:
            continue                    # rational Bob would not fold at x0
        grid.append((b, s, mu, prior, rate))
    print(f"Part A: {len(grid)} configs (coarse T={T0}, seeds={S0})")

    def evaluate(b, s, mu, prior, rate, T, seeds):
        kw = dict(mu=mu, sunk_cost_threshold=THR, prior_sunk_cost=prior,
                  escalation_rate=rate, escalation_ceiling=float(b))
        rr = arm_rewards(b, s, kw, False, T=T, seeds=seeds)
        re = arm_rewards(b, s, kw, True, T=T, seeds=seeds)
        return separation_score(rr, re), rr, re

    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for cfg in tqdm(grid, desc="coarse"):
            sc, _, _ = evaluate(*cfg, T0, S0)
            rows.append((cfg, sc))
        rows.sort(key=lambda r: -r[1]["score"])

        refined = []
        for cfg, _ in tqdm(rows[:top_k], desc="refine"):
            sc, rr, re = evaluate(*cfg, 1000 if not quick else 300,
                                  6 if not quick else 3)
            refined.append((cfg, sc, rr, re))
        refined.sort(key=lambda r: -r[1]["score"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "regime_separation.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "budget", "stake", "bob_mu", "prior_sunk_cost",
                    "escalation_rate", "x0", "score", "A", "B",
                    "max_reward_rational", "max_reward_escalating",
                    "best_arms_rational", "best_arms_escalating"])
        for tag, lst in (("coarse", [(c, s) for c, s in rows]),
                         ("refined", [(c, s) for c, s, _, _ in refined])):
            for (b, s_, mu, prior, rate), sc in lst:
                w.writerow([tag, b, s_, mu, prior, rate, opening_bid(b, s_),
                            round(sc["score"], 4), round(sc["A"], 4), round(sc["B"], 4),
                            round(sc["max_rational"], 4), round(sc["max_escalating"], 4),
                            sc["best_rational"], sc["best_escalating"]])

    print("\nTop configs (refined). score = min(A, B), reward units in [0,1]:")
    print(" score    A      B    | b   s    mu   prior rate | best arms Rational -> Escalating")
    for (b, s_, mu, prior, rate), sc, rr, re in refined:
        br, be = sc["best_rational"], sc["best_escalating"]
        fmt = lambda a: (f"{a[0]}..{a[-1]}" if len(a) > 3 else str(a))
        print(f" {sc['score']:.3f}  {sc['A']:.3f}  {sc['B']:.3f} | {b:2d}  {s_:4.1f}  {mu:.1f}  "
              f"{prior:4.1f}  {rate:3.1f} | {fmt(br)} -> {fmt(be)}")
    print(f"\nSaved {path.relative_to(PROJECT_ROOT)}")
    print("Rule of thumb: pick a config with score >= ~0.10 and a modest budget "
          "(fewer arms = cheaper hindsight later).")


# ----------------------------------------------------------------------
# Part B
# ----------------------------------------------------------------------

def run_lengths(flags):
    runs, cur, n = [], flags[0], 1
    for v in flags[1:]:
        if v == cur:
            n += 1
        else:
            runs.append((cur, n)); cur, n = v, 1
    runs.append((cur, n))
    return runs


def part_b(args):
    quick = args.quick
    windows = (10, 50) if quick else (10, 25, 50, 100)
    dwells = (1, 25) if quick else (1, 10, 25, 50)
    gains = (1.0,) if quick else (1.0, 1.5)
    T = 1500 if quick else 4000
    seeds = 2 if quick else 3

    print(f"Part B: b={args.budget} s={args.stake} mu={args.mu} prior={args.prior} "
          f"rate={args.rate} learner={args.algo} T={T} seeds={seeds}")
    rows = []
    combos = list(itertools.product(windows, dwells, gains))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for W, D, g in tqdm(combos, desc="dynamics"):
            stats = []
            for seed in range(seeds):
                cfg = SimConfig(
                    budget=args.budget, stake=args.stake, T=T, seed=seed,
                    bob_mu=args.mu, bob_window=W, bob_gain=g, bob_min_dwell=D,
                    bob_sunk_cost_threshold=THR, bob_prior_sunk_cost=args.prior,
                    bob_escalation_rate=args.rate, keep_snapshots=False)
                sim = make_simulation(cfg, args.algo)
                recs = sim.run()
                esc = np.array([r["bob_escalating"] for r in recs])
                th = np.array([r["theta"] for r in recs], dtype=float)
                rw = np.array([r["reward"] for r in recs])
                runs = run_lengths(esc.tolist())
                esc_len = [n for v, n in runs if v]
                rat_len = [n for v, n in runs if not v]
                stats.append({
                    "phases": len(runs),
                    "frac_esc": esc.mean(),
                    "mean_len_esc": np.mean(esc_len) if esc_len else 0.0,
                    "mean_len_rat": np.mean(rat_len) if rat_len else 0.0,
                    "theta_in_esc": th[esc].mean() if esc.any() else np.nan,
                    "theta_in_rat": th[~esc].mean() if (~esc).any() else np.nan,
                    "reward": rw.mean(),
                })
            agg = {k: float(np.nanmean([s[k] for s in stats])) for k in stats[0]}
            rows.append((W, D, g, agg))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "bob_dynamics.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window", "min_dwell", "gain", "phases", "frac_escalating",
                    "mean_len_escalating", "mean_len_rational",
                    "mean_theta_in_escalating", "mean_theta_in_rational",
                    "mean_reward"])
        for W, D, g, a in rows:
            w.writerow([W, D, g, round(a["phases"], 1), round(a["frac_esc"], 3),
                        round(a["mean_len_esc"], 1), round(a["mean_len_rat"], 1),
                        round(a["theta_in_esc"], 2), round(a["theta_in_rat"], 2),
                        round(a["reward"], 4)])

    print("\n  W  dwell gain | phases  esc%  len_esc len_rat | theta@esc theta@rat | reward")
    for W, D, g, a in rows:
        print(f"{W:3d}  {D:4d}  {g:.1f} | {a['phases']:6.1f} {100*a['frac_esc']:5.1f} "
              f"{a['mean_len_esc']:7.1f} {a['mean_len_rat']:7.1f} | "
              f"{a['theta_in_esc']:8.2f} {a['theta_in_rat']:8.2f} | {a['reward']:.3f}")
    print(f"\nSaved {path.relative_to(PROJECT_ROOT)}")
    print("Read it as: several phases (not 1, not hundreds), a real share of "
          "escalation, and theta@esc clearly below theta@rat (the learner reacts).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "all"], default="all")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--budget", type=int, default=12)
    ap.add_argument("--stake", type=float, default=8.0)
    ap.add_argument("--mu", type=float, default=0.5)
    ap.add_argument("--prior", type=float, default=3.0)
    ap.add_argument("--rate", type=float, default=3.0)
    ap.add_argument("--algo", choices=["exp3", "elp"], default="exp3")
    args = ap.parse_args()
    if args.part in ("A", "all"):
        part_a(args.quick)
    if args.part in ("B", "all"):
        part_b(args)


if __name__ == "__main__":
    main()