"""
plots.py

ONE shared plot set for every stage, so Stage 0 (ELP vs. rational Alice,
the baseline) and every later stage produce visually identical figures
that can be compared side by side.

Rule of thumb
-------------
    * The STANDARD set (below) is produced by `save_standard_report` for
      every stage with the same file names, colours and axes.
    * A later stage ADDS to it by filling optional fields of `RunBundle`
      (e.g. `regime`, `alarms`) -- the same functions then draw extra
      layers (Bob's regime panel, Page-Hinkley alarms). It never
      re-implements a standard plot.
    * Stage-specific EXTRA figures (detection-delay histogram, per-phase
      regret, price trace + bid caps for the EC2 stage, ...) are separate
      functions added to this module later.

Standard figures (file names are fixed)
---------------------------------------
    regret_cumulative.png   mean +- std cumulative regret
    regret_normalized.png   regret / sqrt(t)   (flat  <=>  ~ sqrt(T) growth)
    regret_vs_bound.png     regret vs. theoretical bound (log-log; only if
                            `bound` is given -- Stage 0 provides Thm 6's)
    moving_avg_reward.png   moving-average reward (+ regime panel and PH
                            alarms if `regime` / `alarms` are given)
    arm_distribution.png    final selection probability per theta  |  mean
                            hindsight reward per theta (is the learner's
                            favourite arm the best one in hindsight?)
    arm_heatmap.png         which theta is played when (theta x time)
    outcomes.png            win rate and mean final bids over time
                            (escalation shows up as rising bids)

Data layout: every array is (n_seeds, T) unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

C_MAIN = "#2A6F77"
C_ALT = "#C1583A"
C_THIRD = "#6A5ACD"
C_FOURTH = "#4C9A2A"
PALETTE = [C_MAIN, C_ALT, C_THIRD, C_FOURTH]
LINESTYLES = ["-", "--", "-.", ":"]


@dataclass
class RunBundle:
    """Everything the standard plots need, for one configuration."""

    label: str
    thetas: Sequence[int]
    reward: np.ndarray                 # (S, T) per-round normalized reward
    regret: np.ndarray                 # (S, T) cumulative regret
    arm_theta: np.ndarray              # (S, T) chosen theta each round
    agent_won: np.ndarray              # (S, T) bool
    agent_bid: np.ndarray              # (S, T) agent's final bid
    opp_bid: np.ndarray                # (S, T) opponent's final bid
    final_probs: np.ndarray            # (S, K) final selection prob per arm
    hindsight_per_arm: Optional[np.ndarray] = None   # (S, K) total hindsight reward
    regret_name: str = "static regret"
    bound: Optional[np.ndarray] = None  # (T,) theoretical upper bound on regret
    # ---- optional layers added by later stages -------------------------
    regime: Optional[np.ndarray] = None            # (S, T) bool: Bob escalating
    alarms: Optional[List[List[int]]] = None       # PH alarm rounds, per seed
    extras: Dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return self.reward.shape[1]

    @property
    def n_seeds(self) -> int:
        return self.reward.shape[0]


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Row-wise moving average; early entries average what is available."""
    x = np.asarray(x, dtype=float)
    c = np.cumsum(x, axis=1)
    out = np.empty_like(c)
    w = min(window, x.shape[1])
    out[:, :w] = c[:, :w] / np.arange(1, w + 1)
    if x.shape[1] > w:
        out[:, w:] = (c[:, w:] - c[:, :-w]) / w
    return out


def _mean_band(ax, y: np.ndarray, color, label, rounds=None, ls="-", band=True):
    rounds = np.arange(1, y.shape[1] + 1) if rounds is None else rounds
    m, s = y.mean(axis=0), y.std(axis=0)
    step = max(1, y.shape[1] // 10)
    ax.plot(rounds, m, color=color, label=label, linestyle=ls,
            marker="|", markevery=step, markersize=8)
    if band and y.shape[0] > 1:
        ax.fill_between(rounds, m - s, m + s, alpha=0.2, color=color,
                        label="±1 std across seeds")


def save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ----------------------------------------------------------------------
# standard figures
# ----------------------------------------------------------------------

def plot_cumulative_regret(b: RunBundle):
    fig, ax = plt.subplots(figsize=(8, 5))
    _mean_band(ax, b.regret, C_MAIN, f"mean {b.regret_name}")
    ax.yaxis.set_major_locator(MaxNLocator(10))
    ax.set_xlabel("Round t (one auction per round)")
    ax.set_ylabel(f"Cumulative {b.regret_name}\n(higher = worse)")
    ax.set_title(f"[{b.label}] cumulative {b.regret_name}")
    ax.legend()
    return fig


def plot_regret_normalized(b: RunBundle):
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(1, b.T + 1)
    _mean_band(ax, b.regret / np.sqrt(t), C_THIRD, "regret / sqrt(t)")
    ax.set_xlabel("Round t")
    ax.set_ylabel("regret / sqrt(t)")
    ax.set_title(f"[{b.label}] regret / sqrt(t)  (flat curve ~ sqrt(T) regret)")
    ax.legend()
    return fig


def plot_regret_vs_bound(b: RunBundle):
    fig, ax = plt.subplots(figsize=(8, 5))
    t = np.arange(1, b.T + 1)
    m = b.regret.mean(axis=0)
    m = np.where(m > 0, m, np.nan)          # log axis: skip exact zeros
    ax.plot(t, m, color=C_MAIN, label=f"empirical mean {b.regret_name}")
    ax.plot(t, b.bound, color=C_ALT, linestyle="--", label="theoretical bound")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Round t (log)")
    ax.set_ylabel("Regret (log)")
    ax.set_title(f"[{b.label}] empirical regret vs. bound")
    ax.legend()
    return fig


def plot_moving_reward(b: RunBundle, window: int = 50):
    has_regime = b.regime is not None
    if has_regime:
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(8, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]})
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        axr = None
    ma = moving_average(b.reward, window)
    _mean_band(ax, ma, C_ALT, f"moving avg reward (window={window})")
    ax.yaxis.set_major_locator(MaxNLocator(10))
    ax.set_ylabel(f"Reward, last {window} rounds\n(local performance)")
    ax.set_title(f"[{b.label}] moving-average reward")
    ax.legend()
    if has_regime:
        rounds = np.arange(1, b.T + 1)
        frac = moving_average(b.regime.astype(float), window).mean(axis=0)
        axr.fill_between(rounds, 0, frac, color=C_THIRD, alpha=0.5,
                         label="P(Bob escalating)")
        if b.alarms:
            first = True
            for seed_alarms in b.alarms:
                for r in seed_alarms:
                    axr.axvline(r, color="k", alpha=0.08, linewidth=0.8,
                                label="PH alarm (each seed)" if first else None)
                    first = False
        axr.set_ylim(0, 1)
        axr.set_ylabel("Bob regime")
        axr.set_xlabel("Round t")
        axr.legend(loc="upper right", fontsize=8)
    else:
        ax.set_xlabel("Round t (one auction per round)")
    return fig


def plot_arm_distribution(b: RunBundle):
    thetas = list(b.thetas)
    has_h = b.hindsight_per_arm is not None
    fig, axes = plt.subplots(1, 2 if has_h else 1,
                             figsize=(11 if has_h else 6, 4.5), squeeze=False)
    ax = axes[0, 0]
    ax.bar(thetas, b.final_probs.mean(axis=0), yerr=b.final_probs.std(axis=0),
           color=C_MAIN, capsize=2)
    ax.set_xlabel("threshold theta (arm)")
    ax.set_ylabel("final selection probability")
    ax.set_title("What the learner ends up playing")
    if has_h:
        ax2 = axes[0, 1]
        per_round = b.hindsight_per_arm.mean(axis=0) / b.T
        # several arms can tie exactly (e.g. every theta above the opponent's
        # fold point behaves identically) -> highlight ALL of them
        best = set(np.where(per_round >= per_round.max() - 1e-9)[0].tolist())
        colors = [C_ALT if i in best else C_MAIN for i in range(len(thetas))]
        ax2.bar(thetas, per_round, color=colors)
        ax2.set_xlabel("threshold theta (arm)")
        ax2.set_ylabel("mean reward per round if played always")
        lo_t, hi_t = thetas[min(best)], thetas[max(best)]
        ax2.set_title(f"Best in hindsight: theta={lo_t}" if len(best) == 1
                      else f"Best in hindsight (tie): theta={lo_t}..{hi_t}")
        lo = per_round.min()
        ax2.set_ylim(max(0.0, lo - 0.05), per_round.max() + 0.02)
    fig.suptitle(f"[{b.label}]")
    return fig


def plot_arm_heatmap(b: RunBundle, n_bins: int = 50):
    thetas = list(b.thetas)
    index = {th: i for i, th in enumerate(thetas)}
    S, T = b.arm_theta.shape
    n_bins = min(n_bins, T)
    edges = np.linspace(0, T, n_bins + 1).astype(int)
    grid = np.zeros((len(thetas), n_bins))
    idx = np.vectorize(index.get)(b.arm_theta)
    for k in range(n_bins):
        chunk = idx[:, edges[k]:edges[k + 1]].ravel()
        if chunk.size:
            grid[:, k] = np.bincount(chunk, minlength=len(thetas)) / chunk.size
    fig, ax = plt.subplots(figsize=(9, 4.5))
    im = ax.imshow(grid, aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, T, -0.5, len(thetas) - 0.5], vmin=0, vmax=1)
    ax.set_yticks(range(len(thetas)))
    ax.set_yticklabels(thetas)
    ax.set_xlabel("Round t")
    ax.set_ylabel("threshold theta")
    ax.set_title(f"[{b.label}] arm play frequency over time (mean over seeds)")
    fig.colorbar(im, ax=ax, label="fraction of rounds")
    return fig


def plot_outcomes(b: RunBundle, window: int = 50):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    _mean_band(ax1, moving_average(b.agent_won.astype(float), window),
               C_MAIN, "learner win rate", band=False)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel(f"win rate (last {window})")
    ax1.set_title(f"[{b.label}] outcomes over time")
    ax1.legend()
    _mean_band(ax2, moving_average(b.agent_bid, window), C_MAIN,
               "learner final bid", band=False)
    _mean_band(ax2, moving_average(b.opp_bid, window), C_ALT,
               "opponent final bid", ls="--", band=False)
    ax2.set_xlabel("Round t")
    ax2.set_ylabel(f"mean final bid (last {window})")
    ax2.legend()
    return fig


def plot_comparison(bundles: Sequence[RunBundle], what: str = "regret",
                    window: int = 50, title: Optional[str] = None):
    """Overlay several bundles (configs, algorithms, stages) on one axis."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, b in enumerate(bundles):
        y = b.regret if what == "regret" else moving_average(b.reward, window)
        _mean_band(ax, y, PALETTE[i % len(PALETTE)], b.label,
                   ls=LINESTYLES[i % len(LINESTYLES)], band=False)
    ax.yaxis.set_major_locator(MaxNLocator(10))
    ax.set_xlabel("Round t")
    ax.set_ylabel("Mean cumulative regret" if what == "regret"
                  else f"Mean reward (window={window})")
    ax.set_title(title or f"Comparison: {what}")
    ax.legend()
    return fig


# ----------------------------------------------------------------------
# the standard report
# ----------------------------------------------------------------------

def save_standard_report(b: RunBundle, out_dir: Path, window: int = 50) -> List[Path]:
    """Write the standard figure set into <out_dir>/figures/."""
    fig_dir = Path(out_dir) / "figures"
    paths = [
        save_fig(plot_cumulative_regret(b), fig_dir / "regret_cumulative.png"),
        save_fig(plot_regret_normalized(b), fig_dir / "regret_normalized.png"),
        save_fig(plot_moving_reward(b, window), fig_dir / "moving_avg_reward.png"),
        save_fig(plot_arm_distribution(b), fig_dir / "arm_distribution.png"),
        save_fig(plot_arm_heatmap(b), fig_dir / "arm_heatmap.png"),
        save_fig(plot_outcomes(b, window), fig_dir / "outcomes.png"),
    ]
    if b.bound is not None:
        paths.append(save_fig(plot_regret_vs_bound(b), fig_dir / "regret_vs_bound.png"))
    return paths