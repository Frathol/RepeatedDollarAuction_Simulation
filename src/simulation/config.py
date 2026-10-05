"""
config.py

Single place for experiment parameters (README: avoid hardcoding).
Everything stochastic is derived from `seed` (see runner.make_simulation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class SimConfig:
    # --- auction -------------------------------------------------------
    budget: int = 12
    stake: float = 10.0
    T: int = 2000
    seed: int = 0

    # --- Bob (adaptive adversary, design_decisions sec. 13) ------------
    bob_mu: float = 0.8
    bob_window: int = 50
    bob_gain: float = 1.0
    bob_min_dwell: int = 1                 # >1 => persistent phases
    bob_min_history: Optional[int] = None  # None => wait for a full window
    bob_fixed_regime: Optional[bool] = None  # calibration/sweeps only (oracle Bob)
    bob_escalation_ceiling: Optional[float] = None  # None => budget
    bob_prior_sunk_cost: float = 0.0       # investment Bob brings into each auction
    bob_escalation_rate: float = 0.5
    bob_sunk_cost_threshold: Optional[float] = None   # None => 0.5*mu*stake
    bob_signal: str = "theta"              # "theta" | "final_bid" (divided by budget)

    # --- Page-Hinkley (must be calibrated per configuration!) ----------
    ph_delta: float = 0.02
    ph_threshold: float = 1.0
    ph_alpha: float = 0.05
    ph_warmup: int = 20
    ph_direction: str = "decrease"

    # --- bookkeeping ---------------------------------------------------
    keep_snapshots: bool = True            # needed for per-phase hindsight


def stage1_config(**overrides) -> "SimConfig":
    """
    Recommended STARTING POINT for Stage 1 (adaptive Bob, 1v1), chosen with
    experiments/sweep_bob.py: the best arm differs between Bob's regimes by
    a wide margin (cross-regime loss ~0.15, see calibrate_regimes.py).
    Adaptive parameters (window, min_dwell, gain) are NOT yet validated --
    see README "Open decisions" #8 before treating them as final.
    """
    base = dict(
        budget=12, stake=8.0, T=3000,
        bob_mu=0.5, bob_sunk_cost_threshold=0.5, bob_prior_sunk_cost=3.0,
        bob_escalation_rate=3.0,
        bob_window=25, bob_min_dwell=25, bob_gain=1.0,
    )
    base.update(overrides)
    return SimConfig(**base)