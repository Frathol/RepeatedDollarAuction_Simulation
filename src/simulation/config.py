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