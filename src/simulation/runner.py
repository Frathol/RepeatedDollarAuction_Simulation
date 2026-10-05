"""
runner.py

Phase 1 simulation engine: learner (MAB over threshold arms) vs Bob
(adaptive adversary), T sequential auctions.

Structure (two layers on purpose)
---------------------------------
1. AuctionSimulation  -- pure Python/numpy core loop. All simulation
   logic and all randomness live here. Fully testable without Mesa.
2. AuctionModel (mesa.Model) -- THIN wrapper: calls sim.step() once per
   Mesa step and lets mesa.DataCollector log the result. Mesa is used
   for orchestration + logging only (project constraint), never for
   the auction/bandit logic.

Mesa 3.0+ conventions followed: agents are created as Agent(model) (no
unique_id argument), no mesa.time schedulers, step = model.step().

Per-round order (this order is what keeps Bob's regime a function of
history up to t-1 only -- design_decisions sec. 13/14)
-----------------------------------------------------------------------
    pre   = bob.snapshot()                  # state BEFORE the regime draw
    bob.begin_round()                       # regime for round t
    (if regime changed: store `pre` as a phase-start snapshot)
    arm   = algorithm.select_arm()
    result= env.run(arm_strategy, bob.get_strategy(), agent_starts)
    reward= normalize(result.agent_payoff)
    algorithm.update(arm, reward, info={"agent_state_trace": ...})
    bob.observe_agent_move(signal)          # now Bob "sees" round t
    detector.update(reward) -> if alarm: algorithm.reset()

Randomness: three INDEPENDENT streams spawned from one SeedSequence
(learner, Bob, environment/starting player). Changing Bob's parameters
therefore cannot change the learner's or the starter's random numbers.
Hindsight code must never touch these live generators; it works on
bob.snapshot() copies (kept in `phase_starts`).

Regret is NOT computed here. Adaptive Bob means hindsight needs full
re-simulation per candidate arm per phase (sec. 16) -- that belongs in
src/metrics/regret.py, which consumes `records` (incl. agent_starts and
realized regimes) and `phase_starts` produced here.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from src.algorithms.base import BanditAlgorithm
from src.algorithms.page_hinkley import PageHinkleyDetector
from src.environment.dollar_auction import DollarAuction
from src.environment.strategies import build_threshold_arm_set
from src.opponents.bob_sunkcost import Bob
from src.simulation.config import SimConfig


def _supports_reset(algorithm: BanditAlgorithm) -> bool:
    return type(algorithm).reset is not BanditAlgorithm.reset


class AuctionSimulation:
    """Core loop. Build it with `make_simulation` unless you need custom parts."""

    def __init__(
        self,
        cfg: SimConfig,
        algorithm: BanditAlgorithm,
        thetas: List[int],
        arms: list,
        bob: Bob,
        env: DollarAuction,
        env_rng: np.random.Generator,
        detector: Optional[PageHinkleyDetector] = None,
    ) -> None:
        if detector is not None and not _supports_reset(algorithm):
            raise ValueError(
                f"{algorithm.name} does not implement reset(); it cannot be "
                "combined with a Page-Hinkley detector."
            )
        if cfg.bob_signal not in ("theta", "final_bid"):
            raise ValueError("cfg.bob_signal must be 'theta' or 'final_bid'")

        self.cfg = cfg
        self.algorithm = algorithm
        self.thetas = thetas
        self.arms = arms
        self.bob = bob
        self.env = env
        self.env_rng = env_rng
        self.detector = detector

        self.t = 0
        self.cumulative_reward = 0.0
        self.records: List[Dict] = []
        # (round_index, Bob snapshot taken BEFORE that round's regime draw)
        self.phase_starts: List[Tuple[int, Bob]] = []
        self._prev_escalating: Optional[bool] = None

    def step(self) -> Dict:
        cfg = self.cfg
        t = self.t + 1

        pre_bob = self.bob.snapshot() if cfg.keep_snapshots else None
        escalating = self.bob.begin_round()
        if cfg.keep_snapshots and (self._prev_escalating is None
                                   or escalating != self._prev_escalating):
            self.phase_starts.append((t, pre_bob))
        self._prev_escalating = escalating
        p_escalate = self.bob.p_escalate()

        agent_starts = bool(self.env_rng.integers(0, 2))

        arm = self.algorithm.select_arm()
        result = self.env.run(
            self.arms[arm], self.bob.get_strategy(),
            agent_starts=agent_starts, rng=self.env_rng,
        )
        reward = DollarAuction.normalize_reward(
            result.agent_payoff, cfg.budget, cfg.stake)
        self.algorithm.update(
            arm, reward, info={"agent_state_trace": result.agent_state_trace})

        if cfg.bob_signal == "theta":
            signal = self.thetas[arm] / cfg.budget
        else:
            signal = result.agent_final_bid / cfg.budget
        self.bob.observe_agent_move(signal)

        detected = False
        if self.detector is not None:
            detected = self.detector.update(reward, global_round=t)
            if detected:
                self.algorithm.reset()

        self.t = t
        self.cumulative_reward += reward
        winning_bid = (result.agent_final_bid if result.winner == "agent"
                       else result.opponent_final_bid)
        rec = {
            "round": t,
            "arm": arm,
            "theta": self.thetas[arm],
            "bob_escalating": escalating,
            "bob_regime": "Escalating" if escalating else "Rational",
            "p_escalate": p_escalate,
            "agent_starts": agent_starts,
            "winner": result.winner,
            "agent_final_bid": result.agent_final_bid,
            "opponent_final_bid": result.opponent_final_bid,
            "winning_bid": winning_bid,
            "reward": reward,
            "bob_reward": DollarAuction.normalize_reward(
                result.opponent_payoff, cfg.budget, cfg.stake),
            "cumulative_reward": self.cumulative_reward,
            "ph_detected": detected,
        }
        self.records.append(rec)
        return rec

    def run(self, T: Optional[int] = None, progress: bool = False) -> List[Dict]:
        n = T if T is not None else self.cfg.T
        it = range(n)
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(it)
            except ImportError:
                pass
        for _ in it:
            self.step()
        return self.records

    def regime_log(self) -> List[Tuple[int, bool]]:
        """realized_regime_log: [(t, escalating), ...] (design sec. 15)."""
        return [(r["round"], r["bob_escalating"]) for r in self.records]


def make_simulation(
    cfg: SimConfig,
    algorithm: Union[str, BanditAlgorithm] = "elp",
    use_ph: bool = False,
) -> AuctionSimulation:
    """
    Build a ready-to-run simulation.

    algorithm : "elp" | "exp3" | a BanditAlgorithm instance you built
        yourself (e.g. EXP3.S; then YOU are responsible for giving it its
        own rng, ideally spawned from cfg.seed). Strings use the learner
        stream spawned below.
    use_ph : attach a PageHinkleyDetector (needs an algorithm with reset(),
        i.e. not ELP until ELP.reset is implemented).
    """
    ss = np.random.SeedSequence(cfg.seed)
    algo_ss, bob_ss, env_ss = ss.spawn(3)
    algo_rng = np.random.default_rng(algo_ss)
    bob_rng = np.random.default_rng(bob_ss)
    env_rng = np.random.default_rng(env_ss)

    thetas = list(range(0, cfg.budget + 1))
    arms = build_threshold_arm_set(cfg.budget, cfg.stake, thetas=thetas)

    if isinstance(algorithm, str):
        key = algorithm.lower()
        if key == "elp":
            from src.algorithms.elp import ELP
            algo = ELP(strategies=arms, thetas=thetas, stake=cfg.stake,
                       budget=cfg.budget, horizon_T=cfg.T, rng=algo_rng)
        elif key == "exp3":
            from src.algorithms.exp3 import EXP3
            algo = EXP3(n_arms=len(arms), horizon_T=cfg.T, rng=algo_rng)
        else:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Use 'elp', 'exp3', or pass "
                "an instance (e.g. EXP3.S once exp3s.py exists)."
            )
    else:
        algo = algorithm

    bob = Bob(
        stake=cfg.stake, budget=cfg.budget, mu=cfg.bob_mu,
        sunk_cost_threshold=cfg.bob_sunk_cost_threshold,
        prior_sunk_cost=cfg.bob_prior_sunk_cost,
        escalation_rate=cfg.bob_escalation_rate,
        window=cfg.bob_window, escalation_gain=cfg.bob_gain,
        min_dwell=cfg.bob_min_dwell, min_history=cfg.bob_min_history,
        fixed_regime=cfg.bob_fixed_regime,
        escalation_ceiling=cfg.bob_escalation_ceiling, rng=bob_rng,
    )
    detector = None
    if use_ph:
        detector = PageHinkleyDetector(
            delta=cfg.ph_delta, threshold=cfg.ph_threshold,
            mean_alpha=cfg.ph_alpha, warmup=cfg.ph_warmup,
            direction=cfg.ph_direction,
        )
    env = DollarAuction(stake=cfg.stake, budget=cfg.budget)
    return AuctionSimulation(cfg, algo, thetas, arms, bob, env, env_rng, detector)


# ----------------------------------------------------------------------
# Mesa layer (orchestration + DataCollector only)
# ----------------------------------------------------------------------
# NOTE: written against Mesa 3.0+ API but NOT executed in the sandbox this
# was authored in (Mesa unavailable there). The core above is tested; if the
# wrapper raises, the likely culprits are listed in the README of this change.

_MESA_IMPORT_ERROR = None
try:
    from mesa import Agent as _MesaAgent, Model as _MesaModel
    from mesa.datacollection import DataCollector as _DataCollector
    _HAS_MESA = True
except ImportError as _err:  # pragma: no cover
    _HAS_MESA = False
    _MESA_IMPORT_ERROR = _err   # keep the REAL reason (not installed vs. broken)

if _HAS_MESA:

    class LearnerAgent(_MesaAgent):
        def __init__(self, model):
            super().__init__(model)          # Mesa 3: no unique_id argument
            self.label = "Learner"
            self.chosen_theta = None
            self.reward = None

    class BobAgent(_MesaAgent):
        def __init__(self, model):
            super().__init__(model)
            self.label = "Bob"
            self.chosen_theta = None         # Bob has no arm
            self.reward = None

    class AuctionModel(_MesaModel):
        """One Mesa step == one auction round (one contention event)."""

        def __init__(self, sim: AuctionSimulation):
            super().__init__()
            self.sim = sim
            self.last: Dict = {}
            self.learner = LearnerAgent(self)
            self.bob_agent = BobAgent(self)
            self.datacollector = _DataCollector(
                model_reporters={
                    "Round": lambda m: m.last.get("round"),
                    "Bob_Regime": lambda m: m.last.get("bob_regime"),
                    "P_Escalate": lambda m: m.last.get("p_escalate"),
                    "Winning_Bid": lambda m: m.last.get("winning_bid"),
                    "Cumulative_Reward": lambda m: m.last.get("cumulative_reward"),
                    "PH_Detected": lambda m: m.last.get("ph_detected"),
                },
                agent_reporters={
                    "Agent_ID": "label",
                    "Chosen_Arm": "chosen_theta",
                    "Reward": "reward",
                },
            )

        def step(self):
            rec = self.sim.step()
            self.last = rec
            self.learner.chosen_theta = rec["theta"]
            self.learner.reward = rec["reward"]
            self.bob_agent.reward = rec["bob_reward"]
            self.datacollector.collect(self)


def run_with_mesa(
    cfg: SimConfig,
    algorithm: Union[str, BanditAlgorithm] = "elp",
    use_ph: bool = False,
    progress: bool = False,
):
    """Run via Mesa and return (model, model_df, agent_df)."""
    if not _HAS_MESA:
        raise ImportError(
            "Mesa could not be imported (see the chained error below for the "
            "real cause). Check: `pip show mesa` inside the active .venv; if "
            "it is missing run `pip install \"mesa>=3.0\"`; if the install or "
            "import fails, try a venv built with Python 3.12/3.13."
        ) from _MESA_IMPORT_ERROR
    sim = make_simulation(cfg, algorithm=algorithm, use_ph=use_ph)
    model = AuctionModel(sim)
    it = range(cfg.T)
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it)
        except ImportError:
            pass
    for _ in it:
        model.step()
    return (model,
            model.datacollector.get_model_vars_dataframe(),
            model.datacollector.get_agent_vars_dataframe())