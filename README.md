# Repeated Dollar Auction — Multi-Armed Bandits vs. an Adaptive Adversary

Bachelor thesis (Computer Science). Extends **Waniek, Tran-Thanh & Michalak (2016), "Repeated Dollar Auctions: A Multi-Armed Bandit Approach"** (AAMAS 2016) from a static to a **non-stationary, adaptive** setting, and maps it onto **computational resource contention (cloud spot-instance bidding)**.

> Rationale for every non-trivial choice lives in `docs/design_decisions.md` (sec. 13–16 cover the adaptive-adversary pivot). Notation: `docs/notation.md`.

---

## 1. Research summary

| | |
|---|---|
| **Base theory** | Waniek et al. (2016): the repeated dollar auction as an adversarial MAB; ELP with prefix side-information (Lemma 1). |
| **What we add** | (1) an **adaptive adversary** (Bob) whose regime depends on the learner's recent play; (2) **switching/dynamic regret** as the benchmark; (3) **EXP3.S + Page-Hinkley** reset as change-responsive learner; (4) a **computational-resource-contention** domain (EC2 spot bidding); (5) an **N-player** extension. |
| **Arm set `S0`** | The *threshold strategy family* (Waniek Sec. 5.2): `f_θ` follows O'Neill's optimal strategy until the opponent's bid reaches θ, then passes. One arm per θ ∈ {0,…,b}. |
| **Reward** | Continuous, normalized to [0,1] (`DollarAuction.normalize_reward`), never binary win/lose. |

### Theoretical scope (read before writing claims)
- **Proven in the paper:** ELP achieves Õ(√T) **static** regret on the threshold family (Thm 6 / Cor 7), because playing a high-θ arm reveals the outcome of all lower-θ arms (prefix property, Lemma 1).
- **Not proven (our empirical contribution):** any guarantee for EXP3, EXP3.S, or EXP3.S+Page-Hinkley against Bob, or for **switching** regret. Present these as hypotheses tested by simulation.
- **Adaptive Bob ⇒ policy-regret flavour.** The learner's own play changes Bob's future behaviour, so a counterfactual "arm g all along" must be **re-simulated** from a Bob snapshot (sec. 16). Bob's rolling window gives him bounded memory, which is what keeps this meaningful.
- **To verify before citing:** Thm 6 claims α(G)=1 for the (directed) side-information graph; check that the cited ELP theorem covers directed graphs.
- ELP's Lemma-1 replay stays valid *within a round* (Bob's regime is fixed before the auction starts) but not across rounds (sec. 14).

---

## 2. Roadmap (three phases)

| Phase | Setting | Environment | Status |
|---|---|---|---|
| **Phase 1** | 1v1 synthetic: Learner (MAB) vs. Bob (adaptive adversary) | abstract Dollar Auction (`dollar_auction.py`) | in progress |
| **Phase 2** | 1v1 EC2 spot simulator | Computational Resource Contention model based on Khandelwal et al. (ICCUBEA 2018) | planned |
| **Phase 3** | N-player MAS: N agents, each running a MAB algorithm, compete in a shared compute pool | N-player contention engine | planned |

Experiment folders map onto phases:

| Phase | Folder | Purpose |
|---|---|---|
| 1 | `experiments/stage0_baseline_replication/` | Validate our ELP against the paper's setting (rational, oblivious opponent; static regret). **Standalone script, frozen.** |
| 1 | `experiments/stage1_adaptive_1v1/` | ELP vs EXP3 vs EXP3.S vs EXP3.S+PH against the **adaptive Bob**, 1v1. Bob scenarios (slow / fast reaction: window `W`, `min_dwell`, gain) are *configurations of one experiment*, not separate stages. |
| 2 | `experiments/stage2_ec2_spot/` | 1v1 on the spot-market simulator. |
| 3 | `experiments/stage3_nplayer/` | N-player contention. |
| — | `experiments/stage_optional_swoopo/` | *Optional / undecided:* Swoopo penny-auction validation (dropped from the core roadmap pending advisor decision). |

> **Numbering changed.** Old README: Stage 2 = N-player, Stage 3 = Swoopo. Now: Phase 2 = EC2 1v1, Phase 3 = N-player. `design_decisions.md` sec. 7, 11, 17, 19–20 still use the old numbering and need updating.

**Stage 1a/1b were merged into one stage.** They only made sense for the old scripted Bob (single vs. recurring *fixed* switch times). With the adaptive Bob, escalation times are an outcome of play (sec. 15), so "rare vs. frequent switching" is just a Bob configuration (e.g. large `W` + large `min_dwell` vs. small `W` + small `min_dwell`). Run both as scenarios inside `stage1_adaptive_1v1` and report the realized switch-time distribution across seeds. `design_decisions.md` sec. 6 and 15 still mention 1a/1b and need updating.

---

## 3. Agents (terminology — important)

- **Learner ("Alice" in the thesis text)** — rational MAB agent; chooses a threshold arm θ each round with ELP / EXP3 / EXP3.S / EXP3.S+PH. In code this is the *learner/agent*.
- **Bob** — **adaptive adversary**, not a learner. Fixed, hand-designed reaction rule that is **history-dependent**: a rolling window (size `W`) of the learner's aggressiveness (e.g. θ/b) → `p_escalate` → per-round regime draw (Rational / Escalating). While escalating, within-auction sunk-cost dynamics apply (probabilistic continuation past a sunk-cost threshold, up to a ceiling).
- **Rational opponent** (`alice_rational.py`, class `Alice`) — the oblivious fixed-rule opponent used only in Stage 0. ⚠ *Name clash with the thesis's "Alice".* Recommended: rename file/class to `rational_opponent.py` / `RationalOpponent`.

Why "adaptive" matters: the reward may depend on the learner's *previous-round* choices, which is exactly what justifies adversarial-bandit methods over non-stationary *stochastic* ones (sec. 1). Within-auction reactivity does not count (sec. 13).

---

## 4. Domain mapping: Dollar Auction → EC2 spot instances

Source: Khandelwal, Chaturvedi & Gupta, *Bidding Strategies for Amazon EC2 Spot Instances — A Comprehensive Review* (ICCUBEA 2018).

| Dollar Auction | Compute / spot-instance meaning |
|---|---|
| Stake `s` | Value of completing the job on contested spot capacity |
| Bid `x` | Bid price / compute committed for the spot instance |
| Threshold `θ` | **Bid cap** — the highest price the user will chase |
| Budget `b` | Maximum affordable bid (e.g. on-demand price or above) |
| All-pay / sunk cost | **Out-of-bid termination**: price rises above the bid, the instance is revoked, un-checkpointed work is lost and must be redone |
| Bob's escalation | A user protecting accumulated progress by raising bids (sunk-cost fallacy) |

The review's static bidding strategies (Bid Min, Mean, On-Demand 30 %, 70 %, On-Demand, Max) are natural candidates for the Phase-2 θ grid, and its job-size / price-volatility suitability table (Table 1) can parametrize scenarios.

**Modelling caveats (state them in the thesis):**
1. In real EC2 the user pays the *market* price, not their bid, and a partial hour ended by the provider is not billed. The all-pay property therefore applies to **lost work**, not to bid payment. Phase 2 must define the cost model explicitly.
2. AWS replaced bidding with simplified pricing in 2018 (ref. [23] of the review). Phase 2 models the legacy market stylistically.
3. The ICCUBEA paper is a literature review (it summarizes strategies proposed in other works; its only own empirical part is an analysis of EC2 price traces whose data/method are not released). Use it for qualitative structure and parametrization, not as a dataset.
4. **Price traces cannot validate auction dynamics.** A trace contains the exogenous market price, not competitors' bids, so a dollar-auction/escalation process cannot be replayed or validated from it. Phase 2 is therefore a **model-based simulator**; traces are optional, only to make the price process realistic (e.g. trace-driven background price).
5. Ben-Yehuda et al. (the work cited as [22] in the review) report that legacy EC2 spot prices were usually *not* market-driven but drawn at random from a tight interval via a dynamic hidden reserve price (their CloudCom 2011 paper; confirm the claim in the journal version before citing). So the legacy market was not a genuine N-bidder auction either: our all-pay/escalation mapping is a **stylized scenario**, and the thesis should say so.

**Public data that exists (optional use):** Calvin Ardi's *Amazon EC2 Spot Price History* (2014–2015 and 2017–2023; https://ant.isi.edu/~calvin/data/ec2-spot-price/, also on Zenodo). Only the 2014–2015 part is clearly from the bidding era; later years follow the 2018 pricing change.

---

## 5. Project structure

Legend: ✅ done · 🔧 done, but needs check · ⬜ not yet written

```
dollar-auction-bandit/
├── README.md
├── requirements.txt
├── pytest.ini                         # ✅ pythonpath=. so tests can `import src...`
├── .gitignore                         # must contain .venv/
│
├── src/
│   ├── environment/
│   │   ├── dollar_auction.py          # ✅ 2-player all-pay auction (Xb formalization)
│   │   ├── strategies.py              # ✅ O'Neill + threshold strategies (arm set S0)
│   │   ├── dollar_auction_nplayer.py  # ⬜ Phase 3
│   │   └── spot_market.py             # ⬜ Phase 2 (resource-contention simulator)
│   │
│   ├── opponents/
│   │   ├── alice_rational.py          # ✅ oblivious rational opponent (Stage 0)
│   │   ├── bob_sunkcost.py            # 🔧 adaptive adversary (rolling window, snapshot-safe)
│   │   └── population.py              # ⬜ Phase 3 heterogeneous population
│   │
│   ├── algorithms/
│   │   ├── base.py                    # ✅ interface: select_arm / update(info) / reset
│   │   ├── elp.py                     # ✅ ELP (no reset() yet -> cannot be paired with PH)
│   │   ├── exp3.py                    # ✅ EXP3
│   │   ├── exp3s.py                   # ⬜ EXP3.S
│   │   └── page_hinkley.py            # 🔧 direction fixed (detects reward DROP), warm-up added
│   │
│   ├── metrics/                       # NOTE: regret.py / detection_delay.py do NOT exist yet
│   │   ├── plots.py                   # ✅ shared standard plot set (Stage 0 = baseline; later stages add layers)
│   │   ├── regret.py                  # ⬜ per-phase switching regret (re-simulation from Bob snapshots)
│   │   └── detection_delay.py         # ⬜ PH alarm round vs. realized Bob regime change
│   │
│   └── simulation/
│       ├── config.py                  # 🔧 SimConfig dataclass: every experiment parameter in one place
│       └── runner.py                  # 🔧 AuctionSimulation (pure core) + thin Mesa wrapper
│
├── experiments/
│   ├── stage0_baseline_replication/run_stage0.py   # ✅ standalone (no runner/Mesa); tqdm + shared plots + Thm-6 bound; `--quick` for a smoke run
│   ├── calibrate_regimes.py           # 🔧 PRE-FLIGHT tool for Stage 1 (see below)
│   ├── smoke_mesa.py                  # 🔧 end-to-end check of the Mesa wrapper
│   ├── stage1_adaptive_1v1/           # ⬜ (merged 1a+1b; Bob scenarios as configs)
│   ├── stage2_ec2_spot/               # ⬜
│   └── stage3_nplayer/                # ⬜
│
├── tests/
│   ├── test_environment.py            # ✅ 5 tests
│   ├── test_opponents.py              # 🔧 6 tests (rewritten for the adaptive Bob)
│   ├── test_page_hinkley.py           # 🔧 4 tests
│   ├── test_bob_sunkcost.py           # 🔧 8 tests
│   ├── test_runner.py                 # 🔧 10 tests
│   └── test_plots.py                  # 🔧 3 tests
│
├── data/            # raw (read-only) / processed / synthetic
├── results/         # per-stage figures and tables
├── notebooks/       # exploration only
└── docs/            # design_decisions.md, notation.md
```

### What the less obvious files are for
- **`src/simulation/config.py`** — one `SimConfig` dataclass (budget, stake, T, seed, Bob parameters, Page-Hinkley parameters). The runner reads *only* from it, so an experiment is fully described by one object and nothing is hard-coded across files. Stage 1+ scripts build a `SimConfig`; Stage 0 predates it and keeps its own `CONFIGS` list.
- **`src/simulation/runner.py`** — `AuctionSimulation` holds all simulation logic and randomness (testable without Mesa). `AuctionModel` (a `mesa.Model`) only calls `sim.step()` and lets `DataCollector` log. It also stores `phase_starts` (Bob snapshots) for hindsight regret. It does **not** compute regret.
- **`experiments/calibrate_regimes.py`** — run it **before** any Stage 1 experiment. It plays every fixed θ against an always-rational Bob and an always-escalating Bob, and reports whether the best arm *differs* between the two regimes. If it does not, there is nothing for EXP3.S/PH to track: retune Bob, budget or stake. Page-Hinkley threshold calibration (`calibrate_threshold`) belongs in the same pre-flight step.
- **`experiments/smoke_mesa.py`** — a 200-round run through the Mesa path; fails loudly if the Mesa API usage is wrong.

---

## 6. Conventions

### Algorithm interface (`src/algorithms/base.py`)
```python
class BanditAlgorithm:
    def select_arm(self) -> int: ...
    def update(self, arm: int, reward: float, info: dict | None = None) -> None: ...
    def reset(self) -> None: ...      # required for use with Page-Hinkley
```
`info={"agent_state_trace": ...}` is read only by ELP (Lemma-1 replay); EXP3-family ignores it. The runner always passes it.

### Per-round order (do not reorder — it keeps Bob's regime a function of history up to t−1)
```
snapshot Bob → bob.begin_round() → select_arm → run auction → update(arm, reward, info)
→ bob.observe_agent_move(signal) → detector.update(reward) → (alarm ⇒ algorithm.reset())
```

### Hindsight / counterfactual rule
Any hindsight computation against Bob **must** use a deep-copied snapshot (`bob.snapshot()`), never the live Bob. Do not cache `bob.get_strategy()` closures on an instance (deepcopy would leave them pointing at the original). Covered by `tests/test_bob_sunkcost.py` and `tests/test_runner.py`.

### Randomness & reproducibility
One `seed` per run → `SeedSequence.spawn(3)` → three **independent** generators: learner, Bob, environment (starting player). Bob's RNG is private; hindsight code never touches live generators. Report means ± CI over many seeds, never single runs.

### Logged data (Mesa `DataCollector`)
Model level: `Round`, `Bob_Regime`, `P_Escalate`, `Winning_Bid`, `Cumulative_Reward`, `PH_Detected`.
Agent level: `Agent_ID`, `Chosen_Arm`, `Reward`.
`Cumulative_Regret` is **computed post-hoc** by `src/metrics/regret.py` from the records and Bob snapshots (it needs re-simulation, so it is not a per-step column).

---

## 7. Environment setup & running

```bash
python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install "mesa>=3.0" networkx numpy pandas matplotlib seaborn scipy tqdm pytest
```
`requirements.txt`: `mesa>=3.0, networkx (Mesa 3.5 imports it but does not declare it), numpy, pandas, matplotlib, seaborn, scipy, tqdm, pytest, jupyter`. **SMPyBandits** is optional (baseline UCB only, later); it is old and may conflict with Mesa 3 / recent NumPy, so install it last and drop it if it does.

```bash
pytest -v                                                      # expect 36 passed
python -m experiments.smoke_mesa                               # Mesa wrapper check
python -m experiments.calibrate_regimes                        # before Stage 1
python -m experiments.stage0_baseline_replication.run_stage0   # Stage 0 replication
```

### Tooling policy
- **Custom engine** is the core (auction, bandits, regret, detection).
- **Mesa 3.0+** only for orchestration and `DataCollector` — no spatial grid. Agents are `Agent(model)` (no `unique_id`), no `mesa.time` schedulers.
- **SMPyBandits** only for single-player baseline algorithms; its multi-player algorithms target collision problems and do not fit.
- **tqdm** for progress bars in experiment drivers.

---

## 8. Open decisions

| # | Decision | Where it matters |
|---|---|---|
| 1 | Bob `min_dwell` (1 = per-round redraw as in sec. 13; >1 = persistent phases) | Per-phase hindsight needs persistent phases |
| 2 | Aggressiveness signal: `θ/b` (sec. 13) vs. observable `final_bid/b` | Realism of Bob's information |
| 3 | Bob/budget/stake tuning so the best arm differs between regimes | Stage 1 is vacuous otherwise (current defaults barely separate them) |
| 4 | Phase-2 cost model (what is "paid" when out-bid) | Validity of the all-pay mapping |
| 5 | N-player payment rule and turn order (sec. 7) | Phase 3 |
| 6 | Keep or drop Swoopo validation | Scope |
| 7 | Asymmetric-cost regret (sec. 18) | Optional extension |

---

## 9. Collaboration

- Branches per feature/stage (`feature/exp3s`, `experiment/stage1a`); PR + review before merging to `main`; descriptive commits referencing the phase/stage.
- Keep `docs/design_decisions.md` current — log every decision when it is made.

| Member | Responsibility |
|---|---|
| **A** | |
| **B** | |
| **Both** | |

---

## 10. Progress

**Phase 1**
- [x] Stage 0 — ELP baseline replication vs. rational opponent
- [x] Page-Hinkley direction fix (+ warm-up)
- [x] Bob rewritten as adaptive adversary (rolling window, snapshot-safe)
- [x] `runner.py` / `config.py` (core tested; Mesa wrapper pending `smoke_mesa`)
- [ ] `exp3s.py`, `ELP.reset()`
- [ ] `regret.py` (per-phase hindsight), `detection_delay.py`
- [ ] Regime calibration (`calibrate_regimes.py`) → choose Stage 1 scenario parameters
- [ ] Stage 1 (adaptive Bob, 1v1; slow/fast Bob scenarios)

**Phase 2** 
- [ ] spot-market simulator 
- [ ] Stage 2 experiments

**Phase 3** 
- [ ] N-player engine 
- [ ] population generator 
- [ ] Stage 3 experiments

---

## 11. Key references

1. Waniek, M., Tran-Thanh, L., & Michalak, T. (2016). *Repeated Dollar Auctions: A Multi-Armed Bandit Approach.* AAMAS 2016.
2. Khandelwal, V., Chaturvedi, A. K., & Gupta, C. P. (2018). *Bidding Strategies for Amazon EC2 Spot Instances — A Comprehensive Review.* ICCUBEA 2018.
3. Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002). *The Nonstochastic Multiarmed Bandit Problem.* SIAM J. Computing 32(1). (EXP3, EXP3.S)
4. Mannor, S., & Shamir, O. (2011). *From Bandits to Experts: On the Value of Side-Observations.* NeurIPS. (ELP)
5. Poland, J. (2005). *FPL analysis for adaptive bandits.* SAGA'05. (extends the regret argument to adaptive adversaries; cited in Waniek footnote 2)
6. O'Neill, B. (1986). *International Escalation and the Dollar Auction.* J. Conflict Resolution 30(1).
7. Besbes, O., Gur, Y., & Zeevi, A. (2014). *Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards.* NeurIPS.
8. Garivier, A., & Moulines, E. (2011). *On Upper-Confidence Bound Policies for Switching Bandit Problems.* ALT.
9. Guo, W., Chen, K., Wu, Y., & Zheng, W. (2015). *Bidding for Highly Available Services with Low Price in Spot Instance Market.* HPDC '15.
10. Zaman, S., & Grosu, D. (2013). *Combinatorial auction-based allocation of virtual machine instances in clouds.* J. Parallel and Distributed Computing 73(4).
11. Karunakaran, S., & Sundarraj, R. P. (2015). *Bidding Strategies for Spot Instances in Cloud Computing Markets.* IEEE Internet Computing 19(3).
12. Kushwaha, V., & Simmhan, Y. (2014). *Cloudy with a Spot of Opportunity: Analysis of Spot-Priced VMs for Practical Job Scheduling.* CCEM 2014.
13. *(Optional, Swoopo)* Byers, J., Mitzenmacher, M., & Zervas, G. (2010). *Information Asymmetries in Pay-Per-Bid Auctions: How Swoopo Makes Bank.* · Augenblick, N. (2015). *The Sunk-Cost Fallacy in Penny Auctions.*