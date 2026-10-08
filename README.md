# Repeated All-Pay (Dollar) Auctions with a Global Budget — Bandits with Knapsacks vs. Reactive Opponents

Bachelor thesis (Computer Science). Starts from **Waniek, Tran-Thanh & Michalak (2016), "Repeated Dollar Auctions: A Multi-Armed Bandit Approach"** (AAMAS 2016) and asks what changes when (i) the bidder has a **global budget across rounds** (→ *Bandits with Knapsacks*, BwK) and (ii) the opponent **reacts to the bidder's own past choices** (→ non-stationarity that is *endogenous*). The application setting is **computational resource contention** (EC2-style spot bidding).

> **Status.** Direction changed to BwK. Everything marked **PROPOSED** is a design that is *not approved and not implemented yet*. Rationale for earlier decisions: `docs/design_decisions.md` (sec. 13–16: adaptive Bob, snapshots, re-simulation). Notation: `docs/notation.md`.

---

## 1. Research summary

**Research questions**
- **RQ1.** In a repeated all-pay auction with a global budget and a changing environment, how do BwK algorithms (stationary, non-stationary, adversarial) compare with bandit algorithms that are *not* budget-aware (EXP3, ELP, UCB1)?
- **RQ2.** Against an opponent that reacts to the agent's own policy, do the theoretical non-stationarity measures of non-stationary BwK (local `V`, global `W`) remain meaningful when the non-stationarity is *caused by the agent*?

| | |
|---|---|
| **Base** | Waniek et al. 2016: ELP with prefix side-information (Lemma 1), threshold strategies, deterministic opponent, static regret, **budget reset every auction**. |
| **Known, therefore not our contribution** | *Non-stationary BwK* (Liu, Jiang & Li 2022): dynamic benchmark, measures `V1,V2,W1,W2`, Sliding-Window UCB-BwK. |
| **Our angle** | all-pay auction structure + **reactive opponent** + empirical study in a compute-contention simulator. Result type: **empirical**, not new theorems. |
| **Arm set `S0`** | Unchanged: threshold strategies `f_θ`, θ ∈ {0,…,b} (Waniek Sec. 5.2). **θ = 0 is the null arm** (never bids). |
| **Budget protocol** (PROPOSED) | `B = ρ·T`, one resource (`d = 1`, money). Play stops at the first round whose cost would exceed the remaining budget; that round earns nothing (Liu et al.). |
| **Reward / cost** (PROPOSED) | `reward = 1{win}` (stake won, normalized), `cost = final_bid / b` (all-pay: paid even when losing). Stage 3 adds a VM cost when winning. Stages 0–1 keep the old net-payoff reward ("reset mode"). |

### Theoretical scope
- **Proven in Waniek:** ELP achieves Õ(√T) static regret on the threshold family (Thm 6) under an oblivious opponent and per-auction budgets.
- **Proven in Liu et al. (checked against the PDF):** Reg ≤ Õ((1/b)√(mT) + m^{1/3}V1^{1/3}T^{2/3} + (1/b)m^{1/3}d·V2^{1/3}T^{2/3} + W1 + q̄W2), `q̄ ≤ 1/b`, with matching lower bounds. Their model treats the distribution sequence `P_t` as fixed in advance (the benchmark is defined with all `P_t` known; `V, W` are computed from fixed sequences). They never state "oblivious" for BwK itself; their OCOwC appendix explicitly allows adaptive deterministic constraints, a different problem.
- **Not covered by any theorem we have:** BwK algorithms against a *reactive* Bob. If `P_t` depends on the agent's past play, `V` and `W` become policy-dependent. Whether the bound still holds pathwise against the realized sequence is **our conjecture, unverified**.
- **Unverified items:** Thm 6's claim α(G)=1 for the *directed* prefix graph (Alon et al. results for directed graphs should be checked); novelty of the combination (all-pay + BwK + reactive opponent + policy regret); standard Hedge/Exp3/Exp3.S bounds (not in the attached papers).

---

## 2. BwK in a nutshell
Each round the learner plays an arm and gets a **reward** *and* consumes **resources** from a total budget `B`; the game ends when the budget runs out or after `T` rounds; an arm with reward 0 and cost 0 (**null arm**) always exists. Optimal behaviour is usually a **mixture of arms**, found by a linear program:

```
maximize   Σ_i  reward_i · x_i          x = distribution over arms (incl. null arm)
subject to Σ_i  cost_i   · x_i ≤ ρ = B/T
```

With `d = 1` this means: play the arm with the best reward/cost ratio as often as the budget allows, fill the rest with the null arm. Sliding-Window UCB-BwK replaces the unknown means by windowed optimistic (reward) / pessimistic (cost) estimates and solves this LP every round. The budget couples all rounds, which is why *local* change measures are not enough and Liu et al. introduce the *global* `W`.

---

## 3. Roadmap

| Stage | Setting | Budget | Opponent | Status |
|---|---|---|---|---|
| **0** | Replicate Waniek: ELP, static regret | reset per auction | Alice (deterministic) | ✅ done |
| **1** | Adaptive Bob, no global budget (policy-regret findings) | reset per auction | Bob | 🔧 infrastructure + sweeps done, experiment not run |
| **2a** | BwK validation: do BwK algorithms approach the stationary LP benchmark? | **global** | Alice | PROPOSED |
| **2b** | BwK vs. reactive Bob; policy regret; endogenous `V̂, Ŵ` | **global** | Bob (+ `w=0` control) | PROPOSED |
| **3** | EC2 layer: AR(1) background price with epochs (exogenous non-stationarity), VM cost | **global** | Alice, Bob | PROPOSED |
| future | N-player population | — | — | out of scope |
| optional | Swoopo (penny auction) as empirical evidence that escalation exists | — | — | undecided |

Stage 2a exists to separate *bugs* from *phenomena*: without a stationary sanity check you cannot interpret results against Bob.

---

## 4. Algorithms (PROPOSED set)

| Family | Algorithm | Source | Role |
|---|---|---|---|
| BwK, stationary | **UCB-BwK** | Agrawal & Devanur 2014 | optimistic reward, pessimistic cost, LP each round |
| BwK, non-stationary | **SW-UCB-BwK** | Liu et al. 2022, Alg. 1 | windows `w1` (reward), `w2` (cost); tuned by grid since `V` is unknown/endogenous |
| BwK, adversarial | **LagrangeBwK** | Immorlica et al. 2019 | black-box reduction to an adversarial bandit algorithm; needs an estimate of OPT |
| non-BwK baseline | **EXP3** | Auer et al. 2002 | adversarial, no side-information |
| non-BwK baseline | **ELP** | Waniek 2016 / Mannor–Shamir 2011 | adversarial + prefix side-information (the Waniek link) |
| non-BwK baseline | **UCB1** | classical | stochastic assumption |

- Non-BwK baselines run under the **same stopping rule** ("play until the budget is exhausted"); one pacing variant (`cost ≤ B/T` per round) may be added for the best of them. All algorithms get equal hyper-parameter tuning effort.
- **Legacy, kept but not in the main experiments:** EXP3.S, Page-Hinkley (`page_hinkley.py`, tested), `ELP` without `reset()`. They stay in the repo so Stages 0–1 remain reproducible.
- Optional if time allows: PrimalDualBwK (Badanidiyuru et al. 2013), EXP3-based BwK for `d=1` (Rangi et al. 2018).
- SMPyBandits provides classical single-player bandits only; **we found no BwK policy there** (not confirmed by inspection). Write BwK ourselves (`scipy.optimize.linprog`, ~100 lines each).

---

## 5. Evaluation design (PROPOSED)

| Situation | Benchmark | Valid because |
|---|---|---|
| Exogenous epochs (Stage 3, Alice) | **Dynamic LP** of Liu et al. with the generator's true per-epoch means | `P_t` fixed in advance |
| Reactive Bob (Stage 2b) | **Policy regret vs. best fixed distribution**, re-simulated from t = 1 against a deep copy of Bob under the same budget/stop rule. With `d=1` candidates are "one arm × fraction of rounds played" (≈130) | the comparator's own play changes Bob, so it must be re-simulated |
| Reactive Bob, diagnostics | (a) best fixed arm, (b) regime-aware oracle, (c) **realized `V̂, Ŵ`**: freeze Bob at its actual state each round, estimate each arm's mean reward/cost by Monte-Carlo (a *within-round* counterfactual), then compute `V, W` along the realized path | does not use the forbidden multi-round counterfactual |

Metrics: total reward until budget exhaustion, ratio to benchmark, budget utilisation, escalation frequency, external vs. policy regret, **adaptivity index** (their gap), and whether `V̂, Ŵ` differ across algorithms on the *same* instance (direct evidence of endogeneity). Controls: Bob with no memory (see Open decisions #7). Report 30–50 seeds, explicit seeds, mean ± CI; report negative results as they are.

**Never** use per-round counterfactual rewards of *other arms* to score a whole trajectory against Bob.

---

## 6. Agents (terminology)
- **Learner** — *our* agent, the only one that learns; chooses an arm θ each round.
- **Alice** (`alice_rational.py`) — an **opponent**: deterministic and oblivious (raises while the next bid ≤ `mu·stake`, then folds).
- **Bob** (`bob_sunkcost.py`) — **adaptive opponent**: regime (Rational/Escalating) drawn each round from `p_escalate = clip(gain · mean(window of the learner's aggressiveness))`; while escalating, sunk-cost dynamics apply. Parameters: `window`, `min_history`, `min_dwell`, `gain`, `prior_sunk_cost`, `escalation_rate`, ceiling, `fixed_regime` (calibration only). Snapshot-safe.

---

## 7. Domain mapping (EC2-style contention) and honest caveats
| Auction | Compute setting |
|---|---|
| stake | value of completing the job on contested capacity |
| bid / θ | bid commitment / bid cap |
| global budget `B` | total spending allowance across jobs |
| all-pay cost | out-of-bid termination: un-checkpointed progress is lost |
| Bob's `prior_sunk_cost` | progress already invested, which the user tries to protect |

1. **All-pay is our modelling assumption.** Real EC2 charged the market price, not the bid, and did not bill the interrupted partial hour.
2. Ben-Yehuda et al. (2013): ~98 % of the time legacy spot prices came from a hidden-reserve-price AR(1) process, not from competing bids; prices say little about real bids. Hence **competition is simulated, not read from data**.
3. **Price generator (Stage 3):** truncated AR(1) in a band `[F, C]` (fractions of the on-demand price): `P_i = P_{i-1} + Δ_i`, `Δ_i = −0.7·Δ_{i-1} + ε(σ = 0.39·(C−F))`, epochs = changes of `(F, C)`. Cost parameters from Abundo et al. Use a *calibrated generator*, not an old trace. Real traces (Ardi's archive) are optional input for calibration only.
4. Traces contain prices, not opponents' bids, so they cannot validate escalation.

---

## 8. Literature map
| Paper | Role |
|---|---|
| Waniek et al. 2016 | base theory, ELP, threshold family |
| **Badanidiyuru et al. 2013; Slivkins book Ch. 10** | BwK definition and basics |
| Agrawal & Devanur 2014 | UCB-BwK |
| **Liu, Jiang & Li 2022** | non-stationary BwK, `V`/`W`, SW-UCB-BwK |
| Immorlica et al. 2019 | adversarial BwK, LagrangeBwK |
| Fikioris et al. (arXiv 2302.14686) | "approximately stationary" BwK: alternative non-stationarity notion; **check whether its adversary may depend on the agent's actions** (the abstract suggests dependence on previous rounds; not verified) |
| Garivier & Moulines 2008 | sliding-window UCB |
| Poland 2005; Arora, Dekel & Tewari 2012 (verify) | adaptive adversaries, policy regret |
| Khandelwal 2018 (review); Ben-Yehuda 2013; Menache et al. 2014; Abundo et al. 2014 | EC2 domain: strategy families, price generator, learning bidders |

---

'''## 9. Project structure
Legend: exists and was confirmed · 🔧 exists, not yet confirmed · ⬜ planned · *(legacy)* kept and tested, but not in the main BwK experiments

```
dollar-auction-bandit/
│
├── README.md                              # ✅ overview, roadmap, decisions (kept current)
├── requirements.txt                       # 🔧 mesa>=3.0, networkx, numpy, pandas, matplotlib, seaborn, scipy, tqdm, pytest, pyyaml
├── pytest.ini                             # ✅ pythonpath = .  (tests can `import src...`)
├── .gitignore                             # ✅ must contain .venv/, results/, data/raw/
│
├── configs/                               # ⬜ one YAML per experiment grid
│   ├── stage1_adaptive_1v1.yaml           # ⬜ Bob scenarios (window w, min_dwell, gain, escalation c)
│   ├── stage2a_bwk_alice.yaml             # ⬜ algorithm × budget ρ=B/T, seeds
│   ├── stage2b_bwk_bob.yaml               # ⬜ algorithm × Bob (w, c, w=0 control) × B
│   └── stage3_ec2_layer.yaml              # ⬜ + AR(1) band/epoch parameters, EC2 feature flag
│
├── src/
│   ├── environment/
│   │   ├── dollar_auction.py              # ✅ 2-player all-pay auction (Xb formalization); mechanics unchanged under BwK
│   │   ├── strategies.py                  # ✅ O'Neill + threshold family S0; θ = 0 is the null arm
│   │   ├── outcomes.py                    # ⬜ from an AuctionResult: gross reward 1{win}, cost = final_bid / b
│   │   ├── price_process.py               # ⬜ Stage 3: truncated AR(1) in [F, C] with epochs (Ben-Yehuda), VM cost
│   │   └── dollar_auction_nplayer.py      # ⬜ future work (N-player)
│   │
│   ├── opponents/
│   │   ├── alice_rational.py              # ✅ Alice: deterministic, oblivious opponent (Stage 0 baseline)
│   │   ├── bob_sunkcost.py                # 🔧 Bob: adaptive opponent (rolling window, snapshot-safe, prior_sunk_cost,
│   │   │                                  #    min_history, min_dwell, fixed_regime); memoryless-Bernoulli control ⬜
│   │   └── population.py                  # ⬜ future work (heterogeneous N-player generator)
│   │
│   ├── algorithms/
│   │   ├── base.py                        # ✅ interface: select_arm(), update(arm, reward, info), reset()
│   │   ├── elp.py                         # ✅ ELP with Lemma-1 prefix side-information (baseline; ELP.reset ⬜)
│   │   ├── exp3.py                        # ✅ classic EXP3 (baseline)
│   │   ├── ucb1.py                        # ⬜ classical UCB1 (stochastic baseline)
│   │   ├── budget_wrappers.py             # ⬜ optional: pacing (cost ≤ B/T), reward − λ·cost, for non-BwK baselines
│   │   ├── exp3s.py                       # ⬜ (legacy) EXP3.S switching variant
│   │   ├── page_hinkley.py                # 🔧 (legacy) change detector, reward-drop direction fixed
│   │   └── bwk/
│   │       ├── lp.py                      # ⬜ shared single-step LP helper (scipy.optimize.linprog, null arm)
│   │       ├── ucb_bwk.py                 # ⬜ UCB-BwK (Agrawal & Devanur 2014)
│   │       ├── sw_ucb_bwk.py              # ⬜ Sliding-Window UCB-BwK (Liu et al. 2022, Alg. 1; windows w1, w2)
│   │       ├── lagrange_bwk.py            # ⬜ LagrangeBwK (Immorlica et al. 2019; needs OPT estimate)
│   │       └── primal_dual_bwk.py         # ⬜ optional (Badanidiyuru et al. 2013)
│   │
│   ├── metrics/
│   │   ├── plots.py                       # ✅ shared standard figure set (RunBundle); Stage 0 = baseline, later stages add layers
│   │   ├── regret.py                      # ⬜ static regret; reward/cost until budget exhaustion; ratio to benchmark
│   │   ├── benchmarks.py                  # ⬜ dynamic LP (Liu et al.) and empirical best-fixed-distribution grid
│   │   ├── policy_regret.py               # ⬜ re-simulate whole trajectory from t=1 against a Bob snapshot
│   │   ├── nonstationarity.py             # ⬜ realized V̂, Ŵ along the path (within-round Monte-Carlo, Bob frozen)
│   │   └── detection_delay.py             # ⬜ (legacy) Page-Hinkley alarm vs. realized regime change
│   │
│   └── simulation/
│       ├── config.py                      # 🔧 SimConfig (+ stage1_config()): every experiment parameter in one place
│       ├── runner.py                      # 🔧 AuctionSimulation (pure core) + thin Mesa wrapper; records, phase_starts
│       ├── calibration.py                 # 🔧 per-arm rewards vs. a regime-pinned Bob, separation score
│       ├── budget.py                      # ⬜ budget_mode = "reset" (default) | "global"; stop rule; alive mask
│       ├── yaml_config.py                 # ⬜ YAML → SimConfig grid expansion
│       └── batch.py                       # ⬜ many seeds × algorithms in parallel (multiprocessing/joblib) with tqdm
│
├── experiments/
│   ├── stage0_baseline_replication/
│   │   └── run_stage0.py                  # ✅ ELP vs Alice, static regret, Thm-6 bound, --quick (folder name on your machine may differ)
│   ├── calibrate_regimes.py               # 🔧 pre-flight: does the best arm differ between Bob's regimes?
│   ├── sweep_bob.py                       # 🔧 pre-flight: Part A regime separation, Part B adaptive dynamics
│   ├── smoke_mesa.py                      # ✅ end-to-end check of the Mesa wrapper
│   ├── calibrate_budget.py                # ⬜ per-arm (reward, cost) → choose ρ so the budget actually binds
│   ├── stage1_adaptive_1v1/               # ⬜ adaptive Bob, per-auction budget reset (policy-regret findings)
│   ├── stage2a_bwk_alice/                 # ⬜ BwK validation: stationary, vs. LP benchmark
│   ├── stage2b_bwk_bob/                   # ⬜ BwK vs. reactive Bob: policy regret, V̂/Ŵ, adaptivity index
│   ├── stage3_ec2_layer/                  # ⬜ AR(1) price epochs + VM cost; dynamic-LP benchmark applies
│   └── stage_optional_swoopo/             # ⬜ undecided: empirical evidence that escalation exists
│
├── tests/
│   ├── test_environment.py                # ✅ auction mechanics, O'Neill, normalization
│   ├── test_opponents.py                  # ✅ Alice; Bob within-auction behaviour
│   ├── test_bob_sunkcost.py               # 🔧 regime/history, snapshot isolation, min_history, fixed_regime, prior_sunk_cost
│   ├── test_page_hinkley.py               # ✅ (legacy) direction, auto-reset
│   ├── test_runner.py                     # ✅ schema, reproducibility, independent RNG streams, phase snapshots
│   ├── test_plots.py                      # ✅ standard figure set
│   ├── test_calibration.py                # 🔧 separation score, prior_sunk_cost effect
│   ├── test_outcomes.py                   # ⬜ reward/cost ranges; θ=0 gives (0, 0) vs. Alice and Bob
│   ├── test_budget.py                     # ⬜ stop rule, reset mode == old behaviour, alive mask
│   ├── test_bwk_lp.py                     # ⬜ LP on toy instances (ratio arm + null arm, d=1)
│   ├── test_bwk_algorithms.py             # ⬜ UCB-BwK → stationary LP; SW-UCB-BwK window logic
│   ├── test_benchmarks.py                 # ⬜ dynamic LP; fixed-distribution grid
│   ├── test_policy_regret.py              # ⬜ re-simulation never mutates the live Bob
│   └── test_nonstationarity.py            # ⬜ V̂, Ŵ on a known piecewise-constant instance
│
├── data/
│   ├── raw/                               # read-only; optional external data (EC2 price archive, Swoopo), not committed
│   ├── processed/                         # small derived artefacts (fitted AR(1) parameters, selected windows)
│   └── synthetic/                         # optional saved simulation outputs
├── results/                               # generated; one folder per stage (stage0/, sweeps/, stage1/, …), git-ignored
├── notebooks/                             # exploration only, never final code
└── docs/
    ├── design_decisions.md                # ✅ must be kept current (sec. 6, 7, 11, 15, 17–20 still use the old stage numbering)
    ├── notation.md                        # ✅ to extend with BwK symbols (B, ρ, d, m, V1, V2, W1, W2, q̄)
    └── bwk_notes.md                       # ⬜ reading notes: Slivkins Ch. 10, Liu et al., Fikioris et al.
```

Design rules: `budget_mode = "reset"` stays the default so Waniek's replication keeps running; `dollar_auction.py` and `strategies.py` do not change for BwK; the **runner owns the stopping rule**, algorithms never see the remaining budget.

---

'''


## 10. Conventions

**Algorithm interface.** `select_arm() -> int`; `update(arm, reward, info=None)`; `reset()` optional. For BwK the cost travels in `info["cost"]` (no signature change); ELP still reads `info["agent_state_trace"]`. Algorithms do not see the remaining budget — SW-UCB-BwK uses `ρ = B/T`; the **runner owns the stopping rule**. Because the horizon is random (stopping time τ ≤ T), results are padded with an `alive` mask.

**Per-round order** (keeps Bob's regime a function of history up to t−1): snapshot Bob → `begin_round` → `select_arm` → auction → compute reward & cost → budget check/stop → `update` → `bob.observe_agent_move`.

**Hindsight rule.** Any counterfactual against Bob uses `bob.snapshot()`, never the live Bob; never cache `get_strategy()` closures.

**Randomness.** One seed per run → `SeedSequence.spawn` → independent learner / Bob / environment streams.

**Logged data** (Mesa `DataCollector`): model level `Round, Bob_Regime, P_Escalate, Winning_Bid, Cumulative_Reward, PH_Detected`; agent level `Agent_ID, Chosen_Arm, Reward`. Regret is post-hoc.

---

## 11. Setup & running
```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install "mesa>=3.0" networkx numpy pandas matplotlib seaborn scipy tqdm pytest pyyaml
```
`networkx` is needed because Mesa 3.5 imports it without declaring it. `pyyaml` is for configs. SMPyBandits is optional (old; may conflict with recent Python/NumPy).

```bash
pytest -v
python -m experiments.smoke_mesa
python -m experiments.calibrate_regimes
python -m experiments.sweep_bob --part A
python -m experiments.stage0_baseline_replication.run_stage0
```
Tooling: custom engine = core; Mesa = orchestration/logging only; tqdm in experiment drivers.

---

## 12. Findings so far (Stage 1 sweep; b=12, s=8, Bob mu=0.5, prior=3, rate=3; T=3000, 3 seeds — indicative)
| Policy | Mean reward |
|---|---|
| EXP3 / ELP | ≈ 0.575 |
| best **fixed** arm (θ=2) | 0.684 |
| regime-aware oracle (θ=2 when Rational, θ=0 when Escalating) | 0.727 |

Bob escalates in proportion to the learner's own aggressiveness, so a moderately cautious fixed arm keeps him mostly rational (θ=2 ⇒ ≈16 % escalating rounds; θ=12 ⇒ 99 %). EXP3/ELP do not find it: aggression looks good *this round* but causes later escalation. The gap (≈0.11/round) is a **policy-regret** effect; the value of perfect tracking over the best fixed arm is small (≈0.04). This is the empirical seed of RQ2.

---

## 13. Open decisions
| # | Decision |
|---|---|
| 1 | Resources: `d = 1` (money) only, or a second resource? (with `d = 1` BwK reduces to a reward/cost ratio index + null arm) |
| 2 | Budget level `ρ = B/T`: must be **below** the cost of the best-ratio arm for the constraint to bind (illustration: in the Stage-0 config the winning arms pay ≈0.3 of `b` per round). Needs `calibrate_budget.py` |
| 3 | Null arm = θ = 0 (verify reward 0 / cost 0 against Alice and Bob) or an explicit "skip auction" arm |
| 4 | Extra arm dimension (own-bid cap κ)? Not needed initially; the LP already meters spending |
| 5 | Non-BwK baseline set (EXP3, ELP, UCB1) and the budget-handling variants applied to them |
| 6 | Regret scheme of sec. 5 (re-simulation cost ≈ 10⁵–10⁶ auctions per run) |
| 7 | Bob control: `window = 0` is not constructible; add a **memoryless Bernoulli regime** with fixed p (same escalation frequency, no reaction) |
| 8 | Bob `min_dwell`, aggressiveness signal (`θ/b` vs `final_bid/b`) |
| 9 | LagrangeBwK: how to estimate OPT (the Liu et al. experiments plug in the exact value) |
| 10 | Stage 3: how epochs enter (cost → `V2/W2`, reward → `V1/W1`, or both) and the VM-cost definition |
| 11 | Which branch is the base for `feature/bwk` (GitHub `main` is an older snapshot) |

---

## 14. Reading list (in order)
| # | Read | Parts | Depth |
|---|---|---|---|
| 1 | Slivkins, *Introduction to Multi-Armed Bandits* (arXiv 1904.07272) | **Ch. 10 (BwK)** | full chapter |
| 2 | **Liu, Jiang & Li 2022** (arXiv 2205.12427) | Sec. 1–3, Appendix A (experiments) | skip proofs (App. B–F) |
| 3 | Agrawal & Devanur 2014 | the UCB-BwK algorithm | only what you implement |
| 4 | Fikioris et al. (arXiv 2302.14686) | Sec. 1–3: model, stationarity notion, **definition of the adversary** | targeted |
| 5 | Slivkins Ch. 5–6; Poland 2005; Arora–Dekel–Tewari 2012 | adaptive adversaries, policy regret | concepts |
| 6 | Immorlica et al. 2019 | only if LagrangeBwK is implemented | algorithm section |
| later | Ben-Yehuda, Menache, Abundo, Khandelwal | Stage 3 | already read once |

---

## 15. Collaboration & progress
Branches per feature (`feature/bwk`), PR + review before merging to `main`; keep `docs/design_decisions.md` current.

| Member | Responsibility |
|---|---|
| **A** | |
| **B** | |
| **Both** | |

- [x] Stage 0 · [x] adaptive Bob + snapshots · [x] runner/Mesa wrapper · [x] plots · [x] regime calibration tooling
- [ ] Stage 1 experiment · [ ] `outcomes.py` + `budget.py` · [ ] UCB-BwK · [ ] SW-UCB-BwK · [ ] baselines under budget · [ ] Stage 2a · [ ] regret/policy-regret/`V̂,Ŵ` · [ ] Stage 2b · [ ] Stage 3

---

## 16. References
1. Waniek, Tran-Thanh & Michalak (2016). *Repeated Dollar Auctions: A Multi-Armed Bandit Approach.* AAMAS.
2. Badanidiyuru, Kleinberg & Slivkins (2013). *Bandits with Knapsacks.* FOCS.
3. Agrawal & Devanur (2014). *Bandits with concave rewards and convex knapsacks.* EC.
4. Immorlica, Sankararaman, Schapire & Slivkins (2019). *Adversarial Bandits with Knapsacks.* FOCS.
5. Liu, Jiang & Li (2022). *Non-stationary Bandits with Knapsacks.* arXiv:2205.12427.
6. Rangi, Franceschetti & Tran-Thanh (2018). *Unifying the stochastic and the adversarial bandits with knapsack.* arXiv:1811.12253.
7. Slivkins (2019). *Introduction to Multi-Armed Bandits.* arXiv:1904.07272.
8. Fikioris et al. (2023). *Approximately Stationary Bandits with Knapsacks.* arXiv:2302.14686 (author list to be confirmed).
9. Auer, Cesa-Bianchi, Freund & Schapire (2002). *The Nonstochastic Multiarmed Bandit Problem.* SICOMP.
10. Mannor & Shamir (2011). *From Bandits to Experts: On the Value of Side-Observations.* NeurIPS.
11. Garivier & Moulines (2008). *On Upper-Confidence Bound Policies for Non-Stationary Bandit Problems.* arXiv:0805.3415.
12. Poland (2005). *FPL analysis for adaptive bandits.* SAGA.
13. O'Neill (1986). *International Escalation and the Dollar Auction.* J. Conflict Resolution.
14. Agmon Ben-Yehuda, Ben-Yehuda, Schuster & Tsafrir (2013). *Deconstructing Amazon EC2 Spot Instance Pricing.* ACM TEAC.
15. Menache, Shamir & Jain (2014). *On-demand, Spot, or Both.* ICAC.
16. Abundo, Di Valerio, Cardellini & Lo Presti (2014). *Bidding Strategies in QoS-Aware Cloud Systems Based on N-Armed Bandit Problems.* NCCA.
17. Khandelwal, Chaturvedi & Gupta (2018). *Bidding Strategies for Amazon EC2 Spot Instances — A Comprehensive Review.* ICCUBEA.