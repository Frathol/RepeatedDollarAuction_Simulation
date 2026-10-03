# Dollar Auction Bandit — Adaptive Adversary

> **Major design pivot (see `docs/design_decisions.md` §13–18 for full rationale):**
> The opponent (Bob) is no longer a purely *scripted/oblivious* agent whose regime
> switches on a fixed, pre-determined schedule. Bob is now a **genuine adaptive
> adversary** in the formal bandit-theory sense: his probability of entering the
> "escalation" (sunk-cost) regime at round *t* depends on the **history of arms the
> agent actually played in previous rounds**, not on a wall-clock schedule. This
> directly matches the definition of adaptive adversary used to justify adversarial
> bandits (EXP3-family) in the first place, and is the reason this project keeps that
> framework rather than switching to non-stationary *stochastic* bandit methods.

This thesis extends the framework of **Waniek, Tran-Thanh, & Michalak (2016), "Repeated Dollar Auctions: A Multi-Armed Bandit Approach"** (AAMAS 2016) in the following directions:

1. **Adaptive opponent** — Bob's regime transitions are driven by the agent's own recent bidding history.
2. **Switching/dynamic regret** — replacing the evaluation benchmark from static regret to switching regret.
3. **Change-responsive algorithms** — comparing ELP against EXP3, EXP3.S, and EXP3.S + **Page-Hinkley test**.
4. **Computational Resource Contention (N-player extension)** — scaling from 2 players to an N-player environment modeling cloud/edge spot instance bidding and all-pay resource contention.
5. **Robust Simulation Orchestration** — leveraging the Mesa framework for strict event scheduling and memory-safe data collection over massive horizons.

---

## 📁 project structure

```
dollar-auction-bandit/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── environment/
│   │   ├── dollar_auction.py          # Core 2-player logic (Xb formalization)
│   │   ├── dollar_auction_nplayer.py  # N-player computational resource contention
│   │   └── strategies.py              # Strategy set S0 definition
│   │
│   ├── opponents/
│   │   ├── alice_rational.py          # Rational control baseline
│   │   ├── bob_sunkcost.py            # Adaptive adversary (memory-based escalation)
│   │   └── population.py              # Heterogeneous N-player generator
│   │
│   ├── algorithms/
│   │   ├── base.py                    # Interface: select_arm() and update()
│   │   ├── elp.py                     # ELP with Lemma-1 prefix side-information
│   │   ├── exp3.py                    # Classic EXP3
│   │   ├── exp3s.py                   # EXP3.S (switching regret variant)
│   │   └── page_hinkley.py            # Standalone changepoint detection module
│   │
│   ├── metrics/
│   │   ├── regret.py                  # Static and dynamic/switching regret computation
│   │   └── detection_delay.py         # Measures Page-Hinkley detection speed
│   │
│   └── simulation/
│       ├── runner.py                  # Mesa Model orchestration and DataCollector
│       └── config.py                  # Centralized experiment parameters
│
├── experiments/
│   ├── stage0_baseline_replication/   # Implementation validation vs Alice
│   ├── stage1a_single_switch/         # Single-episode escalation test
│   ├── stage1b_recurring_switch/      # Continuous adaptation robustness test
│   ├── stage2_nplayer/                # Computational resource bidding
│   └── stage3_real_data/              # Swoopo dataset validation
│
├── data/            # /raw, /processed, /synthetic
├── results/         # Output tables and figures per stage
├── notebooks/       # Exploratory analysis
├── tests/           # Pytest unit tests for core mechanics
└── docs/            # Design decisions and notation glossaries
```

---

## 📂 Folder Descriptions

### `src/environment/`

The "world" where the auction takes place.

- `dollar_auction.py` — core 2-player logic: state `(x, y)` definition, payment rules, auction termination. Follows the `Xb` formalization in Waniek et al. (Section 2.1).
- `dollar_auction_nplayer.py` — generalization to N players (Stage 2). Payment rules (who must pay when the auction ends) and bidding order **must be defined explicitly**, since the original paper provides no precedent for this case — document these decisions in `docs/design_decisions.md`.
- `strategies.py` — defines the strategy set `S0` (arms), e.g. the family of O'Neill threshold strategies with varying `θ`.

### `src/opponents/`

Scripted/rule-based opponents, not learners (see the note in `docs/design_decisions.md` on why only the main agent learns).

- `alice_rational.py` — rational agent, follows a fixed profit-maximizing threshold rule.
- `bob_sunkcost.py` — sunk-cost fallacy agent. Supports **single-switch** mode (permanent one-time regime change, for Stage 1a) and **recurring-switch** mode (repeated regime changes, for Stage 1b).
- `population.py` — generator for a heterogeneous Alice/Bob population with configurable proportions (for Stage 2).

### `src/algorithms/`

The core contribution. All algorithms **must follow the same interface** (see "Interface Convention" below) so they can be swapped in `runner.py` without special-case code.

- `elp.py` — ELP implementation (Algorithm 1 in Waniek et al.), leveraging the *prefix strategy* structure for side-information.
- `exp3.py` — base EXP3 (Auer et al. 2002), adversarial bandit baseline for static regret.
- `exp3s.py` — EXP3.S (Auer et al. 2002, shifting/tracking variant), designed for switching regret.
- `page_hinkley.py` — standalone changepoint detection module, designed to plug into EXP3.S as a reset trigger.

### `src/metrics/`

The "Engine" of the project. It defines *how* the simulation runs, but does not dictate *what* specific scenario is being tested.

- `runner.py` — The core simulation engine. Wraps the environment and algorithms into a `mesa.Model`. It handles the execution of a single simulation run (T rounds) and safely logs data using Mesa's `DataCollector`. 
- `config.py` — A single centralized place for default hyperparameters (T, γ, δ, λ, Bob proportion, etc.) to prevent hardcoding across files.

### `src/simulation/`

- `runner.py` — main loop: run T rounds, call the environment and the algorithm, log results in a standard format.
- `config.py` — a single centralized place for all parameters (T, γ, δ, λ, Bob proportion, etc.) — avoid hardcoding parameters scattered across files.

### `experiments/`

The "Drivers" of the project. These execution scripts import the engine from `src/simulation/runner.py` and inject specific stage configurations. They are responsible for orchestrating multi-seed execution (using `tqdm`), extracting the Pandas DataFrames from Mesa, generating Matplotlib charts, and saving the final CSVs to the `results/` folder.

| Stage                         | Contents                                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `stage0_baseline_replication` | Replicate Waniek et al.'s original setup (2 players, ELP, static regret, Alice-only opponent) — implementation validation |
| `stage1a_single_switch`       | ELP vs EXP3 vs EXP3.S vs EXP3.S+PH, opponent Bob switches regime once (Single-episode escalation test)                    |
| `stage1b_recurring_switch`    | Same as 1a, but Bob switches regime repeatedly (Robustness test for the Page-Hinkley reset mechanism)                     |
| `stage2_nplayer`              | Extension to N players, modeling computational resource contention and spot instance bidding                              |
| `stage3_real_data`            | External validation with the Swoopo dataset                                        |

### `data/`

- `raw/` — original files, **read-only**, never overwrite/modify directly.
- `processed/` — cleaned/preprocessed results ready for analysis.
- `synthetic/` — optional, for saving synthetic simulation outputs to file (so they can be reproduced without re-running).

### `results/`

Outputs per stage (figures & tables) kept in separate folders — makes writing the thesis results chapter easier, just pull from the folder matching the chapter being written.

### `notebooks/`

For quick exploration, interactive debugging, and visualization — not production/final code.

### `tests/`

Unit tests to ensure the implementation is correct before it's used for any scientific claim. Example required test: "if all arms have equal reward, the arm-selection probability should approach uniform."

### `docs/`

- `design_decisions.md` — **must be kept continuously updated**. Log every design decision (N-player payment rules, why the generic regret bound is used instead of the improved version, Bob's escalation definition, etc.) so both of you stay in sync and don't forget the reasoning when writing the thesis.
- `notation.md` — mathematical notation glossary, following the style of Table 1 (Appendix) in Waniek et al., to keep notation consistent across code and writing.

---

### 🌉 Domain Mapping: Dollar Auction to Computational Systems
To bridge the classical game-theoretic model with distributed systems, we map the core mechanics of the Dollar Auction as follows:
* **The Stake ($s$):** Exclusive access to a limited computational resource (e.g., a GPU slot, critical network bandwidth, or spot instance time).
* **The Bid ($x$):** Resource commitment (e.g., CPU cycles spent, prioritized packets, or compute time allocated).
* **The Budget ($b$):** The maximum computational capacity or timeout threshold of a specific node/container.
* **The All-Pay Mechanism:** Preemption loss. If a process is outbid and forcibly preempted before completing its task, the computational resources already invested are lost without yielding a result (sunk-cost).

---

## ⚙️ Environment Requirements
This project uses an isolated Python virtual environment to manage dependencies securely without conflicting with system packages.

**1. Create and activate the virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate
```

- **Python 3.10+**
- Core libraries:
  - `numpy` — numerical computation
  - `pandas` — simulation result data management
  - `matplotlib` / `seaborn` — visualization (cumulative regret plots, etc.)
  - `scipy` — ready-made statistical functions (distributions, etc.)
  - `pytest` — unit testing
  - `jupyter` — exploratory notebooks

```bash
pip install -r requirements.txt
```

Core libraries `requirements.txt`:

```
numpy
pandas
matplotlib
seaborn
scipy
pytest
jupyter
tqdm
mesa
SMPyBandits
```

---

## 🤝 Collaboration Conventions

### Simulation Orchestration via Mesa

To handle multi-round bandit loops cleanly and prevent memory leaks over thousands of iterations, this project utilizes Mesa.
- The DollarAuction environment remains pure logic.
- src/simulation/runner.py wraps the environment in a mesa.Model.
- Agents (Learner and Bob) are wrapped as mesa.Agent.
- Per-round data logging is handled entirely by mesa.DataCollector. Avoid manual list appending for global simulation states.

### Algorithm Interface (must be agreed on before coding)

Every algorithm class in `src/algorithms/` must expose the same methods, at minimum:

```python
class BanditAlgorithm:
    def select_arm(self) -> int:
        """Select an arm for this round."""
        ...

    def update(self, arm: int, reward: float) -> None:
        """Update internal state based on the observed reward."""
        ...
```

This lets `runner.py` call any algorithm (ELP, EXP3, EXP3.S) with the same code, without algorithm-specific if-else branches.

### Simulation Result Data Format (via Mesa DataCollector)
Data is collected automatically without manual looping, outputting two distinct Pandas DataFrames:

**1. Model-Level Data (System wide metrics per round)**
| Column              | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| `Round`             | Time index t (one complete resource contention event)                             |
| `Cumulative_Regret` | The learning agent's total static/dynamic regret up to round t                    |
| `Bob_Regime`        | Current opponent regime label (Rational / Escalating)                             |
| `Winning_Bid`       | The highest resource commitment that won the execution slot                       |

**2. Agent-Level Data (Per-agent metrics per round)**
| Column              | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| `Agent_ID`          | Unique identifier (Learner, Alice, Bob_1, Bob_N)                                  |
| `Chosen_Arm`        | The strategy threshold ($\theta$) selected by the agent                           |
| `Reward`            | Normalized payoff (Positive if slot won, negative based on sunk-cost if preempted)|

### Random Seed Policy

Agree on a seeding mechanism (e.g. explicit seed per run, logged in `config.py`) so every experiment can be reproduced identically.

### Git Workflow

- Use separate branches per feature/stage (e.g. `feature/exp3s`, `experiment/stage0`)
- Pull request + review before merging into `main`
- Descriptive commit messages, referencing the relevant experiment stage

**One addition specific to the pivot:** any function that computes a *hindsight* or *counterfactual* outcome against Bob (used for regret computation) **must** operate on a frozen/deep-copied snapshot of Bob's internal state, never the live Bob instance being updated by the real simulation loop. Mixing the two silently corrupts the regret calculation. Document any such snapshot logic clearly at the call site.

---

## 👥 Work Division (proposed)

| Member   | Responsibility |
| -------- | -------------- |
| **A**    |                |
| **B**    |                |
| **Both** |                |

---

## 📚 Key References

1. Waniek, M., Tran-Thanh, L., & Michalak, T. (2016). *Repeated Dollar Auctions: A Multi-Armed Bandit Approach.* AAMAS 2016.
2. Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002). *The Nonstochastic Multiarmed Bandit Problem.* SIAM Journal on Computing, 32(1). (Source of EXP3 & EXP3.S)
3. Mannor, S., & Shamir, O. (2011). *From Bandits to Experts: On the Value of Side-Observations.* NeurIPS. (Foundation of the ELP algorithm)
4. Besbes, O., Gur, Y., & Zeevi, A. (2014). *Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards.* NeurIPS.
5. Garivier, A., & Moulines, E. (2011). *On Upper-Confidence Bound Policies for Switching Bandit Problems.*
6. O'Neill, B. (1986). *International Escalation and the Dollar Auction.* Journal of Conflict Resolution, 30(1).
7. Augenblick, N. (2015). *The Sunk-Cost Fallacy in Penny Auctions.* (Source of Swoopo data validation)
8. Byers, J., Mitzenmacher, M., & Zervas, G. (2010). *Information Asymmetries in Pay-Per-Bid Auctions: How Swoopo Makes Bank.* (Source of the Swoopo dataset)
9. Poland, J. (2005). *FPL analysis for adaptive bandits.* 3rd Symposium on Stochastic Algorithms, Foundations and Applications (SAGA'05). — cited by Waniek et al. (footnote 2) as the argument extending the Mannor–Shamir regret proof to adaptive adversaries; central to justifying the adaptive-Bob design.
10. Zheng, H., et al. (2015). *Bidding for Highly Available Services with Low Price in Spot Instance Market.* (Or any general paper on Cloud/Edge Spot Instance bidding and preemption).
11. Zaman, S., & Grosu, D. (2013). *Combinatorial Auction-Based Allocation of Virtual Machine Instances in Clouds.* IEEE Transactions on Parallel and Distributed Systems.

---

## 🗺️ Progress Status

- [ ] Stage 0 — Baseline ELP 2-player replication
- [ ] Stage 1a — Single-switch test (ELP vs EXP3 vs EXP3.S vs EXP3.S+PH)
- [ ] Stage 1b — Recurring-switch test
- [ ] Stage 2 — N-player extension
- [ ] Stage 3 — Swoopo data validation
