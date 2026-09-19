# Dollar Auction Bandit — Mitigating Sunk-Cost Fallacy with Adversarial Multi-Armed Bandits

This thesis extends the framework of **Waniek, Tran-Thanh, & Michalak (2016), "Repeated Dollar Auctions: A Multi-Armed Bandit Approach"** (AAMAS 2016) in the following directions:

1. **Switching/dynamic regret** — replacing the evaluation benchmark from _static regret_ (single best fixed strategy over the whole horizon) with _switching regret_ (best strategy per phase), since an opponent can abruptly change behavioral regime due to _sunk-cost fallacy_.
2. **Change-responsive algorithms** — comparing ELP (original baseline) against EXP3, EXP3.S, and EXP3.S + **Page-Hinkley test** as an explicit change-detection mechanism.
3. **N-player extension** — generalizing from 2 players (Waniek et al.'s original setting) to a heterogeneous multi-player population.
4. **External validation** — testing against real auction data (Swoopo penny-auction dataset) after validation on synthetic simulations.

---

## 📁 Project Structure

```
dollar-auction-bandit/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│   ├── environment/
│   │   ├── __init__.py
│   │   ├── dollar_auction.py          # 2-player auction rules (state, bidding turns, payoffs)
│   │   ├── dollar_auction_nplayer.py  # N-player version (Stage 2)
│   │   └── strategies.py              # Definition of S0: the set of available strategies (arms)
│   │
│   ├── opponents/
│   │   ├── __init__.py
│   │   ├── alice_rational.py          # Rational agent (O'Neill threshold strategy)
│   │   ├── bob_sunkcost.py            # Sunk-cost agent (single-switch & recurring)
│   │   └── population.py              # Heterogeneous population generator (Stage 2)
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── elp.py                     # ELP algorithm (Waniek et al.)
│   │   ├── exp3.py                    # Base EXP3
│   │   ├── exp3s.py                   # EXP3.S (switching regret)
│   │   └── page_hinkley.py            # Changepoint detector
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── regret.py                  # Static regret & switching/dynamic regret
│   │   └── detection_delay.py         # Measures Page-Hinkley detection speed
│   │
│   └── simulation/
│       ├── __init__.py
│       ├── runner.py                  # Main loop: run T rounds, log results
│       └── config.py                  # All parameters (T, γ, δ, λ, Bob proportion, etc.)
│
├── experiments/
│   ├── stage0_baseline_replication/   # ELP 2-player replication (validation)
│   │   └── run_stage0.py
│   ├── stage1a_single_switch/         # Bob switches once
│   │   └── run_stage1a.py
│   ├── stage1b_recurring_switch/      # Bob switches back and forth
│   │   └── run_stage1b.py
│   ├── stage2_nplayer/                # Heterogeneous N-player population
│   │   └── run_stage2.py
│   └── stage3_real_data/              # Swoopo dataset validation
│       └── run_stage3.py
│
├── data/
│   ├── raw/
│   │   └── swoopo/                    # Original traces.tsv, outcomes.tsv (DO NOT modify)
│   ├── processed/
│   │   └── swoopo_cleaned.csv         # Preprocessed data ready for analysis
│   └── synthetic/
│       └── (optional, saved outputs from synthetic simulations)
│
├── results/
│   ├── stage0/
│   │   ├── figures/                   # Cumulative regret plots, etc.
│   │   └── tables/                    # Summary numbers (CSV/JSON)
│   ├── stage1a/
│   ├── stage1b/
│   ├── stage2/
│   └── stage3/
│
├── notebooks/
│   ├── 01_exploration_swoopo_data.ipynb
│   ├── 02_sanity_check_elp.ipynb
│   ├── 03_visualize_results.ipynb
│   └── 04_sensitivity_analysis.ipynb
│
├── tests/
│   ├── test_environment.py
│   ├── test_algorithms.py
│   └── test_page_hinkley.py
│
└── docs/
    ├── design_decisions.md            # Log of design decisions (N-player rules, etc.)
    └── notation.md                    # Notation glossary (following Table 1 in Waniek et al.)
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

- `elp.py` — ELP implementation (Algorithm 1 in Waniek et al.), leveraging the _prefix strategy_ structure for side-information.
- `exp3.py` — base EXP3 (Auer et al. 2002), adversarial bandit baseline for static regret.
- `exp3s.py` — EXP3.S (Auer et al. 2002, shifting/tracking variant), designed for switching regret.
- `page_hinkley.py` — standalone changepoint detection module, designed to plug into EXP3.S as a reset trigger.

### `src/metrics/`

Regret computation is kept separate from the algorithms so it's applied consistently across all experiments.

- `regret.py` — implements **static regret** and **switching/dynamic regret**.
- `detection_delay.py` — measures how many rounds Page-Hinkley needs to detect a changepoint from its ground-truth location.

### `src/simulation/`

- `runner.py` — main loop: run T rounds, call the environment and the algorithm, log results in a standard format.
- `config.py` — a single centralized place for all parameters (T, γ, δ, λ, Bob proportion, etc.) — avoid hardcoding parameters scattered across files.

### `experiments/`

Execution scripts per stage, calling components from `src/` with stage-specific configurations. Separates **reusable code** (`src/`) from **experiment execution code** (`experiments/`).

| Stage                         | Contents                                                                                                                  |
| ----------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `stage0_baseline_replication` | Replicate Waniek et al.'s original setup (2 players, ELP, static regret, Alice-only opponent) — implementation validation |
| `stage1a_single_switch`       | ELP vs EXP3 vs EXP3.S vs EXP3.S+PH, opponent Bob switches regime once                                                     |
| `stage1b_recurring_switch`    | Same as 1a, but Bob switches regime repeatedly (robustness test)                                                          |
| `stage2_nplayer`              | Extension to N players, heterogeneous population                                                                          |
| `stage3_real_data`            | External validation with the Swoopo dataset                                                                               |

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

## ⚙️ Environment Requirements

- **Python 3.10+**
- Core libraries:
  - `numpy` — numerical computation
  - `pandas` — simulation result data management
  - `matplotlib` / `seaborn` — visualization (cumulative regret plots, etc.)
  - `scipy` — ready-made statistical functions (distributions, etc.)
  - `pytest` — unit testing
  - `jupyter` — exploratory notebooks

```bash
pip install numpy pandas matplotlib seaborn scipy pytest jupyter
```

Save this list in `requirements.txt`:

```
numpy
pandas
matplotlib
seaborn
scipy
pytest
jupyter
```

---

## 🤝 Collaboration Conventions

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

### Simulation Result Data Format

Every simulation run logs per-round data in a standard structure (a `pandas.DataFrame` is recommended), with at minimum these columns:

| Column              | Description                                                                       |
| ------------------- | --------------------------------------------------------------------------------- |
| `round`             | Time index t                                                                      |
| `algorithm`         | Algorithm name (ELP, EXP3, EXP3.S, EXP3.S+PH)                                     |
| `arm_chosen`        | The selected strategy/arm                                                         |
| `reward`            | The observed reward                                                               |
| `cumulative_regret` | Cumulative regret up to this round                                                |
| `regime`            | Current opponent regime label (rational / escalating) — for analysis and plotting |

### Random Seed Policy

Agree on a seeding mechanism (e.g. explicit seed per run, logged in `config.py`) so every experiment can be reproduced identically.

### Git Workflow

- Use separate branches per feature/stage (e.g. `feature/exp3s`, `feature/page-hinkley`)
- Pull request + review before merging into `main`
- Descriptive commit messages, referencing the relevant experiment stage

---

## 👥 Work Division (proposed)

| Member   | Responsibility |
| -------- | -------------- |
| **A**    |                |
| **B**    |                |
| **Both** |                |

---

## 📚 Key References

1. Waniek, M., Tran-Thanh, L., & Michalak, T. (2016). _Repeated Dollar Auctions: A Multi-Armed Bandit Approach._ AAMAS 2016.
2. Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002). _The Nonstochastic Multiarmed Bandit Problem._ SIAM Journal on Computing, 32(1). (Source of EXP3 & EXP3.S)
3. Mannor, S., & Shamir, O. (2011). _From Bandits to Experts: On the Value of Side-Observations._ NeurIPS. (Foundation of the ELP algorithm)
4. Besbes, O., Gur, Y., & Zeevi, A. (2014). _Stochastic Multi-Armed-Bandit Problem with Non-stationary Rewards._ NeurIPS.
5. Garivier, A., & Moulines, E. (2011). _On Upper-Confidence Bound Policies for Switching Bandit Problems._
6. O'Neill, B. (1986). _International Escalation and the Dollar Auction._ Journal of Conflict Resolution, 30(1).
7. Augenblick, N. (2015). _The Sunk-Cost Fallacy in Penny Auctions._ (Source of Swoopo data validation)
8. Byers, J., Mitzenmacher, M., & Zervas, G. (2010). _Information Asymmetries in Pay-Per-Bid Auctions: How Swoopo Makes Bank._ (Source of the Swoopo dataset)

---

## 🗺️ Progress Status

- [ ] Stage 0 — Baseline ELP 2-player replication
- [ ] Stage 1a — Single-switch test (ELP vs EXP3 vs EXP3.S vs EXP3.S+PH)
- [ ] Stage 1b — Recurring-switch test
- [ ] Stage 2 — N-player extension
- [ ] Stage 3 — Swoopo data validation
