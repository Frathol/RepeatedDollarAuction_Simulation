# Dollar Auction Bandit — Adaptive Adversary

> **Major design pivot (see `docs/design_decisions.md` §13–18 for full rationale):**
> The opponent (Bob) is no longer a purely *scripted/oblivious* agent whose regime
> switches on a fixed, pre-determined schedule. Bob is now a **genuine adaptive
> adversary** in the formal bandit-theory sense: his probability of entering the
> "escalation" (sunk-cost) regime at round *t* depends on the **history of arms the
> agent actually played in previous rounds**, not on a wall-clock schedule. This
> directly matches the definition of adaptive adversary used to justify adversarial
> bandits (EXP3-family) in the first place, and is the reason this project keeps that
> framework rather than switching to non-stationary *stochastic* bandit methods
> (Discounted UCB, Sliding-Window UCB), whose regret guarantees formally require
> reward to be independent of the algorithm's own choices — a requirement this
> design deliberately violates.

This thesis extends the framework of **Waniek, Tran-Thanh, & Michalak (2016), "Repeated Dollar Auctions: A Multi-Armed Bandit Approach"** (AAMAS 2016) in the following directions:

1. **Adaptive opponent** — Bob's regime transitions are driven by the agent's own recent bidding history (an aggressiveness signal built from past rounds), not a fixed schedule. This is what makes the adversarial-bandit framework's guarantees (which explicitly cover adaptive adversaries, per Poland 2005, cited in Waniek et al.'s footnote 2) actually necessary rather than merely convenient.
2. **Switching/dynamic regret** — replacing the evaluation benchmark from *static regret* (single best fixed strategy over the whole horizon) with *switching regret* (best strategy per phase), since the opponent's regime can now change as an emergent consequence of the interaction itself.
3. **Change-responsive algorithms** — comparing ELP (original baseline, with an explicitly flagged validity caveat under an adaptive opponent — see below) against EXP3, EXP3.S, and EXP3.S + **Page-Hinkley test** as an explicit change-detection mechanism.
4. **N-player extension** — from 2 players to a heterogeneous multi-player population, where the "provocation" signal driving escalation can be aggregated across the whole population's interaction history.
5. **Optional asymmetric-cost regret layer** — weighting "slow to detect escalation" more heavily than "false suspicion of a still-rational opponent," reflecting the asymmetric cost structure typical of security/detection domains (the motivating application area for this thesis).
6. **External validation** — testing against real auction data (Swoopo penny-auction dataset) after validation on synthetic simulations.

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
│   │   ├── __init__.py
│   │   ├── dollar_auction.py          # UNCHANGED by the adaptive pivot: still a pure
│   │   │                                function of (agent_strategy, opponent_strategy).
│   │   │                                It has no knowledge of whether the opponent is
│   │   │                                adaptive -- that logic lives entirely in Bob.
│   │   ├── dollar_auction_nplayer.py  # N-player version (Stage 2)
│   │   └── strategies.py              # Definition of S0: the set of available strategies (arms)
│   │
│   ├── opponents/
│   │   ├── __init__.py
│   │   ├── alice_rational.py          # Rational agent (O'Neill threshold strategy) --
│   │   │                                still fully stationary; serves as the control
│   │   │                                baseline contrasting with adaptive Bob.
│   │   ├── bob_sunkcost.py            # MAJOR REVISION: Bob now keeps cross-round state
│   │   │                                (a rolling window of the agent's recent arm
│   │   │                                choices) and probabilistically enters escalation
│   │   │                                based on that history, instead of a fixed
│   │   │                                switch_round/switch_points schedule.
│   │   └── population.py              # Heterogeneous population generator (Stage 2);
│   │                                    aggressiveness signal now aggregated across the
│   │                                    whole population's interaction history.
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── base.py                    # Unchanged interface
│   │   ├── elp.py                     # ELP (Waniek et al.) -- VALIDITY CAVEAT: Lemma-1
│   │   │                                replay remains valid WITHIN a single round (Bob's
│   │   │                                regime is fixed before that round's auction
│   │   │                                starts), but is NOT valid for hindsight spanning
│   │   │                                multiple rounds without a full re-simulation --
│   │   │                                see design_decisions.md §14.
│   │   ├── exp3.py                    # EXP3 -- unaffected by the pivot; its regret
│   │   │                                guarantee already covers adaptive adversaries.
│   │   ├── exp3s.py                   # EXP3.S (switching regret)
│   │   └── page_hinkley.py            # Changepoint detector -- role is now MORE central:
│   │                                    it is also used post-hoc to check whether
│   │                                    detected changepoints line up with the recorded
│   │                                    (realized, not pre-fixed) true regime-switch times.
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── regret.py                  # Static regret & switching/dynamic regret --
│   │   │                                dynamic regret's hindsight computation now
│   │   │                                requires re-simulating each candidate phase-wide
│   │   │                                strategy against a FRESH copy of Bob (so Bob
│   │   │                                reacts naturally to that candidate strategy),
│   │   │                                not a cheap replay of the original trace.
│   │   └── detection_delay.py         # Measures Page-Hinkley detection speed relative
│   │                                    to the REALIZED (recorded during simulation)
│   │                                    regime-switch times, not a fixed constant.
│   │
│   └── simulation/
│       ├── __init__.py
│       ├── runner.py                  # Main loop -- now also calls
│       │                                bob.observe_round_outcome(arm_played) AFTER each
│       │                                round, and logs bob.current_regime() BEFORE each
│       │                                round to build the realized ground-truth
│       │                                regime-switch record.
│       └── config.py                  # Parameters (T, gamma, delta, lambda, window size W
│                                        for Bob's aggressiveness memory, etc.)
│
├── experiments/
│   ├── stage0_baseline_replication/   # ELP 2-player replication vs Alice (stationary
│   │                                    control) -- validation, unaffected by the pivot
│   ├── stage1a_single_switch/         # RENAMED IN SPIRIT: opponent regime change is no
│   │                                    longer a single fixed switch_round, but the
│   │                                    first realized escalation episode -- retained as
│   │                                    a simpler "mostly one episode" configuration
│   ├── stage1b_recurring_switch/      # Opponent regime naturally oscillates as a
│   │                                    consequence of ongoing interaction, not a
│   │                                    pre-set switch_points list
│   ├── stage2_nplayer/                # Heterogeneous N-player population
│   └── stage3_real_data/              # Swoopo dataset validation
│
├── data/            (unchanged)
├── results/         (unchanged)
├── notebooks/       (unchanged)
├── tests/
│   ├── test_environment.py
│   ├── test_opponents.py              # NEEDS NEW TESTS for Bob's adaptive regime logic
│   │                                    (e.g. aggressive play history should raise
│   │                                    escalation probability; a frozen/copied Bob used
│   │                                    for hindsight must not mutate the original)
│   └── test_algorithms.py
│
└── docs/
    ├── design_decisions.md            # See §13-18 for the full adaptive-pivot rationale
    └── notation.md                    # See the new "Adaptive Bob" symbol table
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

---

## 🗺️ Progress Status

- [ ] Stage 0 — Baseline ELP 2-player replication
- [ ] Stage 1a — Single-switch test (ELP vs EXP3 vs EXP3.S vs EXP3.S+PH)
- [ ] Stage 1b — Recurring-switch test
- [ ] Stage 2 — N-player extension
- [ ] Stage 3 — Swoopo data validation
