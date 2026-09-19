# Notation Glossary

Consistent notation used across code, documentation, and the thesis write-up. Based primarily on the notation table (Appendix, Table 1) in Waniek, Tran-Thanh, & Michalak (2016), extended with notation needed for the switching-regret, EXP3.S, and Page-Hinkley additions. When writing code or thesis text, use these symbols/names consistently rather than inventing new ones.

---

## Dollar Auction (game-level notation)

| Symbol | Meaning |
|---|---|
| `i`, `j` | The two players participating in a single auction |
| `s` | Stake / prize value of the auction (also written `V` in some earlier drafts — prefer `s` to match the paper) |
| `δ` (delta) | Minimal bid increment (assumed = 1 throughout, following the paper) |
| `b` | Budget of both players (equal budgets assumed) |
| `X_b` | The set of valid states `(x, y)` where a player can be asked to bid |
| `x` | The last bid made by the player currently choosing their bid (their own accumulated commitment — relevant to Bob's sunk-cost logic) |
| `y` | The last bid made by that player's opponent |
| `S` | The set of *all* valid strategies in the auction (potentially very large/combinatorial) |
| `S0` | The *subset* of strategies actually made available to the bandit algorithm as arms (manually chosen by the researcher — see `design_decisions.md` §10) |
| `f`, `g` | A strategy: a function `f(x, y) → bid`, representing a complete contingency plan for the whole auction |
| `f(0,0) = 0` | Convention: the player lets the opponent move first |
| `f(x,y) = y` | Convention: the player passes / folds at this state |
| `p_i` | Raw payoff of player `i` from one auction: `s − y_i` if won, `−y_i` if lost (all-pay rule) |

---

## Bandit formulation (round-level notation)

| Symbol | Meaning |
|---|---|
| `T` | Total number of rounds (i.e., total number of separate auctions played in sequence) |
| `t` | Index of the current round, `t = 1, ..., T`. **One round = one complete auction**, not one individual bid — do not confuse with the in-auction state `(x, y)` above |
| `K` | Number of arms, `K = \|S0\|` |
| `A` | The bandit meta-algorithm (ELP / EXP3 / EXP3.S / EXP3.S+PH) selecting a strategy at each round |
| `a_t`, `f_t` | The arm/strategy selected by the algorithm at round `t` |
| `r_f(t)` | Reward obtained from using strategy `f` at round `t`, normalized to `[0, 1]` (see `DollarAuction.normalize_reward`) |
| `R_A(T)` | Total (cumulative) reward of algorithm `A` after `T` rounds |
| `θ_k` | The (hidden, stochastic-bandit-only) true success probability of arm `k` — relevant to UCB/Thompson Sampling discussion, not used directly by adversarial algorithms |
| `n_k(t)` | Number of times arm `k` has been played up to round `t` (used in UCB's confidence bonus) |
| `μ̂_k(t)` | Empirical (observed) mean reward of arm `k` up to round `t` |

---

## Regret

| Symbol | Meaning |
|---|---|
| `U_A(T)` | **Static regret** of algorithm `A` after `T` rounds: `max_{g∈S0} Σ_t r_g(t) − Σ_t r_{f_t}(t)` — compares against a single best fixed strategy over the whole horizon |
| Switching / dynamic regret | Compares against the best strategy **per phase** rather than a single fixed strategy; no single fixed symbol in the paper (they don't define it) — in our code/writing, denote it `U_A^{switch}(T)` to distinguish clearly from static `U_A(T)` |
| `regime` | Label attached to each round indicating Bob's true behavioral phase at that round (`rational` / `escalating`), used as ground truth for computing switching regret per phase and for measuring Page-Hinkley detection delay. Corresponds to the `regime` column in the standard results DataFrame (see README) |

---

## ELP-specific notation (Mannor & Shamir / Waniek et al.)

| Symbol | Meaning |
|---|---|
| `N_f` | Neighbourhood of strategy `f`: the set of strategies that are a *prefix* of `f` (i.e., agree with `f` up to some point, then pass) |
| `G` | The neighbourhood graph over `S0`, induced by the prefix relation |
| `α(G)` | Independence number of the neighbourhood graph `G` — controls the tightness of ELP's regret bound |
| `w_f(t)` | Weight assigned to strategy `f` at round `t` (ELP's internal state, analogous to EXP3's weights) |
| `P_f(t)` | Probability of selecting strategy `f` at round `t`, derived from `w_f(t)` |
| `r̂_f(t)` | Estimated reward for strategy `f` at round `t`, inferred via the prefix relation (Lemma 1) without actually playing `f` |
| `β`, `γ(t)`, `q_f(t)` | Tuning parameters of the ELP algorithm (exploration rate, mixing weight, side-information weighting) |
| `θ` (theta) | Threshold parameter of a threshold strategy `f_θ` (Section 5.2) — **note**: this is a different, unrelated `θ` from the stochastic-bandit `θ_k` above; disambiguate by context (auction-strategy threshold vs. hidden arm success probability) |

---

## EXP3 / EXP3.S notation

| Symbol | Meaning |
|---|---|
| `η` (eta) | Learning rate / exploration parameter of EXP3 |
| `p_k(t)` | Probability of playing arm `k` at round `t` under EXP3/EXP3.S |
| `α` (EXP3.S mixing rate) | **Note**: unrelated to ELP's `α(G)` or the Beta-distribution `α` below — this is the EXP3.S-specific parameter controlling how much probability mass is mixed toward the uniform distribution at each step, to enable recovery after a switch. Disambiguate explicitly in writing (`α_{EXP3.S}` if needed) |
| `S` (switching budget) | In the EXP3.S regret bound, an upper bound on the *number of regime switches* expected over the horizon — **not** to be confused with the strategy set `S` above; disambiguate by context |

---

## Page-Hinkley test notation

| Symbol | Meaning |
|---|---|
| `X_t` | Performance metric being monitored at round `t` (e.g. normalized reward) |
| `μ_t` | Running average of `X_t` up to round `t` |
| `δ` (drift parameter) | **Note**: unrelated to the auction's bid increment `δ` above — this `δ` is a small allowable drift tolerance to prevent false alarms from minor noise. Disambiguate explicitly (`δ_{PH}` if needed) |
| `U_t` | Cumulative deviation statistic: `Σ_{τ=1}^{t} (X_τ − μ_τ − δ)` |
| `m_t` | Running minimum of `U_t` observed so far: `min_{1≤τ≤t} U_τ` |
| `λ` (lambda) | Detection threshold: a changepoint is flagged when `U_t − m_t > λ` |

---

## Bayesian bandit notation (Thompson Sampling — used for comparison/critique only, not as the primary algorithm)

| Symbol | Meaning |
|---|---|
| `α_k`, `β_k` | Beta-distribution parameters for arm `k`: `α_k = 1 +` (observed successes), `β_k = 1 +` (observed failures) |
| `θ̃_k` | A single random sample drawn from `Beta(α_k, β_k)` at a given round — **not** the same as `θ_k` (the true hidden probability) or the auction threshold `θ` above; disambiguate by context (`θ̃` with a tilde always denotes a sampled value) |
| `γ` (discount factor) | Used only in the *rejected* discounted-Bayesian-MAB baseline (see `design_decisions.md` §1): `γ ∈ (0,1]` decay applied to historical `α`/`β` counts. Not used by the primary algorithms |

---

## Opponent-agent parameters (Alice & Bob)

| Symbol / Name | Meaning |
|---|---|
| `μ` (mu, Alice/Bob rational-phase parameter) | **Note**: unrelated to the Page-Hinkley running average `μ_t` or the stochastic-bandit `μ̂_k` above — this `μ` is the fraction of the stake beyond which the rational agent folds (e.g. `μ=0.8` folds once their own bid would exceed 80% of the stake). Disambiguate explicitly in writing if all three appear in the same section |
| `sunk_cost_threshold` | The amount of Bob's own accumulated bid (`x`) within one auction beyond which escalation dynamics can trigger |
| `escalation_rate` | Rate parameter of Bob's exponential escalation-probability curve |
| `escalation_ceiling` | Hard cap on how high Bob will ever bid while escalating |
| `switch_round` | (single-switch mode) The round after which Bob permanently enters escalation mode |
| `switch_points` | (recurring-switch mode) Sorted list of rounds at which Bob's regime flips |

---

## Symbol Collision Warning

Several symbols are reused across different sub-literatures with different meanings. Always disambiguate by context or subscript when multiple appear in the same section of the thesis:

- **`δ`**: auction bid increment vs. Page-Hinkley drift tolerance
- **`θ`**: auction threshold-strategy parameter vs. stochastic-bandit hidden success probability `θ_k` vs. sampled value `θ̃_k`
- **`α`**: ELP's `α(G)` (graph independence number) vs. Beta-distribution parameter `α_k` vs. EXP3.S mixing rate
- **`S`**: strategy set vs. EXP3.S's switching-budget parameter
- **`μ`**: Alice/Bob's fold-threshold fraction vs. Page-Hinkley's running average vs. stochastic-bandit true mean
