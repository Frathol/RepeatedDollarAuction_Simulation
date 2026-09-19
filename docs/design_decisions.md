# Design Decisions

This document logs every non-trivial design decision made for this project, along with the reasoning behind it. Keep this updated continuously as the project evolves — it is the single source of truth for "why did we do it this way", both for staying in sync between collaborators and for writing the thesis methodology chapter.

---

## 1. Adversarial bandit over stochastic (Bayesian/frequentist) bandit

**Decision**: The learning agent uses an *adversarial* multi-armed bandit framework (ELP, EXP3, EXP3.S), not a *stochastic* bandit framework (UCB, Thompson Sampling).

**Reasoning**: Stochastic bandit algorithms assume each arm has a fixed underlying success probability θ_k that is independent of the agent's own behavior, only needing to be estimated from accumulated data. In the Dollar Auction, the reward of a given strategy is not a fixed property of that strategy — it depends entirely on how the opponent reacts, and the opponent can be strategic and can change behavior abruptly and drastically (the sunk-cost fallacy transition). This violates the core assumption behind stochastic bandits. Adversarial bandits make no assumption about the source of rewards, and their regret guarantees hold even against a worst-case, adaptive opponent — making them the theoretically appropriate choice. This is also the choice made by our primary reference, Waniek, Tran-Thanh, & Michalak (2016).

**Rejected alternative**: Discounted Bayesian MAB (Thompson Sampling with a decay factor γ<1 on the Beta-Binomial update), as sketched in an early (fabricated/unverified) draft. Rejected because it still fundamentally assumes rewards drift slowly from a stochastic source, rather than being designed for strategic, abruptly-changing opponents.

---

## 2. Only the main agent learns; opponents are scripted

**Decision**: Alice and Bob (the opponent agents) are rule-based/reactive, not learning agents. Only the primary agent (running ELP/EXP3/EXP3.S) accumulates cross-round experience and optimizes its behavior.

**Reasoning**: The research question is specifically about whether a *learning* agent can remain robust and adapt efficiently when facing a population exhibiting *bounded rationality* (the sunk-cost fallacy). If opponents also learned optimally, the experiment would instead be testing competition between two learning algorithms — a different (and valid, but out of scope) question. Bob is deliberately scripted because sunk-cost fallacy is, by definition, a failure to rationally adjust behavior based on past outcomes — making him a learner would contradict the phenomenon being modeled.

**Note on "dynamic but not learning"**: Bob's behavior changes over time (t) and reacts to in-auction state (his own accumulated sunk cost, x), but this is *scripted* non-stationarity, not *learned* adaptation. From the learning agent's point of view, the environment is genuinely non-stationary — the agent does not know Bob's rules or switch points in advance.

**Optional future extension**: A scenario where one opponent also runs a bandit algorithm (mirroring Theorem 8 in Waniek et al., which shows convergence to Nash equilibrium when both players use ELP) may be added as a supplementary experiment, kept separate from the core design to avoid confounding the main switching-regret analysis.

---

## 3. Bob's escalation is probabilistic — breaking ELP's determinism assumption

**Decision**: Bob's decision to continue bidding past the sunk-cost threshold is modeled as *probabilistic* (an exponential escalation curve), not deterministic.

**Reasoning**: This is a deliberate and realistic choice — sunk-cost escalation in humans is not a hard deterministic switch, but a probability that increases with accumulated investment.

**Consequence (must be stated explicitly in the thesis)**: Waniek et al.'s Lemma 1 (the "prefix strategy" side-information trick used by ELP) requires opponent strategies to be deterministic — it relies on being able to "replay" a known sequence of moves to infer the outcome of untried strategies with certainty. A probabilistic opponent breaks this replay logic: playing the same strategy against Bob twice can yield different outcomes. This is the specific point at which our setting diverges from Waniek et al.'s formal guarantees, and it should be framed in the thesis as a deliberate relaxation, not an oversight. Two options for handling this going forward:
  (a) Evaluate ELP empirically under this relaxed assumption without re-deriving its regret bound (acceptable for a bachelor's thesis scope), explicitly noting the bound no longer formally applies;
  (b) Optionally adapt Lemma 1's estimator to an unbiased-estimator formulation (closer to the original Mannor & Shamir framing) — flagged as a stretch goal, not a requirement.

**Current default**: Option (a).

---

## 4. Static regret is an inadequate benchmark for our setting; switching regret is used instead

**Decision**: Algorithms are evaluated primarily on **switching/dynamic regret** (best strategy per phase), not on the static regret used in Waniek et al.'s original theorems.

**Reasoning**: Static regret (Waniek et al.'s Theorem 2 and related results) is measured against a single fixed best strategy played for the entire horizon T. Since Bob transitions between a rational phase and an escalation phase, no single fixed strategy is optimal across both phases — making static regret a weak benchmark for this setting. Note: "adversarial" (robustness to any reward source) and "static vs. dynamic regret" (choice of comparison benchmark) are independent properties — ELP happens to be both adversarial and evaluated via static regret, but this is a consequence of the specific proof technique inherited from Mannor & Shamir (2011), not an inherent requirement of adversarial bandits in general.

---

## 5. EXP3.S + Page-Hinkley as the primary algorithmic contribution

**Decision**: In addition to comparing ELP against plain EXP3.S, we integrate a Page-Hinkley changepoint detector that triggers an explicit weight reset in EXP3.S.

**Reasoning**: EXP3.S alone must pre-commit to an estimate of how many regime switches (S) will occur over the horizon; mis-estimating S causes either excessive vigilance (hurting performance during stable periods) or slow adaptation (hurting performance right after a genuine switch). Page-Hinkley provides an explicit, online signal for *when* a regime change has actually occurred, removing the need to guess S in advance.

**Algorithmic framing**: This is presented as a modification of the *reset/adaptation trigger*, not as a full replacement of the adversarial bandit paradigm — EXP3.S+PH remains within the same adversarial bandit family as ELP and EXP3.

---

## 6. Two Bob modes: single-switch and recurring-switch

**Decision**: Bob supports two temporal escalation patterns:

- **Single-switch** (Stage 1a): rational until round `switch_round`, then permanently escalating.
- **Recurring-switch** (Stage 1b): toggles between rational and escalating regimes multiple times, per a list of `switch_points`.

**Reasoning**: Single-switch with a long remaining horizon (e.g. switching at t=1000 out of T=10,000) is a scenario in which even standard no-regret algorithms (plain EXP3, or even UCB) can eventually adapt, given enough remaining rounds — making the value-add of EXP3.S+PH weaker to demonstrate in isolation. Recurring-switch is the scenario that genuinely demonstrates why an algorithm needs continuous responsiveness (rather than one-time slow adaptation) — this is where the contribution's practical value is most convincingly shown. Single-switch is retained as the cleaner, easier-to-interpret proof-of-concept (clear ground-truth changepoint, easy detection-delay measurement); recurring-switch is the robustness test.

---

## 7. N-player extension: scope and rule definitions

**Decision**: The N-player extension (Stage 2) uses a **generic (worst-case) regret bound** for ELP, not the tighter neighbourhood-graph-based bounds (Theorems 5/6 in Waniek et al.), and treats the N-1 opponents as a single joint environment/adversary from the learning agent's point of view.

**Reasoning**: Waniek et al.'s tighter regret bounds depend on specific side-information/neighbourhood-graph structure that has only been derived for the 2-player case. Re-deriving these bounds for N players is a substantial theoretical undertaking, out of scope for a 4-month bachelor's thesis. The generic worst-case bound (Theorem 2) still holds under the "joint environment" framing, since it makes no structural assumption about how many opponents produced the observed reward.

**Rules requiring explicit definition (not specified in the original paper)**:

- **Payment rule**: [TO BE DECIDED AND RECORDED HERE — e.g., "only the top two bidders pay their final bid" vs. "all bidders who placed at least one bid pay their final bid"]
- **Bidding turn order**: [TO BE DECIDED AND RECORDED HERE — e.g., round-robin among all N players]

*(Fill in the final decision here once agreed with the advisor, along with the justification.)*

---

## 8. Budget resets per auction (not persistent across rounds)

**Decision**: Following Waniek et al.'s original setting, each player's budget `b` is reset at the start of every auction round; losses in one round do not reduce the budget available in subsequent rounds.

**Reasoning**: This matches the original paper's formalization and keeps the bandit formulation clean (avoiding a shift toward a stateful, MDP-like problem). Waniek et al. explicitly acknowledge that extending to a persistent, shared budget across rounds is "not trivial" with their techniques — we treat this as a limitation and explicitly note it as future work rather than attempting it within this thesis's scope.

---

## 9. Reward representation: continuous normalization preferred over binarization

**Decision**: The primary reward signal used by all algorithms is `DollarAuction.normalize_reward()` — a continuous value in [0,1] derived from the actual payoff. `binarize_reward()` (success/failure only) is retained only as an optional baseline comparison.

**Reasoning**: Binarizing reward discards payoff magnitude information (e.g. it cannot distinguish a narrow win from a large one). Continuous normalization preserves more signal and is more consistent with the paper's stated assumption that `r_f(t) ∈ [0,1]` (Section 3.1), without requiring the reward to represent a literal binary success/failure event.

---

## 10. Default strategy set S0: O'Neill threshold-strategy family

**Decision**: The default arm set S0 is the family of "strategies with threshold θ" (Section 5.2, Waniek et al.), built on top of O'Neill's (1986) optimal pure strategy.

**Reasoning**: This choice deliberately aligns with Waniek et al.'s Theorem 6 setting, where the resulting neighbourhood graph is a clique (independence number α(G)=1), giving ELP its tightest published regret bound, O~(√T). Using this family keeps our baseline comparison to ELP fair and grounded in the paper's strongest theoretical result, rather than an arbitrary/simplified arm set (e.g. a small fixed set of opening bids, as used in the early flawed draft).

---

## 11. Staged validation: synthetic first, real data second

**Decision**: All algorithms are validated on fully controlled synthetic simulations (Stages 0, 1a, 1b, 2) before being tested against real auction data (Stage 3).

**Reasoning**: Waniek et al. provide no empirical validation whatsoever (their paper is purely theoretical, with no simulations or real data). There is therefore no empirical precedent to build directly on. Starting with synthetic data allows full control over ground-truth changepoints (needed to measure detection delay and switching regret precisely) before facing the noise and lack of ground truth inherent in real-world data. This staged approach was also the explicit recommendation of the thesis advisor.

**Real-world validation dataset**: The Swoopo penny-auction dataset (Byers, Mitzenmacher, & Zervas, 2010), chosen because its pay-per-bid, all-pay mechanism is structurally close to the Dollar Auction, it involves many simultaneous bidders (suited to the N-player extension), and it has academic precedent for sunk-cost fallacy analysis (Augenblick, 2015).

---

## 12. Random seed / reproducibility policy

**Decision**: [TO BE FINALIZED — recommended: every experiment run accepts an explicit `rng` (numpy `Generator`) seeded from a value stored in `src/simulation/config.py`; each stage's results are averaged over multiple seeds (recommended: ≥50-100 runs per configuration) with mean ± confidence interval reported, not single-run point estimates.]

**Reasoning**: A single simulation run is not sufficient evidence for any claim about regret, detection delay, or survival — this was one of the core methodological critiques of the earlier flawed draft (which reported single-run numbers with no replication). Reporting averages with confidence intervals across many seeds is required for any result presented as a thesis finding.
