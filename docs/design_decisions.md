# Design Decisions

This document logs every non-trivial design decision made for this project, along with the reasoning behind it. Keep this updated continuously as the project evolves.

> **Revision note:** §1 and §2 below were revised after adopting the adaptive-adversary design (see §13). The original versions justified the adversarial-bandit framework partly by analogy ("opponent is strategic") and partly by non-stationarity alone; both were found to be imprecise once checked against the formal definition of *adaptive adversary*. The revised versions below reflect the corrected, formally grounded reasoning.

---

## 1. Adversarial bandit over stochastic (Bayesian/frequentist) bandit

**Decision**: The learning agent uses an *adversarial* multi-armed bandit framework (ELP, EXP3, EXP3.S), not a *stochastic* bandit framework (UCB, Thompson Sampling), and not a *non-stationary stochastic* framework (Discounted UCB, Sliding-Window UCB) either.

**Reasoning (revised)**: The formal definition of an adaptive adversary (see e.g. Slivkins 2019, Ch. 5-6; verified via search) is: the reward/loss vector at round *t* may depend on the algorithm's history of chosen arms in *previous rounds*, π₁, …, π_{t-1}. Bob (§13) is deliberately designed to satisfy exactly this: his probability of entering the escalation regime at round *t* is a function of the agent's arm choices in a rolling window of prior rounds. This is not merely "non-stationary" (reward changing over time on some fixed schedule) — it is reward that changes *because of* the algorithm's own choices.

This distinction matters concretely: non-stationary *stochastic* methods (Discounted UCB, Sliding-Window UCB, CUSUM-UCB) all require, for their regret proofs to hold, that the reward-generating process be independent of the algorithm's own action sequence (i.i.d. within each stationary segment, segment boundaries fixed independently of the algorithm). Bob's design explicitly violates this requirement. Adversarial bandit algorithms (EXP3 and its descendants) retain valid regret guarantees under adaptive adversaries — Auer et al. (2002)'s original proof covers oblivious adversaries, but Poland (2005) provides the argument extending it to adaptive adversaries, which Waniek et al. cite explicitly (footnote 2) to justify applying ELP/Mannor-Shamir-style guarantees in their (already adversarial-by-design) setting.

**Consequence for baselines**: Discounted UCB / Sliding-Window UCB are **not** included as baselines in this design, because their theoretical guarantees do not apply here — including them would require either restricting the comparison to a regime where Bob happens to behave close to oblivious (defeating the purpose of the adaptive design) or reporting their performance without a valid theoretical backing (acceptable only as an empirical curiosity, clearly labeled as such, not as a principled baseline).

**Rejected alternative**: Discounted Bayesian MAB (Thompson Sampling with a decay factor), as sketched in an early (fabricated/unverified) draft. Rejected because it assumes an i.i.d. reward source per segment, which Bob's design violates outright.

---

## 2. Only the main agent learns; opponents are scripted

**Decision**: The learning agent (running ELP/EXP3/EXP3.S) is the only one that maintains an internal model whose purpose is to *maximize* its own long-run reward. Bob maintains internal state (a rolling window of the agent's recent arm choices) and reacts to it, but this state is not used to optimize any objective of Bob's own — it merely determines a probability of entering a scripted escalation behavior.

**Reasoning**: This distinction is worth being precise about now that Bob has cross-round memory (§13), which could superficially look like "learning." The key difference: a learning agent (ELP/EXP3/EXP3.S) chooses actions specifically to improve a measured objective (cumulative reward) based on feedback about that objective. Bob's state update has no such objective — it is a fixed, hand-designed reactive function (aggressiveness score → escalation probability) that does not adapt or improve over time in response to how well "escalating" serves any goal of Bob's. Bob is *adaptive* (in the bandit-theory sense of §1) but not *learning* (in the optimization sense). Both properties are independent; Bob has the former but not the latter.

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

---

## 13. Bob redesigned as a genuine adaptive adversary

**Decision**: Bob's regime transition (rational → escalation) is no longer driven by a fixed `switch_round`/`switch_points` schedule. Instead, Bob maintains a rolling window (size `W`, a new tunable parameter) of the agent's most recently played arm thresholds (θ values), computes an **aggressiveness score** from that window (e.g. the mean θ played), and derives an **escalation probability** from that score (e.g. `p_escalate = min(1, aggressiveness_score / budget)`, exact functional form TBD/tunable). At the start of each round, Bob draws whether it enters escalation for that round based on this probability.

**Reasoning**: See README preamble and `notation.md` — this both (a) satisfies the formal definition of adaptive adversary, directly justifying the adversarial-bandit framework choice (§1) rather than relying on analogy, and (b) is arguably a more psychologically faithful model of sunk-cost fallacy, which in the literature is typically triggered by a pattern of repeated aggressive engagement rather than an external clock.

**What stays unchanged**: Once Bob has entered escalation for a given round, the within-auction dynamics (sunk-cost threshold, exponential escalation-probability curve, escalation ceiling) are exactly as before (see original `bob_sunkcost.py` logic). Only the *mechanism deciding which regime applies this round* has changed.

**Within-round vs. cross-round reactivity — do not conflate these**: Bob (and Alice) have always been reactive *within* a single auction (their strategy function depends on the live bid state (x, y) of that auction) — this is inherent to the f(x,y) formalization in Waniek et al. and does **not**, by itself, make an opponent an "adaptive adversary" in the bandit-theoretic sense. Only dependence on the agent's arm choices in **previous bandit rounds** counts. Conflating these two notions of "reactive" was an error caught and corrected during this project's development; keep the distinction explicit in any write-up.

---

## 14. ELP's Lemma-1 validity under an adaptive opponent — a per-round vs. per-phase distinction

**Decision**: ELP's prefix-replay side-information mechanism (Lemma 1) remains usable **within a single round**, but is **not** valid for computing multi-round hindsight (needed for regret) via cheap trace-replay; multi-round hindsight requires full re-simulation.

**Reasoning**: Bob's regime for round *t* is fully determined *before* round *t*'s auction begins (it depends only on history up to round *t*-1). So within round *t*, Bob behaves as a fixed (if possibly probabilistic) strategy, and the existing Lemma-1 replay logic (see `elp.py`, `_replay_lemma1`) is unaffected for inferring "what arm *g* would have earned this round, given the auction played out as recorded." This is a **single-round** counterfactual, and remains valid.

What is **not** valid: extending this to "what would my total reward have been across an entire phase if I had played arm *g* consistently?" (needed for both static and dynamic regret hindsight, see `run_stage0.py`'s `hindsight_sums` pattern). If arm *g* were actually played across the phase, the agent's revealed aggressiveness history would differ from what actually happened, which would change Bob's realized escalation-probability draws in every subsequent round of that phase — a genuinely different simulated trajectory, not something derivable from the original trace. Multi-round/phase-wide hindsight under an adaptive opponent requires re-simulating the phase from scratch with a frozen copy of Bob's pre-phase state (see §16).

**Consequence**: `elp.py`'s existing per-round replay code does not need to change. What changes is upstream, in how experiment scripts compute the hindsight benchmark needed for regret (§16) — that logic must not naively reuse per-round replayed values as if they compose into a valid multi-round counterfactual.

---

## 15. Ground-truth regime-switch times are recorded during simulation, not fixed in advance

**Decision**: `Bob.is_escalating_at(t)`-style ground truth is replaced by a **runtime log**: at the start of every round, before that round's auction is played, the runner records `(t, bob.current_regime())` into a per-run history. This realized log — not a pre-set constant — is what detection-delay and phase-boundary calculations use downstream.

**Reasoning**: Because escalation is now a probabilistic function of the agent's own play history, the actual round(s) at which Bob enters escalation can differ across seeds, and even across algorithms (an algorithm that happens to play aggressively early may provoke escalation sooner than a conservative one). There is no longer a single "true" switch_round known in advance; it is a realized outcome of each individual simulation run.

**Consequence for experiment naming**: Stage 1a ("single switch") and Stage 1b ("recurring switch") are retained as configuration *regimes* (e.g. tuning Bob's escalation-probability function and window size so that, empirically, escalation episodes tend to be rare/single vs. frequent/recurring), not as literal fixed schedules. Document the realized switch-time distribution observed across seeds for each configuration as part of reporting results — this distribution is itself a meaningful thing to report, not just a nuisance to average away.

---

## 16. Dynamic regret hindsight computation under an adaptive opponent requires full re-simulation

**Decision**: To compute the phase-wide hindsight term `max_g Σ_{t∈phase} r_g(t)` needed for dynamic/switching regret (see prior static-regret critique, unchanged), each candidate arm *g* must be evaluated by **re-simulating the entire phase from a frozen snapshot of Bob's state at the start of that phase**, with Bob reacting naturally (via its normal adaptive logic) to *g* being played consistently — not by replaying/reusing the original trace.

**Reasoning**: See §14. This is a direct consequence of Bob's adaptivity: a counterfactual "what if arm g had been played throughout this phase" is only meaningful if Bob is allowed to react to that counterfactual history, which requires genuinely running the simulation forward under that counterfactual, not inferring it from what actually happened under a different arm sequence.

**Practical implication (must be implemented carefully)**: Any hindsight-computation code path must operate on a **deep-copied / frozen** Bob instance, distinct from the live Bob instance being updated by the real simulation loop. Accidentally sharing state between the two would silently corrupt both the real run and the regret calculation. This is now the single most important correctness invariant in the codebase and should be covered by an explicit unit test (confirm that running hindsight evaluations does not alter the original Bob instance's internal history).

**Cost**: This is computationally more expensive than the old single-replay approach (full phase re-simulation per candidate arm, rather than one cheap trace replay), which should be accounted for when choosing `T`, phase granularity, and the number of seeds for Stage 1a/1b experiments.

---

## 17. N-player: aggressiveness signal aggregated across the population

**Decision**: In the N-player extension, each Bob-type opponent's escalation probability is driven by an aggressiveness score aggregated across **the entire population's** recent interaction history (e.g. the mean θ played by anyone against anyone, within the rolling window), not just that individual opponent's own direct interaction history with the agent.

**Reasoning**: This captures a more realistic "contagion" dynamic — one aggressive participant can provoke elevated escalation risk across the population, not just in their direct opponent — and is a natural, low-complexity way to make the N-player extension meaningfully different from N independent copies of the 2-player case. Exact aggregation function (population mean, some other statistic) is TBD and should be treated as a tunable design choice, documented here once finalized.

---

## 18. Asymmetric-cost regret — optional extension layer

**Decision (optional, not yet committed)**: A variant of switching regret may be defined with asymmetric weighting: rounds where the algorithm was slow to detect an actual escalation are penalized more heavily than rounds where it over-cautiously treated a still-rational opponent as risky.

**Reasoning**: Reflects the asymmetric cost structure typical of security/detection domains (false negatives costlier than false positives) — see the corresponding discussion in the topic-narrative chat log. This is a re-weighting of the *evaluation* of realized outcomes; it does not change how Bob behaves or how the environment/algorithms operate. It can be added independently of, and on top of, the adaptive-Bob design (§13) and the dynamic-regret machinery (§16) without further structural changes.

**Status**: Not yet implemented; flagged as a candidate scope addition pending advisor confirmation on time budget, per the earlier discussion of trade-offs between narrative depth (security relevance) and implementation risk.

## 19. Application domain re-scoped: computational resource contention, not cooperative robot task allocation

**Decision**: The illustrative application domain for this thesis is **competitive allocation of scarce computational resources** among autonomous agents (e.g. processes/nodes bidding for exclusive GPU/bandwidth slots, inspired by real cloud spot-instance bidding systems such as AWS Spot Instances / GCP Preemptible VMs) — **not** cooperative multi-robot physical task allocation (Nanjanath & Gini, 2010).

**Reasoning**: Nanjanath & Gini's algorithm is explicitly cooperative (robots share a common team objective and voluntarily pass tasks to reduce total team cost) and purely deterministic (no learning, no bandit formulation at all — see critique logged during paper review). It cannot be adapted to this thesis's competitive, adversarial-bandit framework without essentially discarding it and designing a new scenario from scratch. Computational resource contention (spot-instance-style bidding) was chosen instead because it is: (a) inherently competitive (agents have individual, not shared, objectives), (b) naturally all-pay (resources/time already committed to a lost bid are not recovered), and (c) has a real, documented sunk-cost-like dynamic (agents with significant uncommitted progress have strong incentive to overbid to avoid losing it), making the Bob/sunk-cost escalation mechanism already built directly applicable without forced analogy. It also connects naturally to the thesis's security/network motivation (resource contention and fairness in distributed systems).

**Consequence**: `design_decisions.md` §17 (N-player aggregate aggressiveness score) and the N-player extension (`dollar_auction_nplayer.py`, not yet implemented) should be described in terms of this domain going forward (multiple agents simultaneously contending for a shared resource slot), not generic "lelang dolar" framing alone.

---

## 20. Round indexing remains event-driven in spirit, not wall-clock time — no architecture change needed

**Decision**: Round `t` continues to represent "the *t*-th resource-contention event" (one full auction), not a fixed wall-clock time step. This is retained as-is from the original design; no change to `dollar_auction.py` or the round-based simulation loop is required.

**Reasoning**: Nanjanath & Gini's system is event-driven because it models physical robot movement with unpredictable completion times. This project's regret formalism (static regret, switching/dynamic regret, all defined as `U_A(T)` over a discrete round index `t = 1, ..., T`) fundamentally requires a countable sequence of decision points — abandoning round-based indexing for literal wall-clock event simulation would break the ability to compute regret at all. The existing design — where each round is already triggered by "a new contention event arising," not an artificial fixed interval — already satisfies the spirit of event-driven modeling without sacrificing the bandit formalism. This should be stated explicitly in the thesis write-up to preempt the question, rather than treated as a gap to fix.

---

## 21. Tooling: custom engine remains the core; SMPyBandits and Mesa used narrowly, not as the simulation backbone

**Decision**:

- **tqdm**: adopted trivially as a progress-bar utility in experiment scripts (`run_stage*.py`). No architectural impact.
- **SMPyBandits**: NOT used as a simulation framework. Its multi-player algorithms (MusicalChair, MEGA, rhoRand, MCTopM/RandTopM) are designed for a **collision-avoidance** problem structure (multiple players simultaneously selecting from a shared arm set, penalized on collision — the cognitive-radio/spectrum-access setting), which is structurally incompatible with this project's bidding/auction/adaptive-adversary structure. It MAY be used narrowly to import 1-2 single-player baseline algorithm implementations (e.g. UCB1, Thompson Sampling) as additional comparison baselines in Stage 0/1a, purely for convenience (avoiding re-implementing well-known algorithms already available), not as the environment or core simulation engine.
- **Mesa**: NOT used for the core bandit/regret computation, which remains in the custom-built environment (`src/environment/`, `src/algorithms/`, `src/simulation/`). MAY be used, as an **optional, later-stage** addition, purely as a visualization/presentation layer (grid-based spatial representation of N agents contending for resources, using Mesa's built-in browser visualization) wrapped around results already computed by the custom engine — not as a replacement for it.

**Reasoning**: The three tools have incompatible underlying design philosophies (Mesa: step-based simultaneous-agent scheduling for spatial ABM; SMPyBandits: standalone numerical bandit simulation research tool, explicitly documented as "not meant to be a library you can use elsewhere"; this project's custom engine: round-based adversarial-bandit simulation with explicit regret/detection-delay instrumentation already validated at Stage 0). Forcing them into one unified framework from the start would consume significant integration effort for no gain to the thesis's core contribution (EXP3.S, Page-Hinkley, dynamic regret, N-player adaptive contention), and risks introducing bugs at integration boundaries not under the author's full understanding. The custom engine, already tested and working (Stage 0), remains authoritative; other tools are adopted only where they provide clear, narrow, low-risk value (extra baseline algorithms; optional visualization), consistent with the project's existing staged validation philosophy (§11).
