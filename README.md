# Simulation of Bandits with Knapsacks under Exogenous Non-Stationarity and Reactive Opponents in Repeated All-Pay Computational Resource Contention

**Simulasi Bandits with Knapsacks pada Ketidakstasioneran Eksogen dan Lawan Reaktif dalam Lelang All-Pay Berulang Sumber Daya Komputasi**

**Author:** Michael Mosses Christianto (235314092)
**Institution:** Informatics Study Program, Faculty of Science and Technology, Sanata Dharma University, Yogyakarta

---

## Abstract

Spot instance markets in cloud computing (such as Amazon EC2) offer substantial cost savings over on-demand instances at the risk of sudden capacity preemption. For workloads without checkpointing, preemption results in sunk resource commitments that are structurally analogous to sequential all-pay auctions (*Shubik's Dollar Auction*). Prior work models repeated dollar auctions with adversarial multi-armed bandits (`ELP`) under per-round budget resets and deterministic opponents. Long-term computational resource contention, however, operates under a finite global budget $B$ and non-stationary conditions.

This repository provides a reproducible simulation framework that evaluates **Bandits with Knapsacks (BwK)** and **Contextual Bandits with Knapsacks (CBwK)** against both **exogenous price volatility** (derived from historical Amazon EC2 spot price traces) and **endogenous opponent reactivity** (a bounded-memory opponent exhibiting sunk-cost escalation). It also investigates empirically whether the theoretical BwK non-stationarity metrics ($V_1, V_2, W_1, W_2$ and $\sigma_r, \sigma_c$) remain diagnostically meaningful when distribution shifts are triggered by the learner's own actions.

---

## 1. Research Questions

1. **RQ1 (Algorithmic Performance & Regret Growth):** How do budget-aware bandit algorithms (stationary `UCB-BwK`, context-conditioned `CTX-UCB-BwK`, and sliding-window `SW-UCB-BwK`) compare against unconstrained stochastic (`UCB1`, `Thompson Sampling`) and adversarial (`EXP3`, `ELP`) bandit baselines in repeated all-pay auctions under a global budget constraint, exogenous price fluctuations, and reactive opponents?
2. **RQ2 (Diagnostic Validity of Non-Stationarity Metrics):** Do the theoretical BwK non-stationarity measures remain meaningful when non-stationarity arises endogenously from a reactive opponent, both before and after conditioning on exogenous price contexts? The measures are local total variation ($V_1, V_2$) and global deviation ($W_1, W_2$) from Liu et al. (2022), and the approximate stationarity ratios ($\sigma_r, \sigma_c$) from Fikioris & Tardos (2023).

---

## 2. Problem Formulation

### 2.1. Repeated All-Pay Auction and Action Space

In each round $t \in \{1, \dots, T\}$, the learning agent and an opponent compete for a computational resource of value (stake) $s \in \mathbb{Z}_+$ under a per-round bidding cap $b \in \mathbb{Z}_+$ ($s < b$). Following O'Neill (1986) and Waniek et al. (2016), the learner selects a discrete threshold strategy $\theta_t \in \mathcal{S}_0 = \{0, 1, \dots, b\}$ ($K = b + 1$ arms):

- **Null arm ($\theta = 0$):** passes immediately without bidding ($\text{bid}_t = 0$). It costs nothing and is the safe, budget-preserving action.
- **Active threshold arms ($\theta \in \{1, \dots, b\}$):** follow O'Neill's backward-induction bidding ladder as long as the opponent's bid stays below $\theta$, and pass once it reaches or exceeds $\theta$.

Under the all-pay rule, both the winner and the loser pay their final bid $\text{bid}_t \in \{0, \dots, b\}$. Raw payoffs $\text{payoff}_t \in [-b, s]$ and resource consumption costs are normalized to $[0, 1]$:

$$\tilde{r}_t = \frac{\text{payoff}_t + b}{b + s} \in [0, 1], \qquad c_t = \text{clip}\left(\frac{m_t \cdot \text{bid}_t}{b}, \; 0, \; 1\right)$$

where $m_t > 0$ is the exogenous spot price multiplier at round $t$ ($m_t \equiv 1$ in the stages without price modulation). Playing the null arm yields a deterministic normalized reward of $\tilde{r}_t(0) = \frac{b}{b + s}$ at zero cost.

### 2.2. Opponent Models

- **Alice (`alice_rational.py`, stationary oblivious opponent):** a boundedly rational bidder with valuation parameter $\mu \in (0, 1]$. She keeps bidding as long as the required bid is strictly below $\mu \cdot s$ and folds otherwise.
- **Bob (`bob_sunkcost.py`, reactive bounded-memory opponent):** a stochastic finite-state adversary with a rolling memory window $W$ and a minimum regime dwell time $D$. Bob tracks the learner's normalized aggressiveness $\text{signal}_{t-1} = \theta_{t-1} / b$ over the last $W$ rounds ($\bar{s}_t$) and switches to an **Escalating** regime with probability $p_{\text{esc}}(t) = \text{clip}(\text{gain} \times \bar{s}_t, 0, 1)$. In the Escalating regime, Bob's bidding limit grows up to $b$ in proportion to his accumulated sunk cost.

### 2.3. Evaluation & Non-Stationarity Metrics

**Regret benchmarks:**

1. *Weak (standard) regret* $R_T^{\text{weak}}$: evaluated against the best fixed arm on top of the realized trajectory $(I_1, \dots, I_{t-1})$.
2. *Counterfactual policy regret* $R_T^{\text{pol}}$: evaluated against counterfactual trajectories in which the learner commits to a fixed strategy $(\theta, \dots, \theta)$ from round $t = 1$ (Arora et al., 2012).
3. *Dynamic LP regret* $R_T^{\text{dyn-LP}}$: evaluated against the offline dynamic linear program $\text{OPT}_{\text{dyn-LP}}$ and decomposed into the budget-pacing gap ($\text{OPT}_{\text{dyn-LP}} - \text{OPT}_{\text{per-round}}$) and the online estimation regret (Liu et al., 2022).

**Non-stationarity measures:**

- *Local total variation:*
  $V_1 = \sum_{t=1}^{T-1} \max_\theta |\mu_{t+1}(\theta) - \mu_t(\theta)|, \quad V_2 = \sum_{t=1}^{T-1} \max_\theta |c_{t+1}(\theta) - c_t(\theta)|$
- *Global deviation:*
  $W_1 = \sum_{t=1}^{T} \max_\theta |\mu_t(\theta) - \bar{\mu}(\theta)|, \quad W_2 = \sum_{t=1}^{T} \max_\theta |c_t(\theta) - \bar{c}(\theta)|$
- *Approximate stationarity ratios:*
  $\sigma_r = \max_{t,\theta} \max\left(\frac{\mu_t(\theta)}{\bar{\mu}(\theta)}, \frac{\bar{\mu}(\theta)}{\mu_t(\theta)}\right), \quad \sigma_c = \max_{t,\theta} \max\left(\frac{c_t(\theta)}{\bar{c}(\theta)}, \frac{\bar{c}(\theta)}{c_t(\theta)}\right)$

---

## 3. Evaluated Algorithms

| Category | Algorithm | Primary Reference | Mechanism |
| :--- | :--- | :--- | :--- |
| **Stochastic MAB** | `UCB1` | Auer et al. (2002) | Empirical mean with a Hoeffding upper-confidence bonus (budget-blind). |
| **Stochastic MAB** | `Thompson Sampling` | Agrawal & Goyal (2012) | Beta-Bernoulli posterior sampling with Bernoulli-trial updates (budget-blind). |
| **Adversarial MAB** | `EXP3` | Auer et al. (2002) | Exponential weights with uniform exploration and IPS estimation (budget-blind). |
| **Adversarial MAB** | `ELP` | Waniek et al. (2016); Mannor & Shamir (2011) | Exponential weights exploiting threshold-prefix side observations (budget-blind). |
| **Stationary BwK** | `UCB-BwK` | Agrawal & Devanur (2014) | Full-history optimistic (UCB reward, LCB cost) linear program solved each round. |
| **Contextual BwK** | `CTX-UCB-BwK` | Badanidiyuru et al. (2014); Slivkins (2019) | Statistics kept per exogenous price regime; the linear program is conditioned on the regime observed at the start of the round. |
| **Non-Stationary BwK** | `SW-UCB-BwK` | Liu et al. (2022) | Sliding-window ($w$) optimistic linear program with forced re-exploration. |

Under a global budget, every BwK algorithm falls back to the null arm once the remaining budget cannot cover one more round. The budget-blind baselines receive the same hard-stop fallback so that comparisons stay fair.

---

## 4. Experimental Stages

| Stage | Environment & Opponent | Budget Regime | Price Multiplier ($m_t$) | Primary Objective |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0** | Stationary (`Alice`) | Per-round reset | Constant ($m_t = 1$) | Baseline replication of Waniek et al. (2016) and verification of prefix feedback. |
| **Stage 1** | Reactive (`Bob`) | Per-round reset | Constant ($m_t = 1$) | Quantify the divergence between weak regret and counterfactual policy regret. |
| **Stage 2a** | Stationary (`Alice`) | Global budget ($B = \rho T$) | Constant ($m_t = 1$) | Validate BwK budget pacing against budget-blind baselines in stationary play. |
| **Stage 2b** | Reactive (`Bob` and a memoryless control) | Global budget ($B = \rho T$) | Constant ($m_t = 1$) | Compare `SW-UCB-BwK` with `UCB-BwK` and examine path-dependent endogenous $V, W, \sigma$. |
| **Stage 3** | Factorial: (`Alice`, `Bob`) $\times$ EC2 traces | Global budget ($B = \rho T$) | Amazon EC2 (`us-east-1`, 2018-2019) | Separate the exogenous price context (`CTX-UCB-BwK`) from endogenous opponent shifts (`SW-UCB-BwK`). |

For **Stage 3**, the exogenous price multipliers come from historical Amazon EC2 spot price traces (`Linux/UNIX (Amazon VPC)`, region `us-east-1`, 2018-2019). The traces are stratified into four active volatility classes (`P25`, `P50`, `P75`, `P95` by daily total variation) plus a flat control series.

---

## 5. Repository Structure

```text
dollar-auction-bandit/
├── README.md
├── AGENTS.md                              # Specification for AI coding assistants
├── requirements.txt
├── pytest.ini
├── data/                                  # Amazon EC2 spot price series (2018-2019)
├── docs/                                  # Project documentation
├── notebooks/
│   ├── ec2_2018.ipynb                     # EC2 spot price exploration, us-east-1 2018
│   └── ec2_2019.ipynb                     # EC2 spot price exploration, us-east-1 2019
├── src/
│   ├── environment/
│   │   ├── dollar_auction.py              # Two-player all-pay auction engine and Min-Max normalization
│   │   └── strategies.py                  # Threshold strategy family and O'Neill bidding ladder
│   ├── opponents/
│   │   ├── alice_rational.py              # Stationary boundedly rational opponent (Alice)
│   │   └── bob_sunkcost.py                # Reactive bounded-memory opponent with sunk-cost escalation (Bob)
│   ├── algorithms/
│   │   ├── base.py                        # Abstract bandit algorithm interface
│   │   ├── ucb1.py                        # UCB1
│   │   ├── thompson_sampling.py           # Beta-Bernoulli Thompson Sampling
│   │   ├── exp3.py                        # EXP3
│   │   ├── elp.py                         # ELP with prefix feedback
│   │   ├── bwk.py                         # LP solver, UCB-BwK, SW-UCB-BwK, and CTX-UCB-BwK
│   │   └── page_hinkley.py                # Page-Hinkley change detector
│   ├── simulation/
│   │   ├── calibration.py                 # O'Neill opening bid and regime calibration utilities
│   │   ├── config.py                      # Simulation configuration and default parameters
│   │   └── runner.py                      # Multi-seed simulation loop and regret computation
│   └── metrics/
│       ├── plots.py                       # Visualization suite and RunBundle container
│       └── nonstationarity.py             # V_1, V_2, W_1, W_2, sigma_r, sigma_c (results stored in RunBundle.extras)
├── experiments/
│   ├── calibrate_regimes.py               # Opponent regime separation and parameter calibration
│   ├── sweep_bob.py                       # Parameter sweep for Bob
│   ├── smoke_mesa.py                      # Smoke-test script
│   ├── stage0_baseline_replication/
│   │   └── run_stage0.py                  # Stage 0: stationary baseline against Alice
│   ├── stage1_adaptive_bob/
│   │   └── run_stage1.py                  # Stage 1: reactive opponent Bob without global budget
│   ├── stage2_nplayer/
│   │   └── run_stage2.py                  # Stage 2: global-budget BwK evaluation (Alice and Bob)
│   └── stage3_real_data/
│       └── run_stage3.py                  # Stage 3: EC2 price traces with reactive opponent
├── tests/
│   ├── test_environment.py                # Auction mechanics and normalization
│   ├── test_opponents.py                  # Alice and Bob dynamics
│   ├── test_bob_sunkcost.py               # Bob's reactive memory
│   ├── test_calibration.py                # O'Neill opening bid
│   ├── test_ucb1_and_ts.py                # UCB1 and Thompson Sampling
│   ├── test_bwk.py                        # LP solver and the BwK/CBwK family
│   ├── test_page_hinkley.py               # Page-Hinkley detector
│   ├── test_runner.py                     # Simulation reproducibility
│   ├── test_plots.py                      # Plotting utilities
│   └── test_nonstationarity.py            # Non-stationarity metrics
└── results/                               # Experiment logs, summary tables, and figures (one folder per stage)
```

---

## 6. Installation and Usage

### Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Running the unit tests

```bash
pytest -v
```

### Running the experimental stages

Run all commands from the repository root.

```bash
# Stage 0: baseline replication against the stationary opponent (Alice)
python -m experiments.stage0_baseline_replication.run_stage0

# Stage 1: reactive opponent (Bob) without a global budget
python -m experiments.stage1_adaptive_bob.run_stage1

# Stage 2: global-budget BwK evaluation (2a vs. Alice, 2b vs. Bob)
python -m experiments.stage2_nplayer.run_stage2

# Stage 3: exogenous Amazon EC2 price traces with contextual BwK
python -m experiments.stage3_real_data.run_stage3
```

---

## 7. Key References

1. Shubik, M. (1971). The Dollar Auction Game: A Paradox in Noncooperative Behavior and Escalation. *Journal of Conflict Resolution*, 15(1), 109-111.
2. O'Neill, B. (1986). International Escalation and the Dollar Auction. *Journal of Conflict Resolution*, 30(1), 33-50.
3. Waniek, M., Tran-Thanh, L., & Michalak, T. (2016). Repeated Dollar Auctions: A Multi-Armed Bandit Approach. In *Proceedings of the 15th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)*, 579-587.
4. Mannor, S., & Shamir, O. (2011). From Bandits to Experts: On the Value of Side-Observations. In *Advances in Neural Information Processing Systems (NeurIPS)*, 24, 684-692.
5. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time Analysis of the Multiarmed Bandit Problem. *Machine Learning*, 47(2-3), 235-256.
6. Auer, P., Cesa-Bianchi, N., Freund, Y., & Schapire, R. E. (2002). The Nonstochastic Multiarmed Bandit Problem. *SIAM Journal on Computing*, 32(1), 48-77.
7. Agrawal, S., & Goyal, N. (2012). Analysis of Thompson Sampling for the Multi-Armed Bandit Problem. In *Proceedings of the 25th Annual Conference on Learning Theory (COLT)*. arXiv:1111.1797.
8. Arora, R., Dekel, O., & Tewari, A. (2012). Online Bandit Learning against an Adaptive Adversary: from Regret to Policy Regret. In *Proceedings of the 29th International Conference on Machine Learning (ICML)*.
9. Badanidiyuru, A., Kleinberg, R., & Slivkins, A. (2013). Bandits with Knapsacks. In *IEEE 54th Annual Symposium on Foundations of Computer Science (FOCS)*, 207-216.
10. Badanidiyuru, A., Langford, J., & Slivkins, A. (2014). Resourceful Contextual Bandits. In *Proceedings of the 27th Conference on Learning Theory (COLT)*, PMLR 35.
11. Agrawal, S., & Devanur, N. R. (2014). Bandits with Concave Rewards and Convex Knapsacks. In *Proceedings of the 15th ACM Conference on Economics and Computation (EC)*, 989-1006.
12. Agrawal, S., Devanur, N. R., & Li, L. (2016). An Efficient Algorithm for Contextual Bandits with Knapsacks, and an Extension to Concave Objectives. In *Proceedings of the 29th Annual Conference on Learning Theory (COLT)*, PMLR 49.
13. Agrawal, S., & Devanur, N. R. (2016). Linear Contextual Bandits with Knapsacks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 29.
14. Chen, Y., Lee, C.-W., Luo, H., & Wei, C.-Y. (2019). A New Algorithm for Non-stationary Contextual Bandits: Efficient, Optimal, and Parameter-free. In *Proceedings of the 32nd Annual Conference on Learning Theory (COLT)*, PMLR 99.
15. Slivkins, A. (2019). Introduction to Multi-Armed Bandits. *Foundations and Trends in Machine Learning*, 12(1-2), 1-286.
16. Liu, S., Jiang, J., & Li, X. (2022). Non-stationary Bandits with Knapsacks. In *Advances in Neural Information Processing Systems (NeurIPS)*, 35. arXiv:2205.12427.
17. Fikioris, G., & Tardos, E. (2023). Approximately Stationary Bandits with Knapsacks. In *Proceedings of the 36th Annual Conference on Learning Theory (COLT)*, PMLR 195, 3758-3782. arXiv:2302.14686.
18. Ardi, C. (2025). Amazon EC2 Spot Price History: 2014-2015, 2017-2023. Zenodo. https://doi.org/10.5281/zenodo.5880792