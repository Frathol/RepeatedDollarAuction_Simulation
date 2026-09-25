"""
elp.py

Full implementation of the ELP algorithm (Algorithm 1, Section 3.3 of
Waniek, Tran-Thanh, & Michalak, 2016), adapted to the Dollar Auction.

This is the paper's original baseline -- the thing Stage 0 is meant to
replicate before we start extending anything. Read alongside:
    - docs/notation.md for every symbol used below (w_f, P_g, gamma(t),
      q_f(t), N_f, epsilon, beta)
    - src/environment/strategies.py for how S0 (the threshold-strategy
      family) is built
    - Section 3.2 (Lemma 1) and Section 5 (Theorem 6) of the paper

DESIGN DECISION -- coupling to the threshold-strategy family:
    A fully generic ELP would need a way to compute, for ANY pair of
    strategies (f, g) in an arbitrary S0, whether g is a prefix of f.
    Doing that generically (comparing two arbitrary Python functions
    over the entire state space X_b) is both expensive and unnecessary
    here, because this thesis's S0 is *always* the threshold-strategy
    family (design_decisions.md, Sec. 10), for which Theorem 6 proves a
    much simpler structural fact: f_theta is a prefix of f_Theta
    whenever theta <= Theta. So this implementation takes the arms'
    threshold values directly and computes the (static, time-invariant)
    neighbourhood structure once, analytically, instead of doing
    generic runtime function comparison. If S0 is ever generalized
    beyond the threshold family, this class's `_build_neighbourhoods`
    method is the one place that would need to change.

HOW LEMMA 1 IS APPLIED HERE:
    After playing arm f_t, we don't just observe r_{f_t}(t) -- we also
    receive `agent_state_trace`, the ordered sequence of (x, y) states
    the agent encountered this round (see AuctionResult in
    dollar_auction.py). For every g that is a prefix of f_t, we replay
    that exact trace against g's own strategy function:
        - if g never chooses to pass at any state in the trace, g must
          have followed f_t's actual path exactly (guaranteed by the
          prefix property for the threshold family) -> r_g(t) = r_f(t)
        - if g passes at some state (x_j, y_j) in the trace, that is
          exactly the point where g folds -> r_g(t) = normalize(-x_j)
    This mirrors the proof of Lemma 1 directly, rather than
    hard-coding a threshold-specific shortcut formula, so the logic
    stays legible against the paper's argument.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

from src.algorithms.base import BanditAlgorithm

StrategyFn = Callable[[int, int], int]


class ELP(BanditAlgorithm):
    """
    ELP (Exponentially weighted algorithm with Linear Programming),
    adapted to the Dollar Auction per Waniek et al. (2016), Algorithm 1.

    Parameters
    ----------
    strategies : Sequence[StrategyFn]
        The S0 arm set, as actual callable strategy functions (needed
        to replay Lemma 1's side-information logic).
    thetas : Sequence[int]
        The threshold value of each strategy in `strategies`, in the
        SAME order. Used to determine the (structural, time-invariant)
        prefix/neighbourhood relation analytically (Theorem 6).
    stake, budget : float, int
        Auction parameters, needed to normalize the -x_j fold cost
        computed via Lemma 1 into the same [0,1] reward scale as
        everything else (see DollarAuction.normalize_reward).
    beta : float, optional
        Learning rate parameter (see notation.md). If not given,
        defaults to the Corollary 7-optimal value for the threshold
        family, computed from `horizon_T` (requires horizon_T to be
        passed).
    horizon_T : int, optional
        Total number of rounds T, used only to compute the default
        beta via `corollary7_beta` if beta is not given explicitly.
    epsilon : float, optional
        Floor probability parameter (see notation.md); defaults to a
        small constant matching q_f(t) uniform and the resulting
        gamma(t).
    rng : numpy.random.Generator, optional
    """

    def __init__(
        self,
        strategies: Sequence[StrategyFn],
        thetas: Sequence[int],
        stake: float,
        budget: int,
        beta: float | None = None,
        horizon_T: int | None = None,
        rng=None,
    ) -> None:
        n_arms = len(strategies)
        if len(thetas) != n_arms:
            raise ValueError("strategies and thetas must have the same length")

        super().__init__(n_arms=n_arms, rng=rng)

        self.strategies = list(strategies)
        self.thetas = list(thetas)
        self.stake = stake
        self.budget = budget

        # q_f(t): kept uniform over S0 for this implementation (a
        # valid, simple choice satisfying q_f(t) >= 0, sum_f q_f(t) = 1
        # from Theorem 2's requirements -- NOT the linear-programming-
        # optimized choice the theorem allows, which would require
        # solving an LP at every step for marginal benefit here since
        # |S0| is small in our experiments).
        self.q = [1.0 / n_arms] * n_arms

        # Structural neighbourhoods, computed once (static for the
        # threshold family -- see class docstring).
        # N_out[f] = indices of strategies that are prefixes OF f
        #            (i.e., what ELP can infer FROM playing f)
        # N_in[g]  = indices of strategies that g is a prefix of
        #            (i.e., who can reveal information ABOUT g)
        self.N_out, self.N_in = self._build_neighbourhoods()

        # epsilon = min_{t,g} gamma(t) q_g(t). Since gamma(t) and q_g(t)
        # are both effectively constant here (q uniform, gamma derived
        # below), this reduces to a single fixed value.
        min_neighbourhood_q_sum = min(
            sum(self.q[g] for g in self.N_out[f]) for f in range(n_arms)
        )
        self._min_neighbourhood_q_sum = min_neighbourhood_q_sum

        if beta is None:
            if horizon_T is None:
                raise ValueError(
                    "Either beta or horizon_T must be provided "
                    "(horizon_T is used to compute the Corollary 7 "
                    "optimal beta)."
                )
            beta = self.corollary7_beta(
                horizon_T, n_arms, min_neighbourhood_q_sum
            )
        self.beta = beta

        # gamma(t) = beta / min_f sum_{g in N_f(t)} q_g(t). Time-
        # invariant here since the neighbourhood structure doesn't
        # change round to round for this S0.
        self.gamma = self.beta / min_neighbourhood_q_sum

        self.epsilon = self.gamma * min(self.q)

        # Weights w_f(t), initialized uniformly.
        self.w = [1.0 / n_arms] * n_arms

    # ------------------------------------------------------------------
    # Neighbourhood construction (threshold-family specific; see the
    # module and class docstrings for why this is analytic rather than
    # generic runtime comparison).
    # ------------------------------------------------------------------

    def _build_neighbourhoods(self):
        n = self.n_arms
        N_out = [set() for _ in range(n)]
        N_in = [set() for _ in range(n)]
        for f in range(n):
            for g in range(n):
                # g is a prefix of f  <=>  theta_g <= theta_f
                # (Theorem 6, Waniek et al. 2016, Section 5.3)
                if self.thetas[g] <= self.thetas[f]:
                    N_out[f].add(g)
                    N_in[g].add(f)
        return N_out, N_in

    # ------------------------------------------------------------------
    # Corollary 7 optimal beta (threshold-family case, alpha(G) = 1)
    # ------------------------------------------------------------------

    @staticmethod
    def corollary7_beta(T: int, n_arms: int, epsilon: float) -> float:
        """
        beta = sqrt( log(|S0|) / (9 T log(6|S0|/epsilon)) )

        This is the beta that balances the two terms of Theorem 6's
        regret bound (see our derivation of Corollary 3/7 -- the
        classic "balance A*beta*T against B/beta" trick from online
        learning). Matches Corollary 7 of the paper for the alpha(G)=1
        (threshold-family / clique) case.
        """
        import math

        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        numerator = math.log(n_arms)
        denominator = 9 * T * math.log(6 * n_arms / epsilon)
        return math.sqrt(numerator / denominator)

    # ------------------------------------------------------------------
    # select_arm: Langkah A + B of Algorithm 1
    # ------------------------------------------------------------------

    def select_arm(self) -> int:
        import numpy as np

        total_w = sum(self.w)
        probs = [
            (1 - self.gamma) * (self.w[g] / total_w) + self.gamma * self.q[g]
            for g in range(self.n_arms)
        ]
        # Guard against floating point drift so probs sum to exactly 1.
        probs = np.array(probs, dtype=float)
        probs = probs / probs.sum()
        self._last_probs = probs.tolist()  # stashed for update()

        arm = int(self.rng.choice(self.n_arms, p=probs))
        return arm

    # ------------------------------------------------------------------
    # update: Langkah C + D + E of Algorithm 1 -- Lemma 1 replay,
    # estimator redistribution, exponential weight update.
    # ------------------------------------------------------------------

    def update(self, arm: int, reward: float, info: dict | None = None) -> None:
        if info is None or "agent_state_trace" not in info:
            raise ValueError(
                "ELP.update() requires info={'agent_state_trace': ...} "
                "from AuctionResult.agent_state_trace -- ELP cannot "
                "apply Lemma 1 side-information without it."
            )
        trace: List[Tuple[int, int]] = info["agent_state_trace"]
        probs = self._last_probs

        # --- Langkah C: Lemma 1 replay for every g in N_out[arm] ---
        r_hat = {}
        for g in self.N_out[arm]:
            r_hat[g] = self._replay_lemma1(g, arm, reward, trace)

        # --- Langkah D: redistribute into tilde_r_g(t) for ALL g in S0 ---
        tilde_r = [0.0] * self.n_arms
        for g in range(self.n_arms):
            if arm in self.N_in[g]:
                # arm (f_t) is a strategy that g is a prefix of, so
                # playing arm DID reveal information about g via
                # Lemma 1 -- r_hat[g] was computed above (g in N_out[arm]
                # is equivalent to arm in N_in[g]).
                denom = sum(probs[h] for h in self.N_in[g])
                tilde_r[g] = r_hat[g] / denom if denom > 0 else 0.0
            else:
                tilde_r[g] = 0.0

        # --- Langkah E: exponential weight update ---
        import math

        for g in range(self.n_arms):
            self.w[g] = self.w[g] * math.exp(self.beta * tilde_r[g])

        self.t += 1

    def _replay_lemma1(
        self,
        g: int,
        played_arm: int,
        actual_reward: float,
        trace: List[Tuple[int, int]],
    ) -> float:
        """
        Implements the proof of Lemma 1 (Section 3.2): given the
        sequence of states the agent actually encountered while
        playing `played_arm`, determine what reward strategy g (a
        prefix of played_arm) would have received, WITHOUT having
        played g.

        For each state (x, y) in the trace, check whether g would have
        passed there (g(x, y) == y). If so, that is g's fold point:
        its reward is normalize(-x) (it loses, having bid x so far).
        If g never passes across the whole trace, g must have followed
        the exact same path as played_arm (guaranteed by g being a
        prefix of played_arm), so g's reward equals the actual
        observed reward.
        """
        from src.environment.dollar_auction import DollarAuction

        if g == played_arm:
            return actual_reward

        strategy_g = self.strategies[g]
        for (x, y) in trace:
            proposed = strategy_g(x, y)
            if proposed <= y:
                # g passes/folds here -> loses, having bid x so far.
                return DollarAuction.normalize_reward(-x, self.budget, self.stake)

        # g never chose to pass across the whole observed trace ->
        # it must match played_arm's actual path exactly.
        return actual_reward