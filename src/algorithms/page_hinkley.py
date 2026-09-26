"""
page_hinkley.py

Page-Hinkley (PH) test for online changepoint detection, following the
formulation used throughout this project (see docs/notation.md and
our derivation of the "d-BayesMAB" critique):

    U_t = sum_{tau=1}^{t} (X_tau - mu_tau - delta)
    m_t = min_{1 <= tau <= t} U_tau

A changepoint is flagged the moment:

    U_t - m_t > lambda

where:
    X_t     : the monitored performance signal at round t (here, the
              reward the agent actually received that round)
    mu_t    : the running mean of X up to and including round t
    delta   : small allowable drift tolerance (prevents false alarms
              from ordinary noise)
    lambda  : detection threshold (higher = less sensitive, fewer
              false alarms, but slower/larger detection delay)

Intuition: U_t only grows when X_t drops meaningfully BELOW its own
historical running mean (beyond the delta tolerance). m_t tracks the
lowest (most negative) U_t has ever been. The gap (U_t - m_t) measures
how far the signal has climbed back up from its historical low point
-- a sustained climb means the signal was consistently below its old
mean for a while and has now "used up" that slack, which is exactly
the signature of a genuine downward mean-shift (e.g. Bob's regime
flipping from rational to escalating, dragging the agent's achievable
reward down).

This module is intentionally standalone: it does not know anything
about bandits, arms, or the Dollar Auction. It only consumes a stream
of scalar values. This is what lets it be "plugged into" EXP3.S later
(see exp3s.py) as an independent reset trigger.
"""

from __future__ import annotations

from typing import List


class PageHinkleyDetector:
    """
    Parameters
    ----------
    delta : float, default 0.005
        Drift tolerance (see module docstring). Small positive value;
        larger delta makes the detector more tolerant of gradual
        drift before accumulating evidence of a change.
    threshold : float, default 5.0
        Detection threshold (lambda). Must be tuned against the scale
        of the monitored signal X_t -- for X_t in [0, 1] (as our
        normalized rewards are), values roughly in the 1-10 range are
        typical starting points; always validate empirically via
        detection-delay experiments (src/metrics/detection_delay.py)
        rather than assuming a default is correct for a new setting.
    mean_alpha : float or None, default 0.05
        Controls how mu_t (the reference mean subtracted in U_t) is
        computed:
            - If None: mu_t is the classic UNWEIGHTED cumulative mean
              over the ENTIRE history since the last reset (mu_t =
              (1/t) * sum of all X seen so far). This is the textbook
              formula, but it has a serious practical flaw we found by
              testing this module: if a regime has run for a long time
              before switching (e.g. Bob's single-switch at t=1000),
              mu_t stays anchored near the OLD regime's mean for a very
              long stretch after the switch, because it is an average
              over hundreds/thousands of old points versus only a few
              new ones. This causes U_t to keep hitting new lows every
              round for a long time post-switch (since X_t - mu_t stays
              strongly negative), which keeps m_t tracking U_t exactly
              and the detection gap stuck at ~0 -- i.e., detection is
              delayed by roughly as many rounds as the PRIOR regime
              lasted. This is the exact same "historical rigidity"
              failure mode we diagnosed in UCB/Thompson Sampling,
              except here it cripples the changepoint detector itself.
            - If a float in (0, 1]: mu_t is instead an EXPONENTIAL
              MOVING AVERAGE, mu_t = mu_{t-1} + alpha*(X_t - mu_{t-1}),
              which forgets old observations at a controlled, BOUNDED
              rate regardless of how long the previous regime lasted.
              This is the setting actually used by default here, since
              the unweighted option is impractical for the horizons
              (T in the thousands) used in this thesis. Smaller alpha
              = longer effective memory = slower to adapt but less
              noisy; larger alpha = shorter memory = faster detection
              but noisier mu_t. alpha=0.05 (~20-round effective memory)
              is a reasonable starting point; tune empirically.
    auto_reset : bool, default True
        If True, the detector automatically resets its internal
        running statistics immediately after flagging a changepoint,
        so it is ready to detect the NEXT changepoint (needed for
        Stage 1b's recurring-switch scenario, where multiple
        changepoints occur over the horizon). If False, the caller is
        responsible for calling `reset()` manually after acting on a
        detected changepoint.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 5.0,
        mean_alpha: float | None = 0.05,
        auto_reset: bool = True,
    ) -> None:
        if threshold <= 0:
            raise ValueError("threshold (lambda) must be positive")
        if mean_alpha is not None and not (0 < mean_alpha <= 1):
            raise ValueError("mean_alpha must be None or in (0, 1]")
        self.delta = delta
        self.threshold = threshold
        self.mean_alpha = mean_alpha
        self.auto_reset = auto_reset

        self.detection_rounds: List[int] = []  # global round index of
                                                 # every changepoint ever
                                                 # flagged (for later
                                                 # detection-delay analysis)
        self._reset_state()

    def _reset_state(self) -> None:
        self._t = 0            # local round counter since last reset
        self._running_mean = 0.0
        self._U = 0.0
        self._m = 0.0

    def update(self, x: float, global_round: int | None = None) -> bool:
        """
        Feed one new observation X_t into the detector.

        Parameters
        ----------
        x : float
            The observed value this round (e.g. the agent's reward).
        global_round : int, optional
            The round index in the OUTER experiment (not the
            detector's own internal counter, which resets after every
            detected changepoint). If given, it is what gets recorded
            in `self.detection_rounds` -- pass this so the detection
            history lines up with the experiment's actual round
            numbering rather than a counter that keeps restarting from
            zero. If omitted, the internal counter is recorded instead.

        Returns
        -------
        bool
            True if a changepoint was flagged THIS round.
        """
        self._t += 1

        if self.mean_alpha is None:
            # Classic unweighted cumulative mean (see mean_alpha
            # docstring above for why this is impractical at the
            # horizons used in this thesis -- kept only for
            # completeness/comparison).
            self._running_mean += (x - self._running_mean) / self._t
        else:
            if self._t == 1:
                self._running_mean = x
            else:
                self._running_mean += self.mean_alpha * (x - self._running_mean)

        self._U += x - self._running_mean - self.delta
        self._m = min(self._m, self._U)

        detected = (self._U - self._m) > self.threshold

        if detected:
            record = global_round if global_round is not None else self._t
            self.detection_rounds.append(record)
            if self.auto_reset:
                self._reset_state()

        return detected

    def reset(self) -> None:
        """Manually reset internal running statistics (U_t, m_t, and
        the running mean) without clearing the detection history."""
        self._reset_state()

    @property
    def current_U(self) -> float:
        return self._U

    @property
    def current_gap(self) -> float:
        """U_t - m_t, the quantity compared against `threshold`."""
        return self._U - self._m