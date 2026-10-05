"""
page_hinkley.py

Page-Hinkley (PH) test for online changepoint detection.

DIRECTION FIX (Phase 1 bug #1)
------------------------------
The previous version accumulated

    U_t = sum (X_tau - mu_tau - delta)

and flagged U_t - m_t > lambda. That statistic grows when X rises ABOVE
its reference mean, i.e. it detects an INCREASE. In this project the
monitored signal X_t is Alice's realized (normalized) reward, and when
Bob switches to the escalation regime her reward DROPS. The correct
statistic for a drop is

    U_t = sum_{tau=1}^{t} (mu_tau - X_tau - delta)
    m_t = min_{tau <= t} U_tau
    alarm  <=>  U_t - m_t > lambda

Reading it: while X stays near mu, each term is about -delta (negative),
so U_t drifts down and m_t follows it; the gap U_t - m_t stays ~0. After
a downward shift, X < mu - delta, each term is positive, U_t climbs away
from its running minimum, and the gap grows until it crosses lambda.

`direction` selects what is monitored:
    "decrease" (default) : drop in the signal   (mu - X - delta)
    "increase"           : rise in the signal   (X - mu - delta)
    "both"               : two-sided; alarm if either side fires

Stage 1b note: when Bob returns to the rational regime, Alice's reward
RISES, and a "decrease" detector will not fire. That is intended for the
primary EXP3.S + PH reset trigger (EXP3.S's own mixing handles recovery),
but use direction="both" if you want to log recoveries as changepoints.

FALSE-ALARM vs DELAY (unchanged, still true)
--------------------------------------------
No single (delta, threshold) pair gives both zero false alarms and zero
delay; U_t is a biased random walk and will eventually cross any fixed
threshold on a long stable stretch. Calibrate `threshold` on a reference
signal known to contain no changepoint (`calibrate_threshold`), per
experiment configuration, instead of reusing a default.

REFERENCE MEAN
--------------
mean_alpha=None -> classic cumulative mean (anchored to old regime; slow
after long regimes). mean_alpha in (0,1] -> exponential moving average
(default 0.05, ~20-round memory). With an EMA reference the total
evidence a step of size d can accumulate is about d / alpha, so
`threshold` must be well below (min shift of interest) / alpha.

CAVEAT: the monitored reward depends on Alice's own arm distribution,
which changes while she learns. Calibrate on data from the same algorithm
in a stable regime (e.g. the same learner against Alice-rational), not on
a synthetic i.i.d. stream.

This module is standalone: it only consumes a stream of scalars.
"""

from __future__ import annotations

from typing import List, Optional

_DIRECTIONS = ("decrease", "increase", "both")


class PageHinkleyDetector:
    """
    Parameters
    ----------
    delta : float, default 0.005
        Drift tolerance.
    threshold : float, default 5.0
        Detection threshold (lambda). Calibrate per configuration.
    mean_alpha : float or None, default 0.05
        EMA weight for the reference mean; None = cumulative mean.
    auto_reset : bool, default True
        Reset running statistics right after an alarm so the next
        changepoint (Stage 1b) can be detected.
    direction : {"decrease", "increase", "both"}, default "decrease"
        Which kind of shift to detect (see module docstring).
    warmup : int, default 20
        Number of observations (after construction and after every
        reset) used only to establish the reference mean (plain average,
        no evidence accumulated, no alarms). Without this, an EMA seeded
        from a single noisy first sample can fire a false alarm within the
        first few rounds. Set 0 to disable.
    """

    def __init__(
        self,
        delta: float = 0.005,
        threshold: float = 5.0,
        mean_alpha: Optional[float] = 0.05,
        auto_reset: bool = True,
        direction: str = "decrease",
        warmup: int = 20,
    ) -> None:
        if warmup < 0:
            raise ValueError("warmup must be >= 0")
        if threshold <= 0:
            raise ValueError("threshold (lambda) must be positive")
        if mean_alpha is not None and not (0 < mean_alpha <= 1):
            raise ValueError("mean_alpha must be None or in (0, 1]")
        if direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}")
        self.delta = delta
        self.threshold = threshold
        self.mean_alpha = mean_alpha
        self.auto_reset = auto_reset
        self.direction = direction
        self.warmup = warmup

        self.detection_rounds: List[int] = []
        self.detection_directions: List[str] = []
        self._reset_state()

    def _reset_state(self) -> None:
        self._t = 0
        self._running_mean = 0.0
        # decrease side
        self._U_dec = 0.0
        self._m_dec = 0.0
        # increase side
        self._U_inc = 0.0
        self._m_inc = 0.0

    def update(self, x: float, global_round: Optional[int] = None) -> bool:
        """Feed one observation; return True if a changepoint is flagged."""
        self._t += 1

        if self._t <= self.warmup:
            # warm-up: reference = plain average of the first `warmup`
            # observations; accumulate no evidence, raise no alarm.
            self._running_mean += (x - self._running_mean) / self._t
            return False

        if self.mean_alpha is None:
            self._running_mean += (x - self._running_mean) / self._t
        elif self._t == 1:
            self._running_mean = x
        else:
            self._running_mean += self.mean_alpha * (x - self._running_mean)

        mu = self._running_mean
        fired = None

        if self.direction in ("decrease", "both"):
            self._U_dec += (mu - x) - self.delta          # <-- the fix
            self._m_dec = min(self._m_dec, self._U_dec)
            if (self._U_dec - self._m_dec) > self.threshold:
                fired = "decrease"

        if self.direction in ("increase", "both"):
            self._U_inc += (x - mu) - self.delta
            self._m_inc = min(self._m_inc, self._U_inc)
            if fired is None and (self._U_inc - self._m_inc) > self.threshold:
                fired = "increase"

        if fired is not None:
            self.detection_rounds.append(
                global_round if global_round is not None else self._t
            )
            self.detection_directions.append(fired)
            if self.auto_reset:
                self._reset_state()
            return True
        return False

    def reset(self) -> None:
        """Reset running statistics; detection history is kept."""
        self._reset_state()

    @property
    def current_U(self) -> float:
        """U_t of the primary monitored side."""
        return self._U_inc if self.direction == "increase" else self._U_dec

    @property
    def current_gap(self) -> float:
        """Gap compared against `threshold` (max over monitored sides)."""
        gaps = []
        if self.direction in ("decrease", "both"):
            gaps.append(self._U_dec - self._m_dec)
        if self.direction in ("increase", "both"):
            gaps.append(self._U_inc - self._m_inc)
        return max(gaps)

    @staticmethod
    def calibrate_threshold(
        stable_reference_signal: List[float],
        delta: float,
        mean_alpha: Optional[float],
        target_false_alarms: int = 0,
        candidate_thresholds: Optional[List[float]] = None,
        direction: str = "decrease",
        warmup: int = 20,
    ) -> float:
        """
        Smallest threshold keeping false alarms on a known-stable signal
        at or below `target_false_alarms` (ARL0-style calibration).
        Raises ValueError if no candidate qualifies.
        """
        if candidate_thresholds is None:
            candidate_thresholds = [
                0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8,
                1.0, 1.5, 2.0, 3.0, 5.0,
            ]
        for threshold in sorted(candidate_thresholds):
            det = PageHinkleyDetector(
                delta=delta, threshold=threshold, mean_alpha=mean_alpha,
                auto_reset=True, direction=direction,
                warmup=warmup,
            )
            n_false = sum(1 for x in stable_reference_signal if det.update(x))
            if n_false <= target_false_alarms:
                return threshold
        raise ValueError(
            f"No candidate threshold in {candidate_thresholds} kept false "
            f"alarms <= {target_false_alarms}. Try a wider candidate list "
            "or a larger delta."
        )