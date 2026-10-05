import numpy as np

from src.algorithms.page_hinkley import PageHinkleyDetector

SWITCH = 500


def _signal(seed, before, after, n_before=SWITCH, n_after=500, sd=0.08):
    rng = np.random.default_rng(seed)
    x = np.concatenate([
        rng.normal(before, sd, n_before),
        rng.normal(after, sd, n_after),
    ])
    return np.clip(x, 0.0, 1.0)


def _first_detection(det, x):
    for t, v in enumerate(x, start=1):
        if det.update(v, global_round=t):
            return t
    return None


def _calibrated(direction="decrease", delta=0.02, alpha=0.05):
    ref = _signal(seed=999, before=0.7, after=0.7, n_before=1500, n_after=1500)
    thr = PageHinkleyDetector.calibrate_threshold(
        ref, delta=delta, mean_alpha=alpha, direction=direction
    )
    return PageHinkleyDetector(delta=delta, threshold=thr, mean_alpha=alpha,
                               direction=direction), thr


def test_detects_drop_after_switch_not_before():
    for seed in range(10):
        det, _ = _calibrated("decrease")
        x = _signal(seed, before=0.7, after=0.4)
        t = _first_detection(det, x)
        assert t is not None, f"seed {seed}: drop never detected"
        assert t > SWITCH, f"seed {seed}: false alarm at t={t}"
        assert t - SWITCH < 100, f"seed {seed}: delay {t - SWITCH} too large"


def test_rise_is_ignored_by_decrease_detector():
    det, _ = _calibrated("decrease")
    x = _signal(seed=1, before=0.4, after=0.8)
    assert _first_detection(det, x) is None


def test_rise_is_detected_by_increase_and_both():
    for direction in ("increase", "both"):
        det, _ = _calibrated(direction)
        x = _signal(seed=1, before=0.4, after=0.8)
        t = _first_detection(det, x)
        assert t is not None and t > SWITCH


def test_auto_reset_allows_second_detection():
    det, _ = _calibrated("decrease")
    # drop, recover, drop again
    x = np.concatenate([
        _signal(3, 0.7, 0.4, 400, 400),
        _signal(4, 0.7, 0.4, 400, 400),
    ])
    for t, v in enumerate(x, start=1):
        det.update(v, global_round=t)
    assert len(det.detection_rounds) >= 2