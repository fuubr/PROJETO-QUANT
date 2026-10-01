import numpy as np
import pandas as pd
import pytest

from paper_trading.drift import block_bootstrap_mean_ci, feature_drift, performance_drift


def _ar1(n, phi, seed):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.normal()
    return x


def test_feature_drift_insufficient_live_data_returns_none():
    ref = pd.DataFrame({"f": _ar1(1000, 0.9, 0)})
    assert feature_drift(ref, ref.iloc[:5], alert_p=0.05, min_live_days=10) is None


def test_feature_drift_detects_a_large_level_shift():
    ref = pd.DataFrame({"f": _ar1(1500, 0.9, 1)})
    live = pd.DataFrame({"f": _ar1(25, 0.9, 2) + 8.0})  # shifted far outside anything seen
    result = feature_drift(ref, live, alert_p=0.05, min_live_days=10)
    assert result[0].alert and result[0].empirical_p < 0.05


def test_feature_drift_false_alarm_rate_is_controlled_on_autocorrelated_data():
    """The reason for the empirical rolling-window null: with strongly
    autocorrelated features a naive test would alarm far more often than
    5%. Live windows drawn from the SAME process must alarm roughly 5%.
    """

    alarms, trials = 0, 150
    for seed in range(trials):
        series = _ar1(1500 + 25, 0.95, 100 + seed)
        ref = pd.DataFrame({"f": series[:1500]})
        live = pd.DataFrame({"f": series[1500:]})
        alarms += feature_drift(ref, live, 0.05, 10)[0].alert
    assert alarms / trials < 0.15  # nominal 5%; generous bound for 150 trials


def test_block_bootstrap_ci_contains_true_mean_for_iid_noise():
    rng = np.random.default_rng(3)
    values = rng.normal(0.5, 1.0, 400)
    mean, lo, hi = block_bootstrap_mean_ci(values, 20, 1000, 0.05, 42)
    assert lo < 0.5 < hi and lo < mean < hi


def test_performance_drift_is_insufficient_without_enough_effective_samples():
    idx = pd.bdate_range("2026-01-01", periods=40)  # 40 days / 20 = 2 effective < 6
    p = pd.Series(0.9, index=idx)
    y = pd.Series(0.0, index=idx)  # terrible predictions on purpose
    r = performance_drift(p, y, 0.55, 20, 6, 500, 0.05, 42)
    assert r.verdict == "INSUFICIENTE"  # even a disastrous run gets no verdict yet
    assert r.mean_skill < 0  # but the number is still reported


def test_performance_drift_flags_a_model_clearly_worse_than_base_rate():
    idx = pd.bdate_range("2026-01-01", periods=400)
    rng = np.random.default_rng(4)
    y = pd.Series(rng.integers(0, 2, 400).astype(float), index=idx)
    p = pd.Series(np.where(y == 1, 0.1, 0.9), index=idx)  # confidently WRONG every day
    r = performance_drift(p, y, 0.5, 20, 6, 500, 0.05, 42)
    assert r.verdict == "DEGRADADO" and r.ci_high < 0


def test_performance_drift_ok_for_a_genuinely_informative_model():
    idx = pd.bdate_range("2026-01-01", periods=400)
    rng = np.random.default_rng(5)
    y = pd.Series(rng.integers(0, 2, 400).astype(float), index=idx)
    p = pd.Series(np.where(y == 1, 0.8, 0.2), index=idx)
    r = performance_drift(p, y, 0.5, 20, 6, 500, 0.05, 42)
    assert r.verdict == "OK" and r.mean_skill > 0


def test_performance_drift_no_data():
    empty = pd.Series(dtype=float)
    assert performance_drift(empty, empty, 0.5, 20, 6, 100, 0.05, 1).verdict == "INSUFICIENTE"


def test_bootstrap_ci_is_unavailable_with_fewer_than_two_independent_blocks():
    """Regression for a real defect found while building the report: with
    one block the bootstrap collapsed to a single point and printed a
    zero-width CI that looked precise. It must say 'unavailable' instead.
    """

    mean, lo, hi = block_bootstrap_mean_ci(np.arange(25, dtype=float), 20, 500, 0.05, 1)
    assert mean == pytest.approx(12.0) and np.isnan(lo) and np.isnan(hi)
