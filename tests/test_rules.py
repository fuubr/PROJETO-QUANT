import numpy as np
import pandas as pd
import pytest

from strategies.rules import trend_vol_target_position, vol_target_position


def _alternating(n, step):
    """Prices that alternate +step / -step: daily returns have a known std."""

    returns = np.array([step if i % 2 == 0 else -step for i in range(n)])
    return pd.Series(100 * np.cumprod(1 + returns), index=pd.bdate_range("2020-01-01", periods=n))


def test_vol_target_scales_exposure_by_target_over_realized_vol():
    # Alternating +/-1% days: sample std of 20 such returns = 0.01 * sqrt(20/19).
    # Annualized: 0.01 * sqrt(20/19) * sqrt(252) = 0.16287 -> 0.10 / 0.16287 = 0.6140
    close = _alternating(60, 0.01)
    assert vol_target_position(close, target=0.10) == pytest.approx(0.6140, abs=1e-3)


def test_vol_target_caps_at_full_exposure_when_vol_is_low():
    close = _alternating(60, 0.001)  # ~1.6% annualized vol -> raw 6.2x, capped
    assert vol_target_position(close, target=0.10) == 1.0


def test_vol_target_is_flat_without_enough_history():
    assert vol_target_position(_alternating(10, 0.01), target=0.10) == 0.0


def _trend(n, daily_drift):
    """Smooth trend plus tiny alternation so realized vol is defined."""

    noise = np.array([0.001 if i % 2 == 0 else -0.001 for i in range(n)])
    return pd.Series(100 * np.cumprod(1 + daily_drift + noise), index=pd.bdate_range("2019-01-01", periods=n))


def test_trend_filter_keeps_vol_target_exposure_in_an_uptrend():
    close = _trend(300, 0.001)  # up over the last 252 days
    assert trend_vol_target_position(close, target=0.10) == vol_target_position(close, target=0.10) > 0


def test_trend_filter_goes_to_cash_in_a_downtrend():
    assert trend_vol_target_position(_trend(300, -0.001), target=0.10) == 0.0


def test_trend_filter_is_flat_without_a_full_year_of_history():
    assert trend_vol_target_position(_trend(200, 0.001), target=0.10) == 0.0


def test_rules_ignore_prices_after_the_decision_date():
    close = _trend(300, 0.001)
    t = close.index[280]
    future_crash = close.copy()
    future_crash.iloc[281:] *= 0.5
    for rule in (vol_target_position, trend_vol_target_position):
        assert rule(close.loc[:t], 0.10) == rule(future_crash.loc[:t], 0.10)
