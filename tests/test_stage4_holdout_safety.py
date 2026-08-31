import numpy as np
import pandas as pd
import pytest

from config import STAGE3_HOLDOUT_START_DATE
from run_stage4_risk_management import _get_oos_predictions


def _fake_close(n: int) -> pd.Series:
    index = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(0)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.011, n))), index=index)


def test_get_oos_predictions_respects_holdout_boundary():
    """Regression test for a real bug found during development: the first
    version of _get_oos_predictions built the walk-forward dataset from
    the FULL close series, letting the expanding-window folds silently
    consume data past STAGE3_HOLDOUT_START_DATE -- exactly the reserved
    period Stage 3 goes to great lengths to never touch. Fixed by
    truncating close to the dev period before building anything.
    """

    close = _fake_close(3000)

    predictions = _get_oos_predictions(close, respect_holdout=True)

    holdout_start = pd.Timestamp(STAGE3_HOLDOUT_START_DATE)
    assert predictions.index.max() < holdout_start


def test_get_oos_predictions_allows_full_series_for_synthetic():
    """The synthetic control asset has no holdout concept -- it's freshly
    generated each run, not a real historical series with a real future to
    protect. respect_holdout=False should use the full series.
    """

    close = _fake_close(3000)

    predictions = _get_oos_predictions(close, respect_holdout=False)

    # With respect_holdout=False, predictions can extend right up to
    # (near) the end of the full series, unconstrained by the holdout date
    # -- confirmed here since this index range extends well past 2024.
    holdout_start = pd.Timestamp(STAGE3_HOLDOUT_START_DATE)
    assert predictions.index.max() > holdout_start


def test_circuit_breaker_threshold_uses_global_default_without_override():
    from unittest.mock import patch

    with patch("run_stage4_risk_management.ASSET_CIRCUIT_BREAKER_OVERRIDES", {}):
        from run_stage4_risk_management import _circuit_breaker_threshold
        from config import CIRCUIT_BREAKER_DRAWDOWN_PCT

        assert _circuit_breaker_threshold("SPY") == CIRCUIT_BREAKER_DRAWDOWN_PCT


def test_circuit_breaker_threshold_uses_override_when_present():
    from unittest.mock import patch

    with patch("run_stage4_risk_management.ASSET_CIRCUIT_BREAKER_OVERRIDES", {"PETR4.SA": 0.35}):
        from run_stage4_risk_management import _circuit_breaker_threshold

        assert _circuit_breaker_threshold("PETR4.SA") == 0.35
        assert _circuit_breaker_threshold("SPY") != 0.35


def test_circuit_breaker_threshold_uses_global_default_without_override():
    from unittest.mock import patch

    with patch("run_stage4_risk_management.ASSET_CIRCUIT_BREAKER_OVERRIDES", {}):
        from run_stage4_risk_management import _circuit_breaker_threshold
        from config import CIRCUIT_BREAKER_DRAWDOWN_PCT

        assert _circuit_breaker_threshold("SPY") == CIRCUIT_BREAKER_DRAWDOWN_PCT


def test_circuit_breaker_threshold_uses_override_when_present():
    from unittest.mock import patch

    with patch("run_stage4_risk_management.ASSET_CIRCUIT_BREAKER_OVERRIDES", {"PETR4.SA": 0.35}):
        from run_stage4_risk_management import _circuit_breaker_threshold

        assert _circuit_breaker_threshold("PETR4.SA") == 0.35
        assert _circuit_breaker_threshold("SPY") != 0.35


def test_active_window_signal_stats_ignores_variation_after_active_window():
    """Regression test for a real diagnostic bug found during development:
    measuring signal variation over the FULL comparison window was
    misleading whenever the circuit breaker cuts activity short. A signal
    constant during a short active window, but highly variable afterward
    (in a period the strategy never reaches), must be reported as
    CONSTANT for that active window -- not as "varies" just because the
    full series has variation somewhere it was never actually exposed to.
    """

    from run_stage4_risk_management import _active_window_signal_stats

    # Constant 1.0 for the first 10 days, then wildly variable afterward --
    # mirrors a signal that's bullish through an early crash (before the
    # circuit breaker fires) and only becomes noisy in a later period the
    # strategy never reaches.
    signal = pd.Series([1.0] * 10 + [0.0, 1.0, 0.0, 1.0, 0.0] * 20)

    mean, std = _active_window_signal_stats(signal, active_days=10)

    assert mean == pytest.approx(1.0)
    assert std == pytest.approx(0.0)


def test_active_window_signal_stats_detects_real_variation_within_window():
    from run_stage4_risk_management import _active_window_signal_stats

    signal = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0])

    mean, std = _active_window_signal_stats(signal, active_days=5)

    assert mean == pytest.approx(0.6)
    assert std > 0.0


def test_asset_level_circuit_breaker_threshold_isolated_per_ticker():
    from unittest.mock import patch

    with patch("run_stage4_risk_management.ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES", {"PETR4.SA": 0.30}), \
         patch("run_stage4_risk_management.ASSET_LEVEL_CIRCUIT_BREAKER_PCT", None):
        from run_stage4_risk_management import _asset_level_circuit_breaker_threshold

        assert _asset_level_circuit_breaker_threshold("PETR4.SA") == 0.30
        # Other tickers stay on the global default (None = disabled) --
        # testing PETR4.SA in isolation must not silently change SPY/AAPL/
        # VALE3.SA, which are already validated.
        assert _asset_level_circuit_breaker_threshold("SPY") is None
        assert _asset_level_circuit_breaker_threshold("AAPL") is None
