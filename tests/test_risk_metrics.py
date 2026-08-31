import numpy as np
import pandas as pd
import pytest

from risk.metrics import annualized_volatility, historical_var, sortino_ratio


def test_historical_var_matches_known_percentile():
    # 100 evenly-spaced returns from -0.10 to +0.09 (deterministic).
    # 5th percentile (for 95% confidence) should be near -0.095.
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))
    var_95 = historical_var(returns, confidence_level=0.95)
    assert var_95 == pytest.approx(0.095, abs=0.005)


def test_historical_var_is_zero_when_no_losses_at_confidence_level():
    returns = pd.Series(np.linspace(0.01, 0.10, 100))  # all positive
    var_95 = historical_var(returns, confidence_level=0.95)
    assert var_95 == pytest.approx(0.0)


def test_historical_var_rejects_invalid_confidence():
    returns = pd.Series([0.01, -0.01])
    with pytest.raises(ValueError, match="confidence_level"):
        historical_var(returns, confidence_level=1.5)


def test_historical_var_handles_empty_series():
    assert np.isnan(historical_var(pd.Series(dtype=float), confidence_level=0.95))


def test_annualized_volatility_matches_manual_calculation():
    rng = np.random.default_rng(0)
    daily_returns = pd.Series(rng.normal(0.0, 0.01, 5000))
    expected = daily_returns.std(ddof=1) * np.sqrt(252)
    assert annualized_volatility(daily_returns) == pytest.approx(expected)


def test_annualized_volatility_handles_insufficient_data():
    assert np.isnan(annualized_volatility(pd.Series([0.01])))


def test_sortino_ratio_ignores_upside_volatility():
    returns = pd.Series([0.05, 0.001, 0.10, 0.001, 0.08])
    ratio = sortino_ratio(returns)
    assert ratio == 0.0


def test_sortino_ratio_matches_manual_calculation():
    rng = np.random.default_rng(0)
    returns = pd.Series(rng.normal(0.0005, 0.01, 2000))
    downside = returns[returns < 0]
    expected = returns.mean() / downside.std(ddof=1) * np.sqrt(252)
    assert sortino_ratio(returns) == pytest.approx(expected)


def test_sortino_ratio_handles_insufficient_data():
    assert np.isnan(sortino_ratio(pd.Series([0.01])))
