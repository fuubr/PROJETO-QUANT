"""Portfolio risk metrics: historical VaR and annualized volatility.

VaR here is historical (empirical quantile), not parametric (e.g. assuming
normally-distributed returns). Consistent with this project's existing
preference for methods that don't assume a distribution shape it hasn't
verified -- the same reasoning that led to Newey-West (HAC) over a naive
t-test, and to reporting the full reliability diagram rather than trusting
a single calibration statistic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def historical_var(returns: pd.Series, confidence_level: float) -> float:
    """Historical (empirical) Value at Risk.

    Returns a POSITIVE number representing the magnitude of loss at the
    given confidence level -- e.g. historical_var(returns, 0.95) == 0.03
    means: at 95% confidence, losses should not exceed 3% (in the sample
    used, 5% of observations were worse than -3%).

    Returns 0.0 if the empirical quantile at this confidence level is
    non-negative (no loss at this confidence level, e.g. a strategy with a
    strongly positive, low-variance return stream).
    """

    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between 0 and 1")

    clean_returns = returns.dropna()
    if len(clean_returns) == 0:
        return float("nan")

    quantile = float(np.percentile(clean_returns.to_numpy(), (1.0 - confidence_level) * 100.0))
    return -quantile if quantile < 0 else 0.0


def annualized_volatility(returns: pd.Series, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized standard deviation of returns."""

    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return float("nan")

    return float(clean_returns.std(ddof=1) * np.sqrt(trading_days_per_year))


def sortino_ratio(returns: pd.Series, trading_days_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Annualized Sortino ratio: mean return over downside deviation (std
    of only negative returns), unlike Sharpe which penalizes upside
    volatility equally with downside. 0.0 if there are no negative returns
    to measure downside against (undefined penalty-free case) or fewer
    than 2 observations.
    """

    clean_returns = returns.dropna()
    if len(clean_returns) < 2:
        return float("nan")

    downside_returns = clean_returns[clean_returns < 0]
    if len(downside_returns) < 2:
        return 0.0

    downside_deviation = downside_returns.std(ddof=1)
    if downside_deviation == 0:
        return 0.0

    return float(clean_returns.mean() / downside_deviation * np.sqrt(trading_days_per_year))
