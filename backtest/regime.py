"""Regime segmentation: slices ALREADY-REALIZED backtest returns by period
(bull vs named crisis windows), for separate reporting metrics.

Deliberately does NOT re-run the sequential risk-managed backtest on a
filtered/discontinuous date range -- stop-loss entry-price tracking and
circuit-breaker drawdown tracking both assume a continuous day-by-day
series. Slicing the REALIZED returns after a single continuous run is the
correct way to get regime-specific metrics without breaking that
continuity assumption.
"""

from __future__ import annotations

import pandas as pd

from risk.metrics import TRADING_DAYS_PER_YEAR, annualized_volatility, historical_var, sortino_ratio


def segment_returns_by_regime(
    net_returns: pd.Series, crisis_periods: list[tuple[str, str]]
) -> dict[str, pd.Series]:
    """Split a realized net_returns series into named segments: one per
    crisis_periods entry (labeled crisis_1, crisis_2, ...) plus "bull" for
    everything else in net_returns' own date range.

    Segments with zero observations (e.g. a crisis window entirely outside
    this particular return series' date range, such as the synthetic
    control, or a period the strategy never reached before a circuit
    breaker) are included as empty series, not silently dropped -- callers
    can see zero-coverage explicitly rather than a shorter, unexplained
    segment list.
    """

    crisis_mask = pd.Series(False, index=net_returns.index)
    segments: dict[str, pd.Series] = {}

    for i, (start, end) in enumerate(crisis_periods, start=1):
        window_mask = (net_returns.index >= pd.Timestamp(start)) & (net_returns.index <= pd.Timestamp(end))
        segments[f"crisis_{i}_{start[:7]}"] = net_returns[window_mask]
        crisis_mask |= window_mask

    segments["bull"] = net_returns[~crisis_mask]
    return segments


def summarize_regime_segment(returns: pd.Series) -> dict:
    """Metrics for one regime segment: compounded total return, annualized
    Sharpe/Sortino, VaR, volatility. Returns NaN/0 metrics (not an error)
    for empty or near-empty segments -- an empty segment is valid
    information (this strategy/asset had zero exposure to this regime
    window), not a failure.
    """

    if len(returns) == 0:
        return {
            "total_return": float("nan"),
            "annualized_sharpe": float("nan"),
            "sortino_ratio": float("nan"),
            "historical_var_95": float("nan"),
            "annualized_volatility": float("nan"),
            "days": 0,
        }

    total_return = float((1.0 + returns).prod() - 1.0)
    mean_return = returns.mean()
    std_return = returns.std(ddof=1) if len(returns) > 1 else 0.0
    sharpe = float(mean_return / std_return * (TRADING_DAYS_PER_YEAR ** 0.5)) if std_return > 0 else 0.0

    return {
        "total_return": round(total_return, 4),
        "annualized_sharpe": round(sharpe, 3),
        "sortino_ratio": round(sortino_ratio(returns), 3),
        "historical_var_95": round(historical_var(returns, 0.95), 4),
        "annualized_volatility": round(annualized_volatility(returns), 4),
        "days": len(returns),
    }
