"""Backtest-vs-paper-trading consistency check.

This is Stage 6's core metric per the original spec: does live paper
trading match what a normal backtest would have predicted for the same
dates, using the same frozen model? Since paper trading's daily script and
the batch backtest engine both ultimately call the SAME step function
(risk.managed_backtest.step_risk_managed_backtest), they should produce
numerically identical results when fed the same inputs -- any real
divergence found here points to an actual bug (a data or timing
difference between the two code paths), not "the market changed", which
is exactly the kind of thing this check is meant to catch early rather
than silently accumulate for months.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from config import (
    KELLY_FRACTION,
    KELLY_MIN_LOSS_OBSERVATIONS,
    KELLY_MIN_WIN_OBSERVATIONS,
    KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
    PAPER_TRADING_INITIAL_CAPITAL,
    PREDICTION_HORIZON_DAYS,
)
from paper_trading.predictions import versioned_probabilities
from risk.kelly import kelly_position_series
from paper_trading.state import state_from_last_row
from risk.managed_backtest import run_risk_managed_backtest, step_risk_managed_backtest


@dataclass
class ConsistencyCheckResult:
    ticker: str
    days_compared: int
    max_absolute_equity_diff: float
    max_relative_equity_diff: float
    matches: bool  # True if max_relative_equity_diff is within tolerance
    max_target_diff: float = 0.0  # max |logged target - recomputed target| (clean-period mode)


def check_consistency(
    ticker: str,
    close: pd.Series,
    versions,
    log: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    circuit_breaker_pct: float,
    asset_circuit_breaker_pct: float | None,
    stop_loss_pct: float,
    var_confidence_level: float,
    relative_tolerance: float = 1e-6,
    clean_after: pd.Timestamp | None = None,
) -> ConsistencyCheckResult:
    """Re-run the exact same period the paper trading log covers through
    the BATCH engine, using the same frozen model, and compare equity
    curves day by day.
    """

    probabilities = versioned_probabilities(close, versions)
    if clean_after is not None:
        return _check_since_clean(
            ticker, close, probabilities, log, fee_bps, slippage_bps, circuit_breaker_pct,
            asset_circuit_breaker_pct, stop_loss_pct, relative_tolerance, clean_after,
        )

    kelly_positions = kelly_position_series(
        close, probabilities, horizon_days=PREDICTION_HORIZON_DAYS, kelly_fraction_multiplier=KELLY_FRACTION,
        win_loss_window_days=KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS, min_win_observations=KELLY_MIN_WIN_OBSERVATIONS,
        min_loss_observations=KELLY_MIN_LOSS_OBSERVATIONS,
    )

    log_dates = log.index
    close_over_log_range = close.loc[log_dates]
    target_over_log_range = pd.Series(0.0, index=log_dates)
    target_over_log_range.update(kelly_positions.reindex(log_dates))

    batch_result = run_risk_managed_backtest(
        close_over_log_range, target_over_log_range, fee_bps=fee_bps, slippage_bps=slippage_bps,
        initial_capital=PAPER_TRADING_INITIAL_CAPITAL, stop_loss_pct=stop_loss_pct,
        circuit_breaker_drawdown_pct=circuit_breaker_pct, var_confidence_level=var_confidence_level,
        asset_circuit_breaker_drawdown_pct=asset_circuit_breaker_pct,
    )

    diff = (batch_result.equity_curve - log["equity"]).abs()
    relative_diff = diff / log["equity"]

    max_absolute_diff = float(diff.max())
    max_relative_diff = float(relative_diff.max())

    return ConsistencyCheckResult(
        ticker=ticker,
        days_compared=len(log_dates),
        max_absolute_equity_diff=max_absolute_diff,
        max_relative_equity_diff=max_relative_diff,
        matches=max_relative_diff < relative_tolerance,
    )


def _check_since_clean(
    ticker, close, probabilities, log, fee_bps, slippage_bps, circuit_breaker_pct,
    asset_circuit_breaker_pct, stop_loss_pct, relative_tolerance, clean_after,
) -> ConsistencyCheckResult:
    """Consistency over the period produced by corrected code only.

    The first row after clean_after is the first whose TARGET was computed
    correctly, so its target drives the next day's return correctly. We
    seed the risk state from that logged row (history before it is already
    realized and is not re-litigated) and replay every later day with the
    step function using targets RECOMPUTED from the frozen model, comparing
    equity day by day and the logged targets directly. A bug that predates
    clean_after therefore cannot poison the comparison, and a new bug is
    localized to the day it appears.
    """

    clean = log[log.index > clean_after]
    if len(clean) < 2:  # need a seed row plus at least one day to compare
        return ConsistencyCheckResult(ticker, 0, 0.0, 0.0, True, 0.0)

    kelly_positions = kelly_position_series(
        close, probabilities, horizon_days=PREDICTION_HORIZON_DAYS, kelly_fraction_multiplier=KELLY_FRACTION,
        win_loss_window_days=KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS, min_win_observations=KELLY_MIN_WIN_OBSERVATIONS,
        min_loss_observations=KELLY_MIN_LOSS_OBSERVATIONS,
    )

    state, _, _ = state_from_last_row(clean.iloc[:1])
    max_abs = max_rel = max_target = 0.0
    for i in range(1, len(clean)):
        prev_date, date = clean.index[i - 1], clean.index[i]
        target_prev = float(kelly_positions.get(prev_date, 0.0))
        state, _ = step_risk_managed_backtest(
            state, price_yesterday=float(close.loc[prev_date]), price_today=float(close.loc[date]),
            target_position_yesterday=target_prev, fee_bps=fee_bps, slippage_bps=slippage_bps,
            stop_loss_pct=stop_loss_pct, circuit_breaker_drawdown_pct=circuit_breaker_pct, today_date=date,
            asset_circuit_breaker_drawdown_pct=asset_circuit_breaker_pct,
        )
        logged_equity = float(clean.iloc[i]["equity"])
        diff = abs(state.equity - logged_equity)
        max_abs, max_rel = max(max_abs, diff), max(max_rel, diff / logged_equity)
        max_target = max(max_target, abs(target_prev - float(clean.iloc[i - 1]["target_position"])))

    ok = max_rel < relative_tolerance and max_target < 1e-6
    return ConsistencyCheckResult(ticker, len(clean) - 1, max_abs, max_rel, ok, max_target)
