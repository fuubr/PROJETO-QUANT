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
    FEATURE_MOMENTUM_WINDOWS,
    FEATURE_SMA_DISTANCE_WINDOW,
    FEATURE_VOLATILITY_WINDOW,
    KELLY_FRACTION,
    KELLY_MIN_LOSS_OBSERVATIONS,
    KELLY_MIN_WIN_OBSERVATIONS,
    KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
    PAPER_TRADING_INITIAL_CAPITAL,
    PREDICTION_HORIZON_DAYS,
)
from models.dataset import build_dataset, feature_columns
from risk.kelly import kelly_position_series
from risk.managed_backtest import run_risk_managed_backtest


@dataclass
class ConsistencyCheckResult:
    ticker: str
    days_compared: int
    max_absolute_equity_diff: float
    max_relative_equity_diff: float
    matches: bool  # True if max_relative_equity_diff is within tolerance


def _frozen_model_probabilities(close: pd.Series, trained) -> pd.Series:
    """Probabilities from the frozen model for every date with enough
    trailing history, over the FULL close series -- same feature
    construction as run_stage6_paper_trading_daily._today_target_position,
    just evaluated for every date at once instead of one at a time.
    """

    dataset_features = build_dataset(
        close, momentum_windows=FEATURE_MOMENTUM_WINDOWS, volatility_window=FEATURE_VOLATILITY_WINDOW,
        sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW, horizon_days=1,
    )
    feature_cols = feature_columns(dataset_features)
    probabilities = trained.xgboost_calibrated.predict_proba(dataset_features[feature_cols])[:, 1]
    return pd.Series(probabilities, index=dataset_features.index)


def check_consistency(
    ticker: str,
    close: pd.Series,
    trained,
    log: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float,
    circuit_breaker_pct: float,
    asset_circuit_breaker_pct: float | None,
    stop_loss_pct: float,
    var_confidence_level: float,
    relative_tolerance: float = 1e-6,
) -> ConsistencyCheckResult:
    """Re-run the exact same period the paper trading log covers through
    the BATCH engine, using the same frozen model, and compare equity
    curves day by day.
    """

    probabilities = _frozen_model_probabilities(close, trained)

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
