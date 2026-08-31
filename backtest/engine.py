"""Vectorized backtest engine.

Design principle: the signal on day t represents a position decided using
information available up to and including the close of day t. That position
is only allowed to earn the return of day t+1. This is enforced with an
explicit shift, so lookahead bias is structurally prevented rather than
relying on the caller to get the timing right.

No predictive logic lives here. This module only simulates PnL given a
position series that some other component (a baseline, later a model)
produces.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    net_returns: pd.Series
    total_return: float
    annualized_sharpe: float
    max_drawdown: float
    win_rate: float
    total_costs_paid: float
    """Cumulative dollar cost paid, weighted by equity at the time of each
    trade. CAUTION when comparing this figure across strategies/signals:
    since cost in dollars scales with the equity available to charge a fee
    against, a strategy whose compounding erodes its own capital toward zero
    will show artificially LOW total dollar costs late in its life, purely
    because there is little capital left -- not because trading became
    cheap. For cross-strategy cost comparisons, prefer a rate-based measure
    (e.g. mean per-day cost as a fraction of returns) over this dollar
    total, especially over long horizons with compounding.
    """


def run_backtest(
    close: pd.Series,
    signal: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    initial_capital: float,
) -> BacktestResult:
    """Simulate PnL for a given position signal against close prices.

    Parameters
    ----------
    close: close prices indexed by date.
    signal: target position (e.g. 1.0 = fully invested, 0.0 = flat) indexed
        by the same dates as `close`. signal[t] must only use information
        available at or before the close of day t; the engine itself defers
        realizing that position's return to day t+1.
    fee_bps, slippage_bps: cost in basis points applied to traded notional
        whenever the position changes.
    initial_capital: starting portfolio value.
    """

    if not close.index.equals(signal.index):
        raise ValueError("close and signal must share the same index")
    if len(close) < 2:
        raise ValueError("need at least 2 periods to backtest")

    raw_pct_change = close.pct_change()
    # Check for internal NaN BEFORE fillna(0.0) below -- fillna erases
    # every NaN indiscriminately, including internal data gaps, so a check
    # placed after it would never fire (confirmed empirically: a deliberately
    # planted internal NaN in `close` silently became a "0% return" instead
    # of raising, when this check was placed post-fillna). Position 0 is
    # excluded since pct_change() always produces NaN there structurally
    # (no prior price to diff against) -- that is expected, not a data gap.
    if raw_pct_change.iloc[1:].isna().any():
        raise ValueError(
            "close contains internal NaN values after the first row. "
            "Ensure data is clean (e.g. via data.real._handle_missing_data) "
            "before running a backtest."
        )
    asset_return = raw_pct_change.fillna(0.0)

    # Position that was actually held going into day t (decided at t-1).
    position_held = signal.shift(1).fillna(0.0)
    # Position held on the previous day, to detect trades.
    position_prev_held = signal.shift(2).fillna(0.0)

    gross_return = position_held * asset_return

    turnover = (position_held - position_prev_held).abs()
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    cost = turnover * cost_rate

    net_return = gross_return - cost
    equity_curve = initial_capital * (1.0 + net_return).cumprod()

    total_return = float(equity_curve.iloc[-1] / initial_capital - 1.0)

    mean_return = net_return.mean()
    std_return = net_return.std(ddof=1)
    annualized_sharpe = (
        float(mean_return / std_return * np.sqrt(TRADING_DAYS_PER_YEAR))
        if std_return > 0
        else 0.0
    )

    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    max_drawdown = float(drawdown.min())

    win_rate = float((net_return > 0).mean())
    # Dollar cost scaled by the REAL (net, post-cost) equity available at
    # the time of each trade -- this matches what a real brokerage would
    # actually deduct, since your account only ever has its real balance,
    # never a hypothetical "as if fees were never charged" balance.
    #
    # A tempting-looking "fix" is to scale by gross_equity (a hypothetical
    # equity path with costs never subtracted) instead, reasoning that the
    # net equity_curve is already "reduced by its own prior costs" and so
    # is somehow circular. That reasoning is backwards: verified numerically
    # (5-day example, 1% cost, alternating position) -- gross-based scaling
    # reports $4000.00 in costs while the account's real balance only ever
    # dropped by $3940.40 (100000.00 -> 96059.60). The net-based formula
    # below exactly matches the real, observed decline in the account;
    # gross-based scaling systematically overstates costs, growing further
    # from reality the more trades accumulate.
    total_costs_paid = float((cost * equity_curve.shift(1).fillna(initial_capital)).sum())

    return BacktestResult(
        equity_curve=equity_curve,
        net_returns=net_return,
        total_return=total_return,
        annualized_sharpe=annualized_sharpe,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_costs_paid=total_costs_paid,
    )
