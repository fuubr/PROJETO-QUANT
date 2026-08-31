"""Risk-managed backtest engine: applies stop-loss and a portfolio-level
circuit breaker on top of a target position series (e.g. from Kelly
sizing).

Unlike backtest/engine.py (purely vectorized, since a fixed signal's
outcome on any given day doesn't depend on what happened on other days),
stop-loss and circuit-breaker rules are PATH-DEPENDENT: whether today's
position gets force-closed depends on the running entry price and the
running drawdown, both of which depend on every prior day's decisions.

The per-day decision logic lives in ONE place (RiskState +
step_risk_managed_backtest) and is used both by the batch backtest below
(looping over a full price history) and by Stage 6's daily paper trading
script (calling it once per day against persisted state) -- extracted
specifically so those two callers can never silently drift apart, the same
class of duplication risk already found and fixed once between
run_stage3_model.py and run_stage3_holdout_evaluation.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from risk.metrics import TRADING_DAYS_PER_YEAR, annualized_volatility, historical_var, sortino_ratio


@dataclass
class RiskState:
    """Everything the risk overlays need to remember between days. Fully
    serializable (plain floats/bools/str/None) so Stage 6 can persist and
    reload it between separate daily process runs.
    """

    held_position: float = 0.0
    entry_price: float | None = None
    equity: float = 0.0
    running_peak_equity: float = 0.0
    running_peak_price: float = 0.0
    circuit_breaker_triggered: bool = False
    circuit_breaker_date: pd.Timestamp | None = None
    circuit_breaker_reason: str | None = None
    stop_loss_triggers: int = 0

    @classmethod
    def initial(cls, initial_capital: float, first_price: float) -> "RiskState":
        return cls(
            held_position=0.0,
            entry_price=None,
            equity=initial_capital,
            running_peak_equity=initial_capital,
            running_peak_price=first_price,
            circuit_breaker_triggered=False,
            circuit_breaker_date=None,
            circuit_breaker_reason=None,
            stop_loss_triggers=0,
        )


@dataclass
class RiskManagedBacktestResult:
    equity_curve: pd.Series
    net_returns: pd.Series
    actual_position: pd.Series
    total_return: float
    annualized_sharpe: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_costs_paid: float
    historical_var_95: float
    annualized_volatility: float
    stop_loss_triggers: int
    circuit_breaker_triggered: bool
    circuit_breaker_date: pd.Timestamp | None
    circuit_breaker_reason: str | None
    active_trading_days: int
    total_days: int


def step_risk_managed_backtest(
    state: RiskState,
    price_yesterday: float,
    price_today: float,
    target_position_yesterday: float,
    fee_bps: float,
    slippage_bps: float,
    stop_loss_pct: float,
    circuit_breaker_drawdown_pct: float,
    today_date: pd.Timestamp,
    asset_circuit_breaker_drawdown_pct: float | None = None,
) -> tuple[RiskState, float]:
    """Advance the risk-managed simulation by exactly one day.

    target_position_yesterday: the position DECIDED using information
    available at or before yesterday's close (same convention as
    backtest/engine.py and the batch version of this function) -- its
    effect is realized in today's return.

    Returns (new_state, net_return_today). Pure function: never mutates
    the passed-in state, so callers (batch loop or a daily script re-reading
    persisted state from disk) can't accidentally share/corrupt state
    between calls.
    """

    if not 0.0 < stop_loss_pct < 1.0:
        raise ValueError("stop_loss_pct must be between 0 and 1")
    if not 0.0 < circuit_breaker_drawdown_pct < 1.0:
        raise ValueError("circuit_breaker_drawdown_pct must be between 0 and 1")
    if asset_circuit_breaker_drawdown_pct is not None and not 0.0 < asset_circuit_breaker_drawdown_pct < 1.0:
        raise ValueError("asset_circuit_breaker_drawdown_pct must be between 0 and 1")

    cost_rate = (fee_bps + slippage_bps) / 10_000.0

    proposed_position = 0.0 if state.circuit_breaker_triggered else float(target_position_yesterday)

    if state.held_position > 0 and state.entry_price is not None:
        loss_from_entry = (price_yesterday - state.entry_price) / state.entry_price
        stop_loss_fired = loss_from_entry <= -stop_loss_pct
    else:
        stop_loss_fired = False

    new_held_position = 0.0 if stop_loss_fired else proposed_position
    new_stop_loss_triggers = state.stop_loss_triggers + (1 if stop_loss_fired else 0)

    if new_held_position > 0 and state.held_position == 0:
        new_entry_price = price_yesterday
    elif new_held_position > state.held_position and state.entry_price is not None:
        added_fraction = new_held_position - state.held_position
        new_entry_price = (
            state.held_position * state.entry_price + added_fraction * price_yesterday
        ) / new_held_position
    elif new_held_position == 0:
        new_entry_price = None
    else:
        new_entry_price = state.entry_price

    asset_return = (price_today - price_yesterday) / price_yesterday
    gross_return = new_held_position * asset_return
    turnover = abs(new_held_position - state.held_position)
    cost = turnover * cost_rate
    net_return = gross_return - cost

    new_equity = state.equity * (1.0 + net_return)
    new_running_peak_equity = max(state.running_peak_equity, new_equity)
    portfolio_drawdown = new_equity / new_running_peak_equity - 1.0

    new_running_peak_price = max(state.running_peak_price, price_today)
    asset_drawdown = price_today / new_running_peak_price - 1.0

    new_circuit_breaker_triggered = state.circuit_breaker_triggered
    new_circuit_breaker_date = state.circuit_breaker_date
    new_circuit_breaker_reason = state.circuit_breaker_reason

    if not state.circuit_breaker_triggered and portfolio_drawdown <= -circuit_breaker_drawdown_pct:
        new_circuit_breaker_triggered = True
        new_circuit_breaker_date = today_date
        new_circuit_breaker_reason = "portfolio"
    elif (
        not state.circuit_breaker_triggered
        and asset_circuit_breaker_drawdown_pct is not None
        and asset_drawdown <= -asset_circuit_breaker_drawdown_pct
    ):
        new_circuit_breaker_triggered = True
        new_circuit_breaker_date = today_date
        new_circuit_breaker_reason = "asset"

    new_state = replace(
        state,
        held_position=new_held_position,
        entry_price=new_entry_price,
        equity=new_equity,
        running_peak_equity=new_running_peak_equity,
        running_peak_price=new_running_peak_price,
        circuit_breaker_triggered=new_circuit_breaker_triggered,
        circuit_breaker_date=new_circuit_breaker_date,
        circuit_breaker_reason=new_circuit_breaker_reason,
        stop_loss_triggers=new_stop_loss_triggers,
    )

    return new_state, net_return


def run_risk_managed_backtest(
    close: pd.Series,
    target_position: pd.Series,
    fee_bps: float,
    slippage_bps: float,
    initial_capital: float,
    stop_loss_pct: float,
    circuit_breaker_drawdown_pct: float,
    var_confidence_level: float,
    asset_circuit_breaker_drawdown_pct: float | None = None,
) -> RiskManagedBacktestResult:
    """Simulate PnL for a target position series, with two risk overlays
    (stop-loss and portfolio/asset circuit breaker).

    target_position[t] represents a decision made using information
    available at or before the close of day t (same convention as
    backtest/engine.py); its effect is deferred to day t+1's return.

    Internally, this is a thin loop around step_risk_managed_backtest --
    see that function for the actual per-day decision logic, shared with
    Stage 6's daily paper trading script.
    """

    if not close.index.equals(target_position.index):
        raise ValueError("close and target_position must share the same index")
    if len(close) < 2:
        raise ValueError("need at least 2 periods to backtest")

    prices = close.to_numpy()
    target = target_position.to_numpy()
    n = len(prices)

    held = np.zeros(n)
    net_returns = np.zeros(n)
    equity = np.empty(n)
    equity[0] = initial_capital

    state = RiskState.initial(initial_capital, prices[0])

    for t in range(1, n):
        state, net_return = step_risk_managed_backtest(
            state,
            price_yesterday=prices[t - 1],
            price_today=prices[t],
            target_position_yesterday=float(target[t - 1]),
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            stop_loss_pct=stop_loss_pct,
            circuit_breaker_drawdown_pct=circuit_breaker_drawdown_pct,
            today_date=close.index[t],
            asset_circuit_breaker_drawdown_pct=asset_circuit_breaker_drawdown_pct,
        )
        held[t] = state.held_position
        equity[t] = state.equity
        net_returns[t] = net_return

    equity_series = pd.Series(equity, index=close.index)
    net_returns_series = pd.Series(net_returns, index=close.index)
    actual_position_series = pd.Series(held, index=close.index)

    total_return = float(equity_series.iloc[-1] / initial_capital - 1.0)

    mean_return = net_returns_series.mean()
    std_return = net_returns_series.std(ddof=1)
    annualized_sharpe = (
        float(mean_return / std_return * np.sqrt(TRADING_DAYS_PER_YEAR))
        if std_return > 0
        else 0.0
    )

    running_max = equity_series.cummax()
    drawdown_series = equity_series / running_max - 1.0
    max_drawdown = float(drawdown_series.min())

    win_rate = float((net_returns_series.iloc[1:] > 0).mean())
    cost_rate = (fee_bps + slippage_bps) / 10_000.0
    total_costs_paid = float(
        (
            actual_position_series.diff().abs().fillna(0.0)
            * cost_rate
            * equity_series.shift(1).fillna(initial_capital)
        ).sum()
    )

    active_trading_days = (
        close.index.get_loc(state.circuit_breaker_date) if state.circuit_breaker_date is not None else n
    )

    return RiskManagedBacktestResult(
        equity_curve=equity_series,
        net_returns=net_returns_series,
        actual_position=actual_position_series,
        total_return=total_return,
        annualized_sharpe=annualized_sharpe,
        sortino_ratio=sortino_ratio(net_returns_series.iloc[1:]),
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        total_costs_paid=total_costs_paid,
        historical_var_95=historical_var(net_returns_series.iloc[1:], var_confidence_level),
        annualized_volatility=annualized_volatility(net_returns_series.iloc[1:]),
        stop_loss_triggers=state.stop_loss_triggers,
        circuit_breaker_triggered=state.circuit_breaker_triggered,
        circuit_breaker_date=state.circuit_breaker_date,
        circuit_breaker_reason=state.circuit_breaker_reason,
        active_trading_days=active_trading_days,
        total_days=n,
    )
