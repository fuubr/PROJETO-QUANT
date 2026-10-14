"""Daily paper trading for the model-free rules (hypotheses H2/H3) plus a
buy-and-hold reference, run beside the model on the SAME canonical prices and
the SAME step engine. Each rule keeps its own log (`<TICKER>__<rule>.csv`), so
the model's log and state are never touched.

Rules run WITHOUT stop-loss or circuit breaker: that is how H2/H3 were
pre-registered (docs/CRITERIOS.md). The breaker cost the model 97% of the
PETR4 period; the rules are tested without it on purpose.
"""

from __future__ import annotations

import pandas as pd

from config import ASSET_COST_OVERRIDES, PAPER_TRADING_INITIAL_CAPITAL, SLIPPAGE_BPS, TRANSACTION_FEE_BPS, VOL_TARGET_ANNUAL
from paper_trading.state import append_row, load_log, state_from_last_row
from risk.managed_backtest import RiskState, step_risk_managed_backtest
from strategies.rules import trend_vol_target_position, vol_target_position

OFF = 0.99  # effectively disables stop-loss and circuit breaker

RULES = {
    "buyhold": lambda close: 1.0,
    "voltarget": lambda close: vol_target_position(close, VOL_TARGET_ANNUAL),
    "trendvol": lambda close: trend_vol_target_position(close, VOL_TARGET_ANNUAL),
}


def rule_log_key(ticker: str, rule: str) -> str:
    return f"{ticker}__{rule}"


def run_rule_day(ticker: str, rule: str, close: pd.Series) -> None:
    key = rule_log_key(ticker, rule)
    today, today_close = close.index[-1], float(close.iloc[-1])
    target = float(RULES[rule](close))
    log = load_log(key)

    if log is None:
        append_row(key, today, today_close, target, 0.0, RiskState.initial(PAPER_TRADING_INITIAL_CAPITAL, today_close))
        return
    if today <= log.index[-1]:
        return  # already processed, or no new bar (weekend/holiday): idempotent

    fee, slip = ASSET_COST_OVERRIDES.get(ticker, (TRANSACTION_FEE_BPS, SLIPPAGE_BPS))
    state, last_close, target_yesterday = state_from_last_row(log)
    new_state, net_return = step_risk_managed_backtest(
        state, price_yesterday=last_close, price_today=today_close, target_position_yesterday=target_yesterday,
        fee_bps=fee, slippage_bps=slip, stop_loss_pct=OFF, circuit_breaker_drawdown_pct=OFF, today_date=today,
    )
    append_row(key, today, today_close, target, net_return, new_state)
