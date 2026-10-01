"""Summary statistics and staleness detection for paper trading logs.

Pure functions operating on an already-loaded log DataFrame (from
paper_trading.state.load_log) -- kept separate from I/O so they're easy to
test and easy to reuse from both a CLI report and the HTML dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class TickerSummary:
    ticker: str
    days_logged: int
    first_date: pd.Timestamp
    last_date: pd.Timestamp
    total_return: float
    current_equity: float
    current_position: float
    stop_loss_triggers: int
    circuit_breaker_triggered: bool
    circuit_breaker_date: pd.Timestamp | None
    circuit_breaker_reason: str | None
    days_since_last_update: int  # calendar days between last_date and "today"
    stale: bool  # True if days_since_last_update exceeds the threshold


def summarize_ticker_log(
    ticker: str, log: pd.DataFrame, today: pd.Timestamp, max_staleness_days: int = 5
) -> TickerSummary:
    """Summarize one ticker's paper trading log.

    max_staleness_days=5 as a default: covers a normal weekend (2 days)
    plus a bit of slack for a single holiday, without being so loose that a
    genuinely broken data feed goes unnoticed for a long stretch.
    """

    first_date = log.index.min()
    last_date = log.index.max()
    last_row = log.loc[last_date]

    initial_equity = log.iloc[0]["equity"]
    current_equity = last_row["equity"]
    total_return = float(current_equity / initial_equity - 1.0) if initial_equity > 0 else float("nan")

    circuit_breaker_date = last_row["circuit_breaker_date"]
    if pd.isna(circuit_breaker_date):
        circuit_breaker_date = None

    circuit_breaker_reason = last_row["circuit_breaker_reason"]
    if pd.isna(circuit_breaker_reason):
        circuit_breaker_reason = None

    days_since_last_update = (today.normalize() - last_date.normalize()).days

    return TickerSummary(
        ticker=ticker,
        days_logged=len(log),
        first_date=first_date,
        last_date=last_date,
        total_return=total_return,
        current_equity=float(current_equity),
        current_position=float(last_row["held_position"]),
        stop_loss_triggers=int(last_row["stop_loss_triggers"]),
        circuit_breaker_triggered=bool(last_row["circuit_breaker_triggered"]),
        circuit_breaker_date=circuit_breaker_date,
        circuit_breaker_reason=circuit_breaker_reason,
        days_since_last_update=days_since_last_update,
        stale=days_since_last_update > max_staleness_days,
    )


def format_summary_line(summary: TickerSummary) -> str:
    warnings = []
    if summary.stale:
        warnings.append(
            f"DADO PARADO HA {summary.days_since_last_update} DIAS -- "
            f"verificar se o feed de dados (yfinance) esta funcionando para {summary.ticker}"
        )
    if summary.circuit_breaker_triggered:
        warnings.append(f"circuit breaker ({summary.circuit_breaker_reason}) disparado em {summary.circuit_breaker_date.date()}")
    if summary.stop_loss_triggers > 0:
        warnings.append(f"{summary.stop_loss_triggers} disparo(s) de stop-loss")

    warning_text = f" -- AVISOS: {'; '.join(warnings)}" if warnings else ""

    return (
        f"{summary.ticker}: {summary.days_logged} dias ({summary.first_date.date()} a "
        f"{summary.last_date.date()}), retorno acumulado {summary.total_return:+.2%}, "
        f"equity {summary.current_equity:,.2f}, posicao atual {summary.current_position:.4f}"
        f"{warning_text}"
    )
