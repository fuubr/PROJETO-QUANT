"""Daily paper trading state persistence.

One append-only CSV per ticker is the entire state: each row is a fully
processed day's outcome. No separate state file -- the CSV's LAST row
contains everything needed to reconstruct a RiskState and resume tomorrow,
which means the log is simultaneously the audit trail and the state
store. Losing a row is losing history, not corrupting live state, since
each row's fields are exactly what RiskState needs.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from risk.managed_backtest import RiskState

STATE_DIR = Path(__file__).resolve().parent / "state"


def state_path(ticker: str) -> Path:
    safe_ticker = ticker.replace("/", "-").replace("^", "")
    return STATE_DIR / f"{safe_ticker}.csv"


def load_log(ticker: str) -> pd.DataFrame | None:
    """Returns the full history log, or None if this ticker has never been
    run before (first-run case).
    """

    path = state_path(ticker)
    if not path.exists():
        return None
    log = pd.read_csv(path, parse_dates=["date", "circuit_breaker_date"])
    log = log.set_index("date")
    return log


def state_from_last_row(log: pd.DataFrame) -> tuple[RiskState, float, float]:
    """Reconstruct a RiskState from the log's last row, plus that row's
    close price and the target_position it computed (for tomorrow's use).

    Returns (state, last_close, next_target_position).
    """

    last = log.iloc[-1]
    circuit_breaker_date = last["circuit_breaker_date"]
    if pd.isna(circuit_breaker_date):
        circuit_breaker_date = None

    entry_price = last["entry_price"]
    if pd.isna(entry_price):
        entry_price = None

    circuit_breaker_reason = last["circuit_breaker_reason"]
    if pd.isna(circuit_breaker_reason):
        circuit_breaker_reason = None

    state = RiskState(
        held_position=float(last["held_position"]),
        entry_price=None if entry_price is None else float(entry_price),
        equity=float(last["equity"]),
        running_peak_equity=float(last["running_peak_equity"]),
        running_peak_price=float(last["running_peak_price"]),
        circuit_breaker_triggered=bool(last["circuit_breaker_triggered"]),
        circuit_breaker_date=circuit_breaker_date,
        circuit_breaker_reason=circuit_breaker_reason,
        stop_loss_triggers=int(last["stop_loss_triggers"]),
    )
    return state, float(last["close"]), float(last["target_position"])


def append_row(
    ticker: str,
    date: pd.Timestamp,
    close: float,
    target_position: float,
    net_return: float,
    state: RiskState,
) -> None:
    """Append one fully processed day's row to this ticker's log.

    target_position here is the NEW position computed TODAY (using data
    through today) for use as tomorrow's "yesterday's decision" -- not the
    position that generated today's net_return, which is state.held_position
    (decided on the PRIOR run).
    """

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    path = state_path(ticker)

    row = pd.DataFrame(
        [
            {
                "date": date,
                "close": close,
                "target_position": target_position,
                "held_position": state.held_position,
                "net_return": net_return,
                "equity": state.equity,
                "entry_price": state.entry_price,
                "running_peak_equity": state.running_peak_equity,
                "running_peak_price": state.running_peak_price,
                "circuit_breaker_triggered": state.circuit_breaker_triggered,
                "circuit_breaker_date": state.circuit_breaker_date,
                "circuit_breaker_reason": state.circuit_breaker_reason,
                "stop_loss_triggers": state.stop_loss_triggers,
            }
        ]
    )

    if path.exists():
        row.to_csv(path, mode="a", header=False, index=False)
    else:
        row.to_csv(path, mode="w", header=True, index=False)
