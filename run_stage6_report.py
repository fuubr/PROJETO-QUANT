"""Generates a full paper trading report: text summary (with staleness
and warning flags), backtest-vs-paper-trading consistency check, and the
HTML dashboard -- in one run, so checking progress never again requires
manually pasting CSVs for review.

Run manually any time: python run_stage6_report.py
Also run automatically by the GitHub Actions workflow after each daily
update, so the dashboard (paper_trading/dashboard.html) always reflects
the latest committed state.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    PAPER_TRADING_BUGGY_THROUGH,
    PAPER_TRADING_TICKERS,
    STALENESS_WARNING_DAYS,
    STOP_LOSS_PCT,
    VAR_CONFIDENCE_LEVEL,
)
from paper_trading.prices import load_canonical
from paper_trading.consistency_check import check_consistency
from paper_trading.dashboard import generate_dashboard
from paper_trading.model_store import list_versions
from paper_trading.report import format_summary_line, summarize_ticker_log
from paper_trading.state import load_log
from run_stage6_paper_trading_daily import _asset_costs, _circuit_breaker_thresholds

DASHBOARD_PATH = Path(__file__).resolve().parent / "paper_trading" / "dashboard.html"
SUMMARY_PATH = Path(__file__).resolve().parent / "paper_trading" / "daily_summary.txt"


def main() -> None:
    today = pd.Timestamp.now()
    summaries = []
    summary_lines = []
    logs = {}
    consistency_results = {}

    for ticker in PAPER_TRADING_TICKERS:
        log = load_log(ticker)
        if log is None:
            print(f"{ticker}: sem historico ainda (paper trading nunca rodou para este ativo).")
            continue

        summary = summarize_ticker_log(ticker, log, today, max_staleness_days=STALENESS_WARNING_DAYS)
        print(format_summary_line(summary))
        summary_lines.append(format_summary_line(summary))
        summaries.append(summary)
        logs[ticker] = log

        versions = list_versions(ticker)
        if not versions:
            continue

        canonical = load_canonical(ticker)
        if canonical is None:
            continue

        fee_bps, slippage_bps = _asset_costs(ticker)
        circuit_breaker_pct, asset_circuit_breaker_pct = _circuit_breaker_thresholds(ticker)

        result = check_consistency(
            ticker, canonical, versions, log, fee_bps=fee_bps, slippage_bps=slippage_bps,
            circuit_breaker_pct=circuit_breaker_pct, asset_circuit_breaker_pct=asset_circuit_breaker_pct,
            stop_loss_pct=STOP_LOSS_PCT, var_confidence_level=VAR_CONFIDENCE_LEVEL,
            clean_after=pd.Timestamp(PAPER_TRADING_BUGGY_THROUGH),
        )
        consistency_results[ticker] = result
        status = "aguardando dias pos-correcao" if result.days_compared == 0 else ("OK" if result.matches else "*** DIVERGE ***")
        print(f"  Consistencia backtest-vs-real [{ticker}]: {status} "
              f"(diferenca relativa maxima: {result.max_relative_equity_diff:.6%})")

    if not summaries:
        print("Nenhum historico de paper trading encontrado ainda para nenhum ativo.")
        return

    SUMMARY_PATH.write_text("\n".join(summary_lines), encoding="utf-8")
    generate_dashboard(summaries, logs, consistency_results, DASHBOARD_PATH, generated_at=today)
    print(f"\nDashboard gerado em: {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
