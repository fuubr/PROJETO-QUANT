import pandas as pd
import pytest

from paper_trading.report import format_summary_line, summarize_ticker_log


def _fake_log(dates: list[str]) -> pd.DataFrame:
    index = pd.to_datetime(dates)
    n = len(index)
    return pd.DataFrame(
        {
            "close": [100.0 + i for i in range(n)],
            "target_position": [0.2] * n,
            "held_position": [0.2] * n,
            "net_return": [0.001] * n,
            "equity": [100_000.0 * (1.001 ** i) for i in range(n)],
            "entry_price": [100.0] * n,
            "running_peak_equity": [100_000.0 * (1.001 ** i) for i in range(n)],
            "running_peak_price": [100.0 + i for i in range(n)],
            "circuit_breaker_triggered": [False] * n,
            "circuit_breaker_date": [pd.NaT] * n,
            "circuit_breaker_reason": [None] * n,
            "stop_loss_triggers": [0] * n,
        },
        index=index,
    )


def test_summarize_ticker_log_matches_manual_calculation():
    log = _fake_log(["2026-08-31", "2026-09-01", "2026-09-02"])
    today = pd.Timestamp("2026-09-02")

    summary = summarize_ticker_log("TEST", log, today)

    assert summary.days_logged == 3
    assert summary.days_since_last_update == 0
    assert summary.stale is False
    expected_return = log.iloc[-1]["equity"] / log.iloc[0]["equity"] - 1.0
    assert summary.total_return == pytest.approx(expected_return)


def test_summarize_ticker_log_flags_staleness():
    log = _fake_log(["2026-08-31", "2026-09-01"])
    today = pd.Timestamp("2026-09-10")  # 9 days after last update

    summary = summarize_ticker_log("TEST", log, today, max_staleness_days=5)

    assert summary.stale is True
    assert summary.days_since_last_update == 9


def test_summarize_ticker_log_within_normal_weekend_gap_not_stale():
    log = _fake_log(["2026-09-04"])  # a Friday
    today = pd.Timestamp("2026-09-07")  # the following Monday, 3 days later

    summary = summarize_ticker_log("TEST", log, today, max_staleness_days=5)

    assert summary.stale is False


def test_format_summary_line_includes_staleness_warning():
    log = _fake_log(["2026-08-31"])
    today = pd.Timestamp("2026-09-20")

    summary = summarize_ticker_log("TEST", log, today, max_staleness_days=5)
    line = format_summary_line(summary)

    assert "DADO PARADO" in line
    assert "TEST" in line


def test_format_summary_line_clean_when_no_warnings():
    log = _fake_log(["2026-09-01", "2026-09-02"])
    today = pd.Timestamp("2026-09-02")

    summary = summarize_ticker_log("TEST", log, today)
    line = format_summary_line(summary)

    assert "AVISOS" not in line
