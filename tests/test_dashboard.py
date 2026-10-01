import numpy as np
import pandas as pd

from paper_trading.consistency_check import ConsistencyCheckResult
from paper_trading.dashboard import generate_dashboard
from paper_trading.report import summarize_ticker_log


def _fake_log(n: int) -> pd.DataFrame:
    index = pd.bdate_range("2026-08-01", periods=n)
    rng = np.random.default_rng(0)
    equity = 100_000.0 * np.cumprod(1 + rng.normal(0.0005, 0.01, n))
    return pd.DataFrame(
        {
            "close": 100.0 + np.arange(n) * 0.3,
            "target_position": [0.2] * n,
            "held_position": [0.2] * n,
            "net_return": [0.0005] * n,
            "equity": equity,
            "entry_price": [100.0] * n,
            "running_peak_equity": np.maximum.accumulate(equity),
            "running_peak_price": 100.0 + np.arange(n) * 0.3,
            "circuit_breaker_triggered": [False] * n,
            "circuit_breaker_date": [pd.NaT] * n,
            "circuit_breaker_reason": [None] * n,
            "stop_loss_triggers": [0] * n,
        },
        index=index,
    )


def test_generate_dashboard_creates_html_with_expected_content(tmp_path):
    log = _fake_log(15)
    today = log.index[-1]
    summary = summarize_ticker_log("FAKE", log, today)

    consistency = ConsistencyCheckResult(
        ticker="FAKE", days_compared=15, max_absolute_equity_diff=0.01,
        max_relative_equity_diff=1e-7, matches=True,
    )

    output_path = tmp_path / "dashboard.html"
    generate_dashboard(
        summaries=[summary], logs={"FAKE": log}, consistency_results={"FAKE": consistency},
        output_path=output_path, generated_at=today,
    )

    assert output_path.exists()
    content = output_path.read_text()
    assert "FAKE" in content
    assert "bate com o backtest" in content
    assert "<img src=\"data:image/png;base64," in content


def test_generate_dashboard_flags_divergence(tmp_path):
    log = _fake_log(10)
    today = log.index[-1]
    summary = summarize_ticker_log("DIVERGENT", log, today)

    consistency = ConsistencyCheckResult(
        ticker="DIVERGENT", days_compared=10, max_absolute_equity_diff=500.0,
        max_relative_equity_diff=0.05, matches=False,
    )

    output_path = tmp_path / "dashboard.html"
    generate_dashboard(
        summaries=[summary], logs={"DIVERGENT": log}, consistency_results={"DIVERGENT": consistency},
        output_path=output_path, generated_at=today,
    )

    content = output_path.read_text()
    assert "DIVERGE do backtest" in content
