import numpy as np
import pandas as pd
import pytest


def _make_fake_close(n: int, seed: int) -> pd.Series:
    index = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, n)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index)


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    import paper_trading.model_store as model_store
    import paper_trading.state as state

    state_dir = tmp_path / "state"
    model_dir = tmp_path / "models"
    monkeypatch.setattr(state, "STATE_DIR", state_dir)
    monkeypatch.setattr(model_store, "MODEL_DIR", model_dir)
    return state_dir, model_dir


def test_batch_consistency_check_matches_daily_script_exactly(isolated_dirs):
    """The central assumption behind Stage 6's whole design: running the
    daily script day-by-day and running the batch engine over the same
    period with the same frozen model must agree almost exactly, since
    both call the same underlying step function. This is what makes
    "divergence" a meaningful signal instead of expected noise.
    """

    from config import (
        ASSET_LEVEL_CIRCUIT_BREAKER_PCT, CIRCUIT_BREAKER_DRAWDOWN_PCT,
        SLIPPAGE_BPS, STOP_LOSS_PCT, TRANSACTION_FEE_BPS, VAR_CONFIDENCE_LEVEL,
    )
    from paper_trading.consistency_check import check_consistency
    from paper_trading.model_store import list_versions
    from paper_trading.state import load_log
    from run_stage6_paper_trading_daily import run_one_day_for_ticker

    close = _make_fake_close(1400, seed=9)

    # Simula 5 execucoes diarias, estendendo a serie um dia por vez.
    running_close = close.copy()
    for i in range(5):
        next_date = pd.bdate_range(running_close.index[-1], periods=2)[1]
        rng = np.random.default_rng(200 + i)
        new_price = running_close.iloc[-1] * (1 + rng.normal(0.0003, 0.012))
        running_close = pd.concat([running_close, pd.Series([new_price], index=[next_date])])
        run_one_day_for_ticker("CONSISTX", running_close)

    log = load_log("CONSISTX")
    versions = list_versions("CONSISTX")

    result = check_consistency(
        "CONSISTX", running_close, versions, log,
        fee_bps=TRANSACTION_FEE_BPS, slippage_bps=SLIPPAGE_BPS,
        circuit_breaker_pct=CIRCUIT_BREAKER_DRAWDOWN_PCT,
        asset_circuit_breaker_pct=ASSET_LEVEL_CIRCUIT_BREAKER_PCT,
        stop_loss_pct=STOP_LOSS_PCT, var_confidence_level=VAR_CONFIDENCE_LEVEL,
    )

    assert result.matches, (
        f"Divergencia inesperada: max_relative_diff={result.max_relative_equity_diff}"
    )
    assert result.days_compared == len(log)


def _run_days(close, ticker, n, stale_for):
    """Run n daily steps; the first `stale_for` use the LEGACY bug (target
    computed from yesterday's data), reproducing the real production defect.
    """

    import io, contextlib
    import run_stage6_paper_trading_daily as daily

    real_target = daily._today_target_position
    running = close.copy()
    for i in range(n):
        next_date = pd.bdate_range(running.index[-1], periods=2)[1]
        rng = np.random.default_rng(300 + i)
        running = pd.concat([running, pd.Series([running.iloc[-1] * (1 + rng.normal(0.0003, 0.012))], index=[next_date])])
        if i < stale_for:
            daily._today_target_position = lambda c, t, _r=real_target: _r(c.iloc[:-1], t)  # stale: drops today
        else:
            daily._today_target_position = real_target
        with contextlib.redirect_stdout(io.StringIO()):
            daily.run_one_day_for_ticker(ticker, running)
    daily._today_target_position = real_target
    return running


def _check(ticker, running, clean_after):
    from config import (ASSET_LEVEL_CIRCUIT_BREAKER_PCT, CIRCUIT_BREAKER_DRAWDOWN_PCT, SLIPPAGE_BPS, STOP_LOSS_PCT,
                        TRANSACTION_FEE_BPS, VAR_CONFIDENCE_LEVEL)
    from paper_trading.consistency_check import check_consistency
    from paper_trading.model_store import list_versions
    from paper_trading.state import load_log
    log = load_log(ticker)
    return log, check_consistency(
        ticker, running, list_versions(ticker), log, fee_bps=TRANSACTION_FEE_BPS, slippage_bps=SLIPPAGE_BPS,
        circuit_breaker_pct=CIRCUIT_BREAKER_DRAWDOWN_PCT, asset_circuit_breaker_pct=ASSET_LEVEL_CIRCUIT_BREAKER_PCT,
        stop_loss_pct=STOP_LOSS_PCT, var_confidence_level=VAR_CONFIDENCE_LEVEL, clean_after=clean_after,
    )


def test_clean_period_check_is_not_poisoned_by_pre_fix_history(isolated_dirs):
    close = _make_fake_close(1400, seed=12)
    running = _run_days(close, "LEGACY", n=10, stale_for=4)

    log, full = _check("LEGACY", running, clean_after=None)
    assert not full.matches  # the legacy bug really is visible in the full-history check

    legacy_last = log.index[3]  # 4th row = last one computed with the bug
    _, clean = _check("LEGACY", running, clean_after=legacy_last)
    assert clean.matches and clean.days_compared == 5, clean  # rows 6..10 compared after the seed row


def test_clean_period_check_still_catches_a_new_bug(isolated_dirs):
    """The point of the check: a bug appearing AFTER the cutoff must still
    be flagged, not hidden by the cutoff.
    """

    close = _make_fake_close(1400, seed=13)
    running = _run_days(close, "NEWBUG", n=10, stale_for=0)
    log, _ = _check("NEWBUG", running, clean_after=pd.Timestamp("2000-01-01"))

    from paper_trading.state import state_path
    path = state_path("NEWBUG")
    table = pd.read_csv(path)
    table.loc[7, "target_position"] += 0.05  # corrupt one logged decision, mid clean period
    table.to_csv(path, index=False)

    _, result = _check("NEWBUG", running, clean_after=log.index[0])
    assert not result.matches and result.max_target_diff > 0.04


def test_clean_period_check_reports_no_days_before_any_clean_row_exists(isolated_dirs):
    close = _make_fake_close(1400, seed=14)
    running = _run_days(close, "WAIT", n=3, stale_for=3)
    log, result = _check("WAIT", running, clean_after=log_last(running, "WAIT"))
    assert result.days_compared == 0 and result.matches


def log_last(running, ticker):
    from paper_trading.state import load_log
    return load_log(ticker).index[-1]
