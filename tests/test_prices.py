import numpy as np
import pandas as pd
import pytest

import paper_trading.prices as prices
from paper_trading.prices import bootstrap_canonical, load_canonical, update_canonical


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(prices, "PRICES_DIR", tmp_path / "prices")


def _fresh(n=30, seed=0, start="2026-01-01"):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.01, n)), index=pd.bdate_range(start, periods=n))


def test_first_call_stores_the_fresh_history():
    fresh = _fresh()
    out = update_canonical("T", fresh)
    pd.testing.assert_series_equal(out, fresh, check_names=False)
    pd.testing.assert_series_equal(load_canonical("T"), fresh, check_names=False, check_freq=False)


def test_old_bars_are_never_rewritten_when_history_is_readjusted():
    """The core guarantee. Simulate yfinance re-adjusting ALL history for a
    dividend (older prices scaled down 5%) and then adding one new bar:
    the stored past must be untouched and the new bar must carry the true
    one-day return from the SAME download.
    """

    fresh = _fresh(30)
    update_canonical("T", fresh.iloc[:25])
    stored_before = load_canonical("T").copy()

    readjusted = fresh * 1.0
    readjusted.iloc[:26] *= 0.95  # retroactive dividend adjustment of history
    out = update_canonical("T", readjusted.iloc[:26])  # one new bar (index 25)

    pd.testing.assert_series_equal(out.iloc[:25], stored_before, check_freq=False)
    expected = stored_before.iloc[-1] * readjusted.iloc[25] / readjusted.iloc[24]
    assert out.iloc[-1] == pytest.approx(expected)


def test_same_download_ratio_includes_the_dividend_as_total_return():
    # Raw price drops 3% on ex-dividend day, but the adjusted download shows a flat day.
    idx = pd.bdate_range("2026-01-01", periods=3)
    first = pd.Series([100.0, 100.0], index=idx[:2])
    update_canonical("T", first)
    adjusted_next = pd.Series([97.0, 97.0, 97.0], index=idx)  # dividend-adjusted: flat
    out = update_canonical("T", adjusted_next)
    assert out.iloc[-1] == pytest.approx(100.0)  # no phantom -3% "loss"


def test_no_new_bars_is_a_noop_and_refuses_unlinkable_data():
    fresh = _fresh(20)
    update_canonical("T", fresh)
    assert len(update_canonical("T", fresh)) == 20
    with pytest.raises(ValueError, match="missing"):
        update_canonical("T", fresh.iloc[5:15].shift(0).iloc[:5].rename(lambda d: d + pd.Timedelta(days=400)))


def test_bootstrap_from_existing_log_keeps_logged_closes_verbatim():
    fresh = _fresh(40)
    log_idx = fresh.index[30:35]
    log = pd.DataFrame({"close": (fresh.loc[log_idx] * 1.02).to_numpy()}, index=log_idx)  # logged on a different basis
    out = bootstrap_canonical(fresh, log)
    pd.testing.assert_series_equal(out.loc[log_idx], log["close"], check_names=False, check_freq=False)
    # history is rescaled so the join has no artificial jump
    jump = out.iloc[29] / out.iloc[30]
    assert jump == pytest.approx(fresh.iloc[29] / fresh.iloc[30])


def test_end_to_end_consistency_survives_daily_retroactive_readjustment(tmp_path, monkeypatch):
    """The real guarantee: every day the 'download' re-adjusts ALL history by
    a random factor (as dividends/revisions do). Decisions run on the
    canonical series, so the replay still matches the log EXACTLY -- the
    failure mode found on the real repo (logged vs re-downloaded prices up
    to 2.7% apart) cannot recur.
    """

    import io, contextlib
    import paper_trading.model_store as model_store
    import paper_trading.state as state
    from config import (ASSET_LEVEL_CIRCUIT_BREAKER_PCT, CIRCUIT_BREAKER_DRAWDOWN_PCT, SLIPPAGE_BPS, STOP_LOSS_PCT,
                        TRANSACTION_FEE_BPS, VAR_CONFIDENCE_LEVEL)
    from paper_trading.consistency_check import check_consistency
    from paper_trading.model_store import list_versions
    from run_stage6_paper_trading_daily import run_one_day_for_ticker

    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(model_store, "MODEL_DIR", tmp_path / "models")

    rng = np.random.default_rng(5)
    truth = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, 1420))), index=pd.bdate_range("2015-01-01", periods=1420))
    for day in range(1400, 1412):
        download = truth.iloc[: day + 1] * 1.0
        download.iloc[:-1] *= rng.uniform(0.95, 1.0)  # retroactive re-adjustment of the whole past, new every day
        canonical = update_canonical("E2E", download, state.load_log("E2E"))
        with contextlib.redirect_stdout(io.StringIO()):
            run_one_day_for_ticker("E2E", canonical)

    log = state.load_log("E2E")
    result = check_consistency(
        "E2E", load_canonical("E2E"), list_versions("E2E"), log, fee_bps=TRANSACTION_FEE_BPS, slippage_bps=SLIPPAGE_BPS,
        circuit_breaker_pct=CIRCUIT_BREAKER_DRAWDOWN_PCT, asset_circuit_breaker_pct=ASSET_LEVEL_CIRCUIT_BREAKER_PCT,
        stop_loss_pct=STOP_LOSS_PCT, var_confidence_level=VAR_CONFIDENCE_LEVEL,
    )
    assert len(log) == 12 and result.matches, result


def test_migration_from_log_chains_bars_newer_than_the_last_logged_day():
    """Regression: bootstrapping from an existing log must also append the
    bars after the last logged date in the same call (the real repo's log
    ends 2026-09-30 while a newer bar already exists).
    """

    fresh = _fresh(40)
    log_idx = fresh.index[30:35]
    log = pd.DataFrame({"close": (fresh.loc[log_idx] * 1.02).to_numpy()}, index=log_idx)
    out = update_canonical("T", fresh, log)
    assert out.index[-1] == fresh.index[-1] and len(out) == 40
    # the new bars carry the true same-download returns on top of the logged level
    assert out.iloc[-1] / out.iloc[34] == pytest.approx(fresh.iloc[-1] / fresh.iloc[34])
    pd.testing.assert_series_equal(out.loc[log_idx], log["close"], check_names=False, check_freq=False)
