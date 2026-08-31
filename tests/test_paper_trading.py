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
    """Redirect state/model persistence to a temp dir so tests never touch
    the real paper_trading/ directories.
    """

    import paper_trading.model_store as model_store
    import paper_trading.state as state

    state_dir = tmp_path / "state"
    model_dir = tmp_path / "models"
    monkeypatch.setattr(state, "STATE_DIR", state_dir)
    monkeypatch.setattr(model_store, "MODEL_DIR", model_dir)
    return state_dir, model_dir


def test_first_run_freezes_model_and_writes_day_one(isolated_dirs):
    from paper_trading.model_store import load_frozen_model
    from paper_trading.state import load_log
    from run_stage6_paper_trading_daily import run_one_day_for_ticker

    close = _make_fake_close(1400, seed=1)
    run_one_day_for_ticker("TESTX", close)

    assert load_frozen_model("TESTX") is not None
    log = load_log("TESTX")
    assert len(log) == 1
    assert log.iloc[0]["held_position"] == 0.0
    assert log.iloc[0]["net_return"] == 0.0
    assert log.iloc[0]["equity"] > 0


def test_second_run_applies_yesterdays_decision_to_todays_return(isolated_dirs):
    from paper_trading.state import load_log
    from run_stage6_paper_trading_daily import run_one_day_for_ticker

    close_day1 = _make_fake_close(1400, seed=2)
    run_one_day_for_ticker("TESTY", close_day1)

    log_day1 = load_log("TESTY")
    target_from_day1 = log_day1.iloc[0]["target_position"]

    next_date = pd.bdate_range(close_day1.index[-1], periods=2)[1]
    close_day2 = pd.concat(
        [close_day1, pd.Series([close_day1.iloc[-1] * 1.02], index=[next_date])]
    )
    run_one_day_for_ticker("TESTY", close_day2)

    log_day2 = load_log("TESTY")
    assert len(log_day2) == 2
    # Day 2's held_position must equal day 1's TARGET (yesterday's decision
    # applied today) -- not day 2's own newly-computed target.
    assert log_day2.iloc[1]["held_position"] == pytest.approx(target_from_day1)
    # A +2% day with a positive held position should show a positive net
    # return (unless held_position is exactly 0, in which case 0).
    if target_from_day1 > 0:
        assert log_day2.iloc[1]["net_return"] > 0


def test_running_twice_on_the_same_day_is_idempotent(isolated_dirs):
    from paper_trading.state import load_log
    from run_stage6_paper_trading_daily import run_one_day_for_ticker

    close = _make_fake_close(1400, seed=3)
    run_one_day_for_ticker("TESTZ", close)
    run_one_day_for_ticker("TESTZ", close)  # same data, same "today" -> no-op

    log = load_log("TESTZ")
    assert len(log) == 1  # not duplicated
