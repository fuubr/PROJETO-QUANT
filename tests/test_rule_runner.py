import numpy as np
import pandas as pd
import pytest

import paper_trading.state as state
from paper_trading.rule_runner import RULES, rule_log_key, run_rule_day


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(state, "STATE_DIR", tmp_path / "state")


def _close(n):
    rng = np.random.default_rng(4)
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.0005, 0.01, n)), index=pd.bdate_range("2019-01-01", periods=n))


def test_registered_rules_are_exactly_the_preregistered_ones():
    assert set(RULES) == {"buyhold", "voltarget", "trendvol"}


def test_rule_logs_are_separate_from_the_model_log():
    close = _close(400)
    run_rule_day("SPY", "voltarget", close)
    assert state.load_log(rule_log_key("SPY", "voltarget")) is not None
    assert state.load_log("SPY") is None  # the model's own log is never touched


def test_second_day_applies_yesterdays_target_to_todays_return():
    close = _close(401)
    run_rule_day("SPY", "voltarget", close.iloc[:-1])
    yesterday_target = state.load_log(rule_log_key("SPY", "voltarget")).iloc[-1]["target_position"]
    run_rule_day("SPY", "voltarget", close)
    today = state.load_log(rule_log_key("SPY", "voltarget")).iloc[-1]
    asset_return = close.iloc[-1] / close.iloc[-2] - 1
    assert today["held_position"] == pytest.approx(yesterday_target)
    # SPY costs 0 bps fee + 5 bps slippage (config) on the turnover from 0 to yesterday's target
    expected = yesterday_target * asset_return - yesterday_target * 0.0005
    assert today["net_return"] == pytest.approx(expected, abs=1e-12)


def test_buyhold_reference_is_always_fully_invested():
    close = _close(402)
    run_rule_day("AAPL", "buyhold", close.iloc[:-1])
    run_rule_day("AAPL", "buyhold", close)
    log = state.load_log(rule_log_key("AAPL", "buyhold"))
    assert log["target_position"].eq(1.0).all() and log.iloc[-1]["held_position"] == 1.0


def test_running_twice_on_the_same_day_is_idempotent():
    close = _close(400)
    run_rule_day("SPY", "trendvol", close)
    run_rule_day("SPY", "trendvol", close)
    assert len(state.load_log(rule_log_key("SPY", "trendvol"))) == 1
