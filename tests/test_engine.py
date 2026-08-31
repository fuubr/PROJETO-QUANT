import numpy as np
import pandas as pd
import pytest

from backtest.baselines import (
    buy_and_hold_signal,
    moving_average_crossover_signal,
    random_entry_persistent_signal,
    random_entry_signal,
)
from backtest.engine import run_backtest
from backtest.statistics import newey_west_t_stat
from config import INITIAL_CAPITAL, SEED, SLIPPAGE_BPS, SYNTHETIC_PERIODS, TRANSACTION_FEE_BPS
from data.synthetic import generate_random_walk_prices

STATISTICAL_TEST_PERIODS = 50_000


def test_engine_rejects_mismatched_index():
    close = pd.Series([100.0, 101.0], index=pd.bdate_range("2020-01-01", periods=2))
    signal = pd.Series([1.0, 1.0], index=pd.bdate_range("2020-01-02", periods=2))

    with pytest.raises(ValueError, match="index"):
        run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0)


def test_engine_has_no_lookahead_bias():
    """A signal that only "knows" about a future price jump must not be able
    to capture that jump's return on the same day it is set."""

    index = pd.bdate_range("2020-01-01", periods=5)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 200.0], index=index)

    # Signal turns on the exact day of the jump, as if it had perfect
    # foresight. If the engine had lookahead bias, this would capture the
    # 100% jump on day 5. It must not.
    signal = pd.Series([0.0, 0.0, 0.0, 0.0, 1.0], index=index)

    result = run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0)

    assert result.total_return == pytest.approx(0.0)


def test_engine_captures_return_one_day_after_signal():
    index = pd.bdate_range("2020-01-01", periods=5)
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 200.0], index=index)

    # Signal turns on the day BEFORE the jump: this position should be held
    # into the jump and capture the full return.
    signal = pd.Series([0.0, 0.0, 0.0, 1.0, 1.0], index=index)

    result = run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0)

    assert result.total_return == pytest.approx(1.0, abs=1e-6)


def test_engine_charges_cost_only_on_position_changes():
    index = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 100.0, 100.0, 100.0], index=index)
    signal = pd.Series([1.0, 1.0, 1.0, 1.0], index=index)

    result = run_backtest(close, signal, fee_bps=10.0, slippage_bps=0.0, initial_capital=100_000.0)

    # Position only "opens" once (shift from 0 -> 1 on day 2), so cost
    # should be charged exactly once, not on every day the position is held.
    non_zero_cost_days = (result.net_returns != 0.0).sum()
    assert non_zero_cost_days == 1


def test_engine_zero_cost_flat_position_has_zero_return():
    index = pd.bdate_range("2020-01-01", periods=5)
    close = pd.Series([100.0, 105.0, 95.0, 110.0, 90.0], index=index)
    signal = buy_and_hold_signal(index) * 0.0  # always flat

    result = run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0)

    assert result.total_return == pytest.approx(0.0)
    assert result.equity_curve.iloc[-1] == pytest.approx(100_000.0)


def test_buy_and_hold_on_synthetic_asset_has_no_significant_edge():
    """Statistical gate for this stage: the engine must not manufacture
    spurious edge out of an asset that has none by construction. If this
    fails, the bug is in the engine (e.g. hidden lookahead or a cost
    calculation error), not in the market.
    """

    prices = generate_random_walk_prices(periods=STATISTICAL_TEST_PERIODS)
    signal = buy_and_hold_signal(prices.index)

    result = run_backtest(
        prices["close"],
        signal,
        fee_bps=TRANSACTION_FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        initial_capital=INITIAL_CAPITAL,
    )

    returns = result.net_returns.iloc[1:].to_numpy()
    t_statistic = newey_west_t_stat(returns)

    assert abs(t_statistic) < 1.96


def test_random_entry_on_synthetic_asset_has_no_significant_edge():
    """Isolates predictive edge from transaction costs by using zero cost.

    random_entry_signal is i.i.d. Bernoulli each day (no persistence), so it
    has ~50% daily turnover. With real transaction costs that turnover
    produces a large, EXPECTED, and entirely explainable negative drag (see
    test_random_entry_high_turnover_cost_drag_matches_expectation below).
    That cost drag is not a spurious edge, so it must not be mixed into this
    gate: this test isolates whether the engine manufactures directional
    bias out of pure price noise, independent of cost effects.
    """

    prices = generate_random_walk_prices(periods=STATISTICAL_TEST_PERIODS)
    signal = random_entry_signal(prices.index, seed=SEED, probability=0.5)

    result = run_backtest(
        prices["close"],
        signal,
        fee_bps=0.0,
        slippage_bps=0.0,
        initial_capital=INITIAL_CAPITAL,
    )

    returns = result.net_returns.iloc[1:].to_numpy()
    t_statistic = newey_west_t_stat(returns)

    assert abs(t_statistic) < 1.96


def test_random_entry_high_turnover_cost_drag_matches_expectation():
    """Documents and validates a real (not spurious) effect: a strategy with
    ~50% daily turnover and no persistence pays transaction costs on roughly
    half of all days. The resulting drag should match turnover * cost_rate
    within sampling noise, confirming the cost calculation itself -- not the
    statistical edge gate -- is what's responsible for the large negative
    return seen with costs enabled.
    """

    prices = generate_random_walk_prices(periods=STATISTICAL_TEST_PERIODS)
    signal = random_entry_signal(prices.index, seed=SEED, probability=0.5)

    result = run_backtest(
        prices["close"],
        signal,
        fee_bps=TRANSACTION_FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        initial_capital=INITIAL_CAPITAL,
    )

    observed_turnover_rate = (signal.diff().abs() > 0).mean()
    # Bernoulli(0.5) with no persistence flips on ~50% of days.
    assert observed_turnover_rate == pytest.approx(0.5, abs=0.02)

    cost_rate = (TRANSACTION_FEE_BPS + SLIPPAGE_BPS) / 10_000.0
    expected_daily_cost_drag = observed_turnover_rate * cost_rate

    # Net return should be materially worse than gross return by
    # approximately the expected cost drag (not zero, and not double).
    gross_result = run_backtest(
        prices["close"], signal, fee_bps=0.0, slippage_bps=0.0, initial_capital=INITIAL_CAPITAL
    )
    observed_drag = gross_result.net_returns.mean() - result.net_returns.mean()

    assert observed_drag == pytest.approx(expected_daily_cost_drag, rel=0.1)


def test_max_drawdown_is_non_positive_and_bounded():
    prices = generate_random_walk_prices(periods=SYNTHETIC_PERIODS)
    signal = buy_and_hold_signal(prices.index)

    result = run_backtest(
        prices["close"],
        signal,
        fee_bps=TRANSACTION_FEE_BPS,
        slippage_bps=SLIPPAGE_BPS,
        initial_capital=INITIAL_CAPITAL,
    )

    assert result.max_drawdown <= 0.0
    assert result.max_drawdown >= -1.0


def test_random_entry_persistent_has_lower_turnover_than_iid():
    index = pd.bdate_range("2020-01-01", periods=STATISTICAL_TEST_PERIODS)

    iid_signal = random_entry_signal(index, seed=SEED, probability=0.5)
    persistent_signal = random_entry_persistent_signal(
        index, seed=SEED, probability=0.5, avg_holding_days=20.0
    )

    iid_turnover = (iid_signal.diff().abs() > 0).mean()
    persistent_turnover = (persistent_signal.diff().abs() > 0).mean()

    # ~50% for the i.i.d. version, ~1/20 = 5% for the persistent version.
    assert iid_turnover == pytest.approx(0.5, abs=0.02)
    assert persistent_turnover == pytest.approx(0.05, abs=0.02)
    assert persistent_turnover < iid_turnover


def test_random_entry_persistent_on_synthetic_asset_has_no_significant_edge():
    prices = generate_random_walk_prices(periods=STATISTICAL_TEST_PERIODS)
    signal = random_entry_persistent_signal(
        prices.index, seed=SEED, probability=0.5, avg_holding_days=20.0
    )

    result = run_backtest(
        prices["close"],
        signal,
        fee_bps=0.0,
        slippage_bps=0.0,
        initial_capital=INITIAL_CAPITAL,
    )

    returns = result.net_returns.iloc[1:].to_numpy()
    t_statistic = newey_west_t_stat(returns)

    assert abs(t_statistic) < 1.96


def test_random_entry_persistent_pays_less_cost_drag_than_iid():
    """Compares cost DRAG AS A RATE, not compounded dollar totals.

    Over STATISTICAL_TEST_PERIODS (~200 years, chosen only for t-test power
    elsewhere in this file), compounding makes dollar-total cost comparisons
    meaningless: a high-turnover strategy can erode its own equity down to
    near zero, making its later daily costs tiny in dollar terms simply
    because there is almost no capital left to charge a fee against -- not
    because it became cheap to trade. The rate-based comparison below (mean
    return drag per day) is unaffected by that compounding artifact.
    """

    prices = generate_random_walk_prices(periods=STATISTICAL_TEST_PERIODS)

    iid_signal = random_entry_signal(prices.index, seed=SEED, probability=0.5)
    persistent_signal = random_entry_persistent_signal(
        prices.index, seed=SEED, probability=0.5, avg_holding_days=20.0
    )

    def _mean_daily_cost_drag(signal: pd.Series) -> float:
        gross = run_backtest(
            prices["close"], signal, fee_bps=0.0, slippage_bps=0.0,
            initial_capital=INITIAL_CAPITAL,
        )
        net = run_backtest(
            prices["close"], signal, fee_bps=TRANSACTION_FEE_BPS,
            slippage_bps=SLIPPAGE_BPS, initial_capital=INITIAL_CAPITAL,
        )
        return float(gross.net_returns.mean() - net.net_returns.mean())

    iid_drag = _mean_daily_cost_drag(iid_signal)
    persistent_drag = _mean_daily_cost_drag(persistent_signal)

    assert persistent_drag < iid_drag
    # Turnover ratio is ~10x (0.5 vs 0.05), so the drag ratio should be in
    # the same ballpark, not just marginally smaller.
    assert persistent_drag < iid_drag / 3


def test_sma_crossover_rejects_fast_window_not_smaller_than_slow():
    close = pd.Series([100.0] * 60, index=pd.bdate_range("2020-01-01", periods=60))

    with pytest.raises(ValueError, match="fast_window"):
        moving_average_crossover_signal(close, fast_window=50, slow_window=20)


def test_sma_crossover_is_flat_during_warmup():
    """Before slow_window observations exist, the slow SMA is NaN and the
    comparison must resolve to flat (0.0), not raise or propagate NaN into
    the signal (which would corrupt the backtest's cost/turnover math).
    """

    index = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(range(100, 110), index=index, dtype=float)

    signal = moving_average_crossover_signal(close, fast_window=3, slow_window=8)

    assert not signal.isna().any()
    assert (signal.iloc[:7] == 0.0).all()  # slow SMA needs 8 observations


def test_sma_crossover_has_no_lookahead():
    """A late price spike must not retroactively change the SMA values (and
    therefore the signal) on earlier days -- rolling means are trailing by
    construction, this locks that in as a regression test.
    """

    index = pd.bdate_range("2020-01-01", periods=10)
    close_without_spike = pd.Series([100.0] * 10, index=index)
    close_with_late_spike = close_without_spike.copy()
    close_with_late_spike.iloc[-1] = 1000.0

    signal_without_spike = moving_average_crossover_signal(
        close_without_spike, fast_window=3, slow_window=8
    )
    signal_with_late_spike = moving_average_crossover_signal(
        close_with_late_spike, fast_window=3, slow_window=8
    )

    # Every day except the very last (where the spike itself lives) must be
    # identical regardless of the future spike.
    pd.testing.assert_series_equal(
        signal_without_spike.iloc[:-1], signal_with_late_spike.iloc[:-1]
    )


def test_sma_crossover_on_synthetic_asset_has_no_significant_edge():
    """Statistical gate required before any predictive model is introduced:
    a rule that actually reacts to price (unlike buy-and-hold) must still
    show no significant edge on an asset with no real structure. This is a
    stronger check on the engine than buy-and-hold, since crossover signals
    are exactly the kind of rule that could reveal a subtle lookahead bug.
    """

    prices = generate_random_walk_prices(periods=STATISTICAL_TEST_PERIODS)
    signal = moving_average_crossover_signal(prices["close"], fast_window=20, slow_window=50)

    result = run_backtest(
        prices["close"], signal, fee_bps=0.0, slippage_bps=0.0, initial_capital=INITIAL_CAPITAL,
    )

    returns = result.net_returns.iloc[1:].to_numpy()
    t_statistic = newey_west_t_stat(returns)

    assert abs(t_statistic) < 1.96


def test_engine_supports_fractional_position_sizing():
    """The engine has always accepted arbitrary float signals, but until now
    that was never actually tested -- every baseline used only 0.0/1.0. This
    matters because Stage 4 (fractional Kelly sizing) depends entirely on
    fractional positions being handled correctly, not just binary ones.
    """

    index = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0, 110.0, 121.0, 108.9], index=index)  # +10%, +10%, -10%
    signal = pd.Series([0.5, 0.5, 0.5, 0.5], index=index)  # constant half position

    result = run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0)

    # day0: no prior position (shift gives 0) -> 0 return.
    # day1/day2: half-exposed to +10% each day -> +5% net return each day.
    # day3: half-exposed to -10% -> -5% net return.
    expected_returns = [0.0, 0.05, 0.05, -0.05]
    assert result.net_returns.to_numpy() == pytest.approx(expected_returns, abs=1e-9)

    expected_total_return = 1.0 * 1.05 * 1.05 * 0.95 - 1.0
    assert result.total_return == pytest.approx(expected_total_return, abs=1e-9)


def test_engine_fractional_sizing_scales_cost_proportionally_to_change_magnitude():
    """Cost must scale with HOW MUCH the position changed, not just whether
    it changed -- a move of 0.2 should cost 1/5th of a move of 1.0, given
    the same cost rate.
    """

    index = pd.bdate_range("2020-01-01", periods=3)
    close = pd.Series([100.0, 100.0, 100.0], index=index)

    # Single position change of magnitude 0.2 on day 1, then held flat.
    small_change_signal = pd.Series([0.2, 0.2, 0.2], index=index)
    # Single position change of magnitude 1.0 on day 1, then held flat.
    large_change_signal = pd.Series([1.0, 1.0, 1.0], index=index)

    small_result = run_backtest(
        close, small_change_signal, fee_bps=10.0, slippage_bps=0.0, initial_capital=100_000.0
    )
    large_result = run_backtest(
        close, large_change_signal, fee_bps=10.0, slippage_bps=0.0, initial_capital=100_000.0
    )

    assert small_result.total_costs_paid < large_result.total_costs_paid
    assert large_result.total_costs_paid == pytest.approx(
        small_result.total_costs_paid * 5, rel=1e-6
    )


def test_engine_raises_on_internal_nan_in_close():
    """Regression test for a bug found while auditing an external agent's
    "fix": a check for internal NaN in close was added AFTER
    close.pct_change().fillna(0.0), which already erases every NaN
    (including internal data gaps, not just the expected position-0 NaN).
    That made the check permanently unreachable -- verified empirically by
    planting a NaN and confirming no error was raised, when it should have
    been. The check must run on the raw pct_change, before fillna.
    """

    index = pd.bdate_range("2020-01-01", periods=6)
    close = pd.Series([100.0, 101.0, float("nan"), 103.0, 104.0, 105.0], index=index)
    signal = pd.Series([1.0] * 6, index=index)

    with pytest.raises(ValueError, match="internal NaN"):
        run_backtest(close, signal, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0)


def test_total_costs_paid_matches_actual_decline_in_real_equity():
    """Regression test for a bug found while auditing an external agent's
    "fix": total_costs_paid was changed to scale cost by a hypothetical
    gross (pre-cost) equity path instead of the real (net, post-cost)
    equity the account actually had. With zero market return, ALL equity
    decline is attributable to costs, so total_costs_paid must exactly
    equal initial_capital minus the final real equity -- the gross-based
    version overstated this by $59.60 on a simple 5-day, 4-trade example
    (reported $4000.00 in costs when the account only ever lost $3940.40).
    """

    index = pd.bdate_range("2020-01-01", periods=5)
    close = pd.Series([100.0] * 5, index=index)  # zero market return -- isolates cost effect
    signal = pd.Series([1.0, 0.0, 1.0, 0.0, 1.0], index=index)

    result = run_backtest(close, signal, fee_bps=100.0, slippage_bps=0.0, initial_capital=100_000.0)

    actual_decline = 100_000.0 - result.equity_curve.iloc[-1]
    assert result.total_costs_paid == pytest.approx(actual_decline, abs=1e-6)
