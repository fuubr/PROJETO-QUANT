import numpy as np
import pandas as pd
import pytest

from backtest.engine import run_backtest
from risk.managed_backtest import run_risk_managed_backtest


def test_managed_backtest_matches_plain_engine_when_risk_never_triggers():
    """Consistency check: with stop-loss/circuit-breaker thresholds set so
    loose they never fire, the sequential risk-managed engine should
    produce the same result as the plain vectorized engine for an
    identical (binary 0/1) target position -- confirms the sequential loop
    isn't silently doing something different from the established,
    already-validated vectorized logic.
    """

    index = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))), index=index)
    target_position = pd.Series(1.0, index=index)  # buy and hold

    plain_result = run_backtest(
        close, target_position, fee_bps=5.0, slippage_bps=0.0, initial_capital=100_000.0
    )
    managed_result = run_risk_managed_backtest(
        close, target_position, fee_bps=5.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.99, circuit_breaker_drawdown_pct=0.99, var_confidence_level=0.95,
    )

    assert managed_result.total_return == pytest.approx(plain_result.total_return, rel=1e-9)
    assert managed_result.max_drawdown == pytest.approx(plain_result.max_drawdown, rel=1e-9)
    assert managed_result.stop_loss_triggers == 0
    assert managed_result.circuit_breaker_triggered is False


def test_stop_loss_forces_exit_after_threshold_breach():
    index = pd.bdate_range("2020-01-01", periods=6)
    # 100 -> 100 -> 89 (an 11% single-day drop, breaching a 10% stop) -> flat
    close = pd.Series([100.0, 100.0, 89.0, 89.0, 89.0, 89.0], index=index)
    target_position = pd.Series(1.0, index=index)  # always wants to be invested

    result = run_risk_managed_backtest(
        close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.10, circuit_breaker_drawdown_pct=0.99, var_confidence_level=0.95,
    )

    assert result.stop_loss_triggers >= 1
    # Position is forced flat on the day the stop-loss fires (day index 3,
    # the first day after the entry-relative loss breached -10%).
    # Stop-loss is a per-position exit, not a lockout: since the target
    # position still wants exposure and the fresh re-entry hasn't lost
    # anything yet, the position may re-open the following day -- that is
    # expected stop-loss behavior (distinct from the circuit breaker's
    # permanent halt, tested separately below).
    assert result.actual_position.iloc[3] == 0.0


def test_circuit_breaker_halts_trading_permanently():
    index = pd.bdate_range("2020-01-01", periods=10)
    # Price collapses hard, then recovers strongly -- circuit breaker
    # should keep the strategy flat through the recovery too (no re-arm).
    close = pd.Series(
        [100.0, 90.0, 75.0, 70.0, 70.0, 70.0, 90.0, 110.0, 130.0, 150.0], index=index
    )
    target_position = pd.Series(1.0, index=index)

    result = run_risk_managed_backtest(
        close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.99, circuit_breaker_drawdown_pct=0.20, var_confidence_level=0.95,
    )

    assert result.circuit_breaker_triggered is True
    assert result.circuit_breaker_date is not None
    # The breaker fires based on TODAY's drawdown, after today's return has
    # already been realized -- it can only prevent TOMORROW's position
    # onward, not retroactively undo today's. Check strictly after the
    # trigger day.
    trigger_loc = close.index.get_loc(result.circuit_breaker_date)
    assert (result.actual_position.iloc[trigger_loc + 1:] == 0.0).all()


def test_managed_backtest_rejects_mismatched_index():
    close = pd.Series([100.0, 101.0], index=pd.bdate_range("2020-01-01", periods=2))
    target_position = pd.Series([1.0, 1.0], index=pd.bdate_range("2020-01-02", periods=2))

    with pytest.raises(ValueError, match="index"):
        run_risk_managed_backtest(
            close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
            stop_loss_pct=0.1, circuit_breaker_drawdown_pct=0.2, var_confidence_level=0.95,
        )


def test_managed_backtest_rejects_invalid_thresholds():
    close = pd.Series([100.0, 101.0], index=pd.bdate_range("2020-01-01", periods=2))
    target_position = pd.Series([1.0, 1.0], index=close.index)

    with pytest.raises(ValueError, match="stop_loss_pct"):
        run_risk_managed_backtest(
            close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
            stop_loss_pct=1.5, circuit_breaker_drawdown_pct=0.2, var_confidence_level=0.95,
        )
    with pytest.raises(ValueError, match="circuit_breaker_drawdown_pct"):
        run_risk_managed_backtest(
            close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
            stop_loss_pct=0.1, circuit_breaker_drawdown_pct=-0.1, var_confidence_level=0.95,
        )


def test_managed_backtest_reports_var_and_volatility():
    index = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(1)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))), index=index)
    target_position = pd.Series(1.0, index=index)

    result = run_risk_managed_backtest(
        close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.5, circuit_breaker_drawdown_pct=0.9, var_confidence_level=0.95,
    )

    assert result.historical_var_95 >= 0.0
    assert result.annualized_volatility > 0.0


def test_entry_price_updates_with_weighted_average_when_position_grows():
    """Regression coverage for a real gap found during audit: with Kelly's
    continuous sizing, a position that grows without ever passing through
    zero should have its stop-loss reference move to a blended cost basis,
    not stay pinned to the original (much smaller) entry.

    Position: 0.5 at entry price 100, then grows to 1.0 at price 120.
    Weighted average entry = (0.5*100 + 0.5*120) / 1.0 = 110.
    A subsequent price of 100 is a -9.1% move from 110 (not a -16.7% move
    from the original 100), so a 10% stop-loss should NOT fire yet.
    """

    index = pd.bdate_range("2020-01-01", periods=5)
    close = pd.Series([100.0, 100.0, 120.0, 100.0, 100.0], index=index)
    # day0->day1: enter at 0.5; day1->day2: grow to 1.0 (add 0.5 at price 120)
    target_position = pd.Series([0.5, 0.5, 1.0, 1.0, 1.0], index=index)

    result = run_risk_managed_backtest(
        close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.10, circuit_breaker_drawdown_pct=0.99, var_confidence_level=0.95,
    )

    # At day 3 (price=100, blended entry=110): loss = (100-110)/110 = -9.1%,
    # inside the 10% stop -- position should NOT be forced flat yet.
    assert result.actual_position.iloc[3] > 0.0


def test_entry_price_keeps_original_basis_when_position_shrinks():
    """A position that shrinks (but stays positive) keeps its original
    entry price -- selling down doesn't change the cost basis of the
    shares still held.
    """

    index = pd.bdate_range("2020-01-01", periods=6)
    # Enter at 100, price drops to 89 on day 4, held flat at 89 on day 5
    # (stop-loss checks yesterday's close, so it needs a following day to
    # actually detect and act on the drop -- same timing convention as the
    # circuit breaker).
    close = pd.Series([100.0, 100.0, 100.0, 100.0, 89.0, 89.0], index=index)
    # day0->day1: enter at 1.0; day1->day2: shrink to 0.3 (still positive)
    target_position = pd.Series([1.0, 0.3, 0.3, 0.3, 0.3, 0.3], index=index)

    result = run_risk_managed_backtest(
        close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.10, circuit_breaker_drawdown_pct=0.99, var_confidence_level=0.95,
    )

    # Entry price should still be the original 100 (not reset by shrinking),
    # so the -11% move (89 vs entry 100) should trigger the stop on the day
    # after the drop is reflected in yesterday's close.
    assert result.stop_loss_triggers >= 1


def test_active_trading_days_equals_full_period_when_breaker_never_fires():
    index = pd.bdate_range("2020-01-01", periods=50)
    close = pd.Series(100.0 + np.arange(50) * 0.1, index=index)  # steady, mild uptrend
    target_position = pd.Series(1.0, index=index)

    result = run_risk_managed_backtest(
        close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.99, circuit_breaker_drawdown_pct=0.99, var_confidence_level=0.95,
    )

    assert result.active_trading_days == result.total_days == 50


def test_active_trading_days_stops_at_breaker_trigger():
    index = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(
        [100.0, 90.0, 75.0, 70.0, 70.0, 70.0, 90.0, 110.0, 130.0, 150.0], index=index
    )
    target_position = pd.Series(1.0, index=index)

    result = run_risk_managed_backtest(
        close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.99, circuit_breaker_drawdown_pct=0.20, var_confidence_level=0.95,
    )

    assert result.circuit_breaker_triggered is True
    assert result.active_trading_days < result.total_days
    expected_active_days = close.index.get_loc(result.circuit_breaker_date)
    assert result.active_trading_days == expected_active_days


def test_asset_level_circuit_breaker_catches_diluted_exposure_blind_spot():
    """Regression test for a real structural gap found during development
    (PETR4.SA real result): a diluted (e.g. Kelly-sized) position needs the
    UNDERLYING ASSET to fall much further than a fully-exposed position
    before the same PORTFOLIO drawdown threshold trips. Confirmed with a
    controlled example: a steady -1.5%/day decline tripped a fully-exposed
    (1.0) portfolio breaker after the asset fell ~20%, but a diluted (0.2)
    position needed the asset to fall ~68% before its portfolio drawdown
    reached the same 20% -- letting it stay exposed through a much deeper,
    still-ongoing crash. Setting asset_circuit_breaker_drawdown_pct closes
    that blind spot.
    """

    n = 100
    index = pd.bdate_range("2018-05-01", periods=n)
    daily_decline = -0.015
    close = pd.Series(100 * (1 + daily_decline) ** np.arange(n), index=index)
    diluted_position = pd.Series(0.2, index=index)

    without_asset_breaker = run_risk_managed_backtest(
        close, diluted_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.99, circuit_breaker_drawdown_pct=0.20, var_confidence_level=0.95,
    )
    with_asset_breaker = run_risk_managed_backtest(
        close, diluted_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.99, circuit_breaker_drawdown_pct=0.20, var_confidence_level=0.95,
        asset_circuit_breaker_drawdown_pct=0.20,
    )

    assert without_asset_breaker.circuit_breaker_reason == "portfolio"
    assert with_asset_breaker.circuit_breaker_reason == "asset"
    # The asset-level breaker should trip much earlier, once the asset
    # itself (not the diluted portfolio) has fallen 20%.
    assert with_asset_breaker.active_trading_days < without_asset_breaker.active_trading_days


def test_asset_circuit_breaker_disabled_by_default():
    n = 100
    index = pd.bdate_range("2018-05-01", periods=n)
    close = pd.Series(100 * (0.985) ** np.arange(n), index=index)
    target_position = pd.Series(0.2, index=index)

    result = run_risk_managed_backtest(
        close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
        stop_loss_pct=0.99, circuit_breaker_drawdown_pct=0.20, var_confidence_level=0.95,
    )

    # Without opting in, only the portfolio-level breaker can fire.
    assert result.circuit_breaker_reason in (None, "portfolio")


def test_asset_circuit_breaker_rejects_invalid_threshold():
    close = pd.Series([100.0, 99.0], index=pd.bdate_range("2020-01-01", periods=2))
    target_position = pd.Series([1.0, 1.0], index=close.index)

    with pytest.raises(ValueError, match="asset_circuit_breaker_drawdown_pct"):
        run_risk_managed_backtest(
            close, target_position, fee_bps=0.0, slippage_bps=0.0, initial_capital=100_000.0,
            stop_loss_pct=0.1, circuit_breaker_drawdown_pct=0.2, var_confidence_level=0.95,
            asset_circuit_breaker_drawdown_pct=1.5,
        )


def test_step_by_step_matches_batch_result():
    """The core promise of the RiskState/step refactor: calling
    step_risk_managed_backtest once per day (as Stage 6's daily paper
    trading script will) must produce EXACTLY the same result as the batch
    run_risk_managed_backtest looping internally -- otherwise paper trading
    and backtesting could silently diverge from day one.
    """

    from risk.managed_backtest import RiskState, step_risk_managed_backtest

    index = pd.bdate_range("2020-01-01", periods=60)
    rng = np.random.default_rng(3)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0002, 0.015, 60))), index=index)
    target_position = pd.Series(rng.uniform(0.0, 0.6, 60), index=index)

    batch_result = run_risk_managed_backtest(
        close, target_position, fee_bps=5.0, slippage_bps=2.0, initial_capital=100_000.0,
        stop_loss_pct=0.10, circuit_breaker_drawdown_pct=0.20, var_confidence_level=0.95,
    )

    prices = close.to_numpy()
    target = target_position.to_numpy()
    state = RiskState.initial(100_000.0, prices[0])
    manual_equity = [100_000.0]
    for t in range(1, len(prices)):
        state, _ = step_risk_managed_backtest(
            state, price_yesterday=prices[t - 1], price_today=prices[t],
            target_position_yesterday=float(target[t - 1]), fee_bps=5.0, slippage_bps=2.0,
            stop_loss_pct=0.10, circuit_breaker_drawdown_pct=0.20, today_date=close.index[t],
        )
        manual_equity.append(state.equity)

    assert manual_equity == pytest.approx(batch_result.equity_curve.to_numpy().tolist())
    assert state.stop_loss_triggers == batch_result.stop_loss_triggers
    assert state.circuit_breaker_triggered == batch_result.circuit_breaker_triggered
