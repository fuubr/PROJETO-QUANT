import numpy as np
import pandas as pd
import pytest

from risk.kelly import estimate_win_loss_ratio, kelly_fraction, kelly_position_series


def test_kelly_fraction_matches_known_textbook_value():
    # Classic example: p=0.6, b=1 (even-money bet) -> f* = 0.6 - 0.4/1 = 0.2
    assert kelly_fraction(0.6, 1.0) == pytest.approx(0.2)


def test_kelly_fraction_is_zero_at_no_edge():
    # p=0.5 with b=1 (symmetric win/loss) -> f* = 0.5 - 0.5/1 = 0.0
    assert kelly_fraction(0.5, 1.0) == pytest.approx(0.0)


def test_kelly_fraction_negative_raw_value_clips_to_zero():
    # p=0.3, b=1 -> raw f* = 0.3 - 0.7/1 = -0.4, clipped to 0 (no shorting)
    assert kelly_fraction(0.3, 1.0) == pytest.approx(0.0)


def test_kelly_fraction_never_exceeds_win_probability():
    """Mathematical property of this formula, not just an implementation
    detail: f* = p - (1-p)/b subtracts a non-negative term from p, so
    f* <= p <= 1 always -- the upper clip in the implementation is a
    defensive safeguard that this specific formula can never actually
    trigger, not a case this test can force via p and b alone.
    """

    for p in (0.5, 0.7, 0.9, 0.99, 1.0):
        for b in (0.01, 0.1, 1.0, 10.0, 1000.0):
            assert kelly_fraction(p, b) <= p + 1e-9


def test_kelly_fraction_at_certainty_with_tiny_edge_needed():
    # p=1.0 (certain win) -> f* = 1.0 - 0/b = 1.0 regardless of b.
    assert kelly_fraction(1.0, 0.01) == pytest.approx(1.0)


def test_kelly_fraction_handles_degenerate_win_loss_ratio():
    assert kelly_fraction(0.7, 0.0) == 0.0
    assert kelly_fraction(0.7, -1.0) == 0.0


def test_kelly_fraction_rejects_invalid_probability():
    with pytest.raises(ValueError, match="win_probability"):
        kelly_fraction(1.5, 1.0)


def test_estimate_win_loss_ratio_matches_manual_calculation():
    returns = pd.Series([0.10, 0.10, -0.05, -0.05, 0.10])
    # wins: [0.10, 0.10, 0.10] mean=0.10; losses: [-0.05,-0.05] mean=-0.05
    # b = 0.10 / 0.05 = 2.0
    ratio = estimate_win_loss_ratio(returns, min_win_observations=2, min_loss_observations=2)
    assert ratio == pytest.approx(2.0)


def test_estimate_win_loss_ratio_returns_none_with_too_few_observations():
    returns = pd.Series([0.10, -0.05])
    assert estimate_win_loss_ratio(returns, min_win_observations=5, min_loss_observations=5) is None


def test_estimate_win_loss_ratio_ignores_nan():
    returns = pd.Series([0.10, 0.10, np.nan, -0.05, -0.05])
    ratio = estimate_win_loss_ratio(returns, min_win_observations=2, min_loss_observations=2)
    assert ratio == pytest.approx(2.0)


def test_kelly_position_series_has_no_lookahead():
    """A future price shock must not change Kelly-sized positions on
    earlier days -- the win/loss ratio estimate at day t must only use
    data up to and including day t.
    """

    index = pd.bdate_range("2020-01-01", periods=400)
    rng = np.random.default_rng(0)
    close_without_shock = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=index)
    close_with_late_shock = close_without_shock.copy()
    close_with_late_shock.iloc[-1] *= 3

    probability = pd.Series(0.6, index=index[300:])  # constant confident prediction

    position_without_shock = kelly_position_series(
        close_without_shock, probability, horizon_days=20, kelly_fraction_multiplier=0.5,
        win_loss_window_days=100, min_win_observations=5, min_loss_observations=5,
    )
    position_with_late_shock = kelly_position_series(
        close_with_late_shock, probability, horizon_days=20, kelly_fraction_multiplier=0.5,
        win_loss_window_days=100, min_win_observations=5, min_loss_observations=5,
    )

    pd.testing.assert_series_equal(
        position_without_shock.iloc[:-1], position_with_late_shock.iloc[:-1]
    )


def test_kelly_position_series_defaults_to_flat_without_enough_data():
    index = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(100.0 + np.arange(10) * 0.1, index=index)
    probability = pd.Series(0.9, index=index)  # very confident, but too little history

    position = kelly_position_series(
        close, probability, horizon_days=20, kelly_fraction_multiplier=0.5,
        win_loss_window_days=100, min_win_observations=20, min_loss_observations=20,
    )

    assert (position == 0.0).all()


def test_kelly_position_series_rejects_invalid_multiplier():
    index = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(100.0, index=index)
    probability = pd.Series(0.6, index=index)

    with pytest.raises(ValueError, match="kelly_fraction_multiplier"):
        kelly_position_series(
            close, probability, horizon_days=5, kelly_fraction_multiplier=1.5,
            win_loss_window_days=10, min_win_observations=2, min_loss_observations=2,
        )


def test_estimate_win_loss_ratio_shrinks_toward_one_with_small_effective_sample():
    """With horizon_days provided, a small effective sample (few
    independent overlapping-window observations) should pull the estimate
    toward 1.0 (neutral), even if the raw sample ratio is far from 1.0 --
    reduces day-to-day whiplash in Kelly-sized bets caused by noisy
    small-sample win/loss estimates.
    """

    # 40 observations with horizon_days=20 -> effective_n = 40/20 = 2, tiny.
    returns = pd.Series([0.20] * 20 + [-0.02] * 20)  # raw b would be 10.0

    raw_ratio = estimate_win_loss_ratio(returns, min_win_observations=5, min_loss_observations=5)
    shrunk_ratio = estimate_win_loss_ratio(
        returns, min_win_observations=5, min_loss_observations=5, horizon_days=20
    )

    assert raw_ratio == pytest.approx(10.0)
    assert shrunk_ratio < raw_ratio
    assert shrunk_ratio < 3.0  # pulled substantially toward 1.0


def test_estimate_win_loss_ratio_converges_to_raw_with_large_effective_sample():
    """With a large effective sample (many independent windows), shrinkage
    should have little effect -- the estimate should stay close to the raw
    sample ratio.
    """

    # 2000 observations with horizon_days=20 -> effective_n = 100, large.
    returns = pd.Series([0.20] * 1000 + [-0.10] * 1000)  # raw b = 2.0

    raw_ratio = estimate_win_loss_ratio(returns, min_win_observations=5, min_loss_observations=5)
    shrunk_ratio = estimate_win_loss_ratio(
        returns, min_win_observations=5, min_loss_observations=5, horizon_days=20
    )

    assert raw_ratio == pytest.approx(2.0)
    assert shrunk_ratio == pytest.approx(2.0, rel=0.1)


def test_estimate_win_loss_ratio_shrinkage_is_opt_in():
    """Without horizon_days, behavior is unchanged from before shrinkage
    was added -- backward compatible with existing callers/tests.
    """

    returns = pd.Series([0.10, 0.10, -0.05, -0.05])
    ratio_no_horizon = estimate_win_loss_ratio(returns, min_win_observations=2, min_loss_observations=2)
    assert ratio_no_horizon == pytest.approx(2.0)
