import numpy as np
import pandas as pd
import pytest

from backtest.regime import segment_returns_by_regime, summarize_regime_segment


def test_segment_returns_by_regime_splits_crisis_and_bull():
    index = pd.bdate_range("2018-01-01", periods=400)
    returns = pd.Series(0.001, index=index)

    crisis_periods = [("2018-10-01", "2018-12-31")]
    segments = segment_returns_by_regime(returns, crisis_periods)

    assert "bull" in segments
    assert "crisis_1_2018-10" in segments
    crisis_dates = segments["crisis_1_2018-10"].index
    assert not crisis_dates.isin(segments["bull"].index).any()
    assert len(segments["bull"]) + len(segments["crisis_1_2018-10"]) == len(returns)


def test_segment_returns_by_regime_handles_no_overlap():
    """A crisis window entirely outside the series' date range (e.g. the
    synthetic control, which has no real calendar dates tied to real
    events) must produce an empty segment, not an error or a silently
    dropped key.
    """

    index = pd.bdate_range("2050-01-01", periods=50)
    returns = pd.Series(0.001, index=index)

    segments = segment_returns_by_regime(returns, [("2018-10-01", "2018-12-31")])

    assert len(segments["crisis_1_2018-10"]) == 0
    assert len(segments["bull"]) == 50


def test_summarize_regime_segment_matches_manual_compounding():
    returns = pd.Series([0.01, -0.02, 0.03])
    summary = summarize_regime_segment(returns)

    expected_total = (1.01 * 0.98 * 1.03) - 1.0
    assert summary["total_return"] == pytest.approx(expected_total, rel=1e-3)
    assert summary["days"] == 3


def test_summarize_regime_segment_handles_empty_segment():
    summary = summarize_regime_segment(pd.Series(dtype=float))

    assert summary["days"] == 0
    assert np.isnan(summary["total_return"])
