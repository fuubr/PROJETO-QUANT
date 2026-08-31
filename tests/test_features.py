import numpy as np
import pandas as pd
import pytest

from models.features import build_features, check_feature_correlation


def test_build_features_has_no_lookahead():
    """A future price spike must not change feature values on earlier days
    -- every feature here uses trailing windows exclusively.
    """

    index = pd.bdate_range("2020-01-01", periods=100)
    rng = np.random.default_rng(0)
    close_without_spike = pd.Series(100 + np.cumsum(rng.normal(0, 1, 100)), index=index)
    close_with_late_spike = close_without_spike.copy()
    close_with_late_spike.iloc[-1] = close_with_late_spike.iloc[-1] * 3

    features_without_spike = build_features(
        close_without_spike, momentum_windows=(5, 20, 60), volatility_window=20,
        sma_distance_window=50,
    )
    features_with_late_spike = build_features(
        close_with_late_spike, momentum_windows=(5, 20, 60), volatility_window=20,
        sma_distance_window=50,
    )

    pd.testing.assert_frame_equal(
        features_without_spike.iloc[:-1], features_with_late_spike.iloc[:-1]
    )


def test_build_features_warmup_is_nan_not_zero():
    """Insufficient trailing history must produce NaN, not a silently wrong
    zero -- a zero would look like "no momentum" instead of "unknown",
    which could bias a model trained on it.
    """

    index = pd.bdate_range("2020-01-01", periods=10)
    close = pd.Series(range(100, 110), index=index, dtype=float)

    features = build_features(
        close, momentum_windows=(5,), volatility_window=5, sma_distance_window=8,
    )

    assert features["momentum_5"].iloc[:5].isna().all()
    assert features["sma_distance_8"].iloc[:7].isna().all()
    assert not features["sma_distance_8"].iloc[7:].isna().any()


def test_build_features_rejects_empty_momentum_windows():
    close = pd.Series([100.0] * 20, index=pd.bdate_range("2020-01-01", periods=20))

    with pytest.raises(ValueError, match="momentum_windows"):
        build_features(close, momentum_windows=(), volatility_window=5, sma_distance_window=8)


def test_check_feature_correlation_flags_known_correlated_pair():
    """Regression/confirmation test for a real finding: momentum_20 and
    sma_distance_50 are highly correlated (~0.88) on realistic price data,
    since both measure similar price-trend information over overlapping
    windows. This matters for interpreting SHAP importance and logistic
    regression coefficients.
    """

    index = pd.bdate_range("2020-01-01", periods=2000)
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 2000))), index=index)

    features = build_features(
        close, momentum_windows=(5, 20, 60), volatility_window=20, sma_distance_window=50,
    ).dropna()

    flagged = check_feature_correlation(features, threshold=0.7)

    assert not flagged.empty
    pair_names = set(flagged["feature_a"]) | set(flagged["feature_b"])
    assert "momentum_20" in pair_names
    assert "sma_distance_50" in pair_names


def test_check_feature_correlation_empty_when_threshold_very_high():
    index = pd.bdate_range("2020-01-01", periods=500)
    rng = np.random.default_rng(1)
    features = pd.DataFrame(
        {
            "independent_a": rng.normal(size=500),
            "independent_b": rng.normal(size=500),
        },
        index=index,
    )

    flagged = check_feature_correlation(features, threshold=0.99)

    assert flagged.empty


def test_check_feature_correlation_rejects_invalid_threshold():
    features = pd.DataFrame({"a": [1.0, 2.0, 3.0]})

    with pytest.raises(ValueError, match="threshold"):
        check_feature_correlation(features, threshold=1.5)
