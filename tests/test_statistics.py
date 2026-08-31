import numpy as np
import pandas as pd
import pytest

from backtest.statistics import (
    bonferroni_alpha,
    mean_return_significance,
    newey_west_t_stat,
    two_sided_p_value,
)


def test_bonferroni_alpha_scales_with_num_tests():
    assert bonferroni_alpha(0.05, 1) == pytest.approx(0.05)
    assert bonferroni_alpha(0.05, 20) == pytest.approx(0.0025)


def test_bonferroni_alpha_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="num_tests"):
        bonferroni_alpha(0.05, 0)
    with pytest.raises(ValueError, match="alpha"):
        bonferroni_alpha(1.5, 10)


def test_two_sided_p_value_matches_known_normal_reference():
    # t = 1.96 is the classic ~5% two-sided threshold under the normal
    # approximation.
    assert two_sided_p_value(1.96) == pytest.approx(0.05, abs=0.001)
    assert two_sided_p_value(0.0) == pytest.approx(1.0)


def test_newey_west_matches_naive_se_when_returns_are_iid():
    """On truly i.i.d. returns (no autocorrelation), Newey-West and the
    naive t-test should be close -- HAC correction should not distort a
    case where it isn't needed.
    """

    rng = np.random.default_rng(42)
    returns = rng.normal(loc=0.0, scale=0.01, size=20_000)

    naive_se = returns.std(ddof=1) / np.sqrt(len(returns))
    naive_t = returns.mean() / naive_se

    hac_t = newey_west_t_stat(returns)

    assert hac_t == pytest.approx(naive_t, rel=0.15)


def test_newey_west_reduces_false_positive_rate_under_positive_autocorrelation():
    """Construct returns with strong positive autocorrelation (e.g. from a
    persistent signal or overlapping trades) where the true mean is exactly
    zero. The naive t-test should be inflated (overconfident, prone to false
    positives); Newey-West should pull the t-statistic back down.
    """

    rng = np.random.default_rng(7)
    n = 20_000
    noise = rng.normal(loc=0.0, scale=0.01, size=n)

    # AR(1)-style positive autocorrelation with zero unconditional mean.
    autocorrelated = np.empty(n)
    autocorrelated[0] = noise[0]
    phi = 0.6
    for t in range(1, n):
        autocorrelated[t] = phi * autocorrelated[t - 1] + noise[t]

    naive_se = autocorrelated.std(ddof=1) / np.sqrt(n)
    naive_t = autocorrelated.mean() / naive_se

    hac_t = newey_west_t_stat(autocorrelated)

    # The naive test understates the true standard error under positive
    # autocorrelation, so it should show a larger-magnitude (more
    # "significant"-looking) statistic than the HAC-corrected one.
    assert abs(hac_t) < abs(naive_t)


def test_mean_return_significance_applies_bonferroni_correction():
    index = pd.bdate_range("2020-01-01", periods=5000)
    rng = np.random.default_rng(1)
    # Small but nonzero mean, sized to be naively "significant" without
    # correction but not after a strict Bonferroni correction.
    returns = pd.Series(rng.normal(loc=0.0006, scale=0.01, size=5000), index=index)

    single_test = mean_return_significance(returns, num_tests=1)
    many_tests = mean_return_significance(returns, num_tests=1000)

    assert single_test["bonferroni_alpha"] > many_tests["bonferroni_alpha"]
    # Same underlying data, but the corrected threshold is 1000x stricter.
    assert many_tests["bonferroni_alpha"] == pytest.approx(single_test["bonferroni_alpha"] / 1000)


def test_newey_west_t_stat_requires_at_least_two_observations():
    with pytest.raises(ValueError, match="observations"):
        newey_west_t_stat(np.array([0.01]))
