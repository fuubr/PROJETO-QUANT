import numpy as np
import pandas as pd
import pytest

from backtest.deflated_sharpe import deflated_sharpe_ratio, expected_max_sharpe, probabilistic_sharpe


def test_expected_max_sharpe_is_zero_with_a_single_trial():
    assert expected_max_sharpe(n_trials=1, sr_variance=1.0) == 0.0


def test_expected_max_sharpe_matches_the_known_value_for_100_trials():
    # (1-g)*z(0.99) + g*z(1 - 1/(100e)), g = Euler-Mascheroni 0.5772:
    # 0.4228*2.3263 + 0.5772*2.6801 = 2.53 -- the best of 100 zero-skill
    # strategies is expected to show ~2.53 standard deviations of Sharpe.
    assert expected_max_sharpe(n_trials=100, sr_variance=1.0) == pytest.approx(2.53, abs=0.01)


def test_probabilistic_sharpe_worked_example_for_normal_returns():
    # SR = 0.1/period, T = 101, normal (skew 0, kurtosis 3), benchmark 0:
    # z = 0.1 * sqrt(100) / sqrt(1 + 0.1^2 / 2) = 0.99751 -> Phi(z) = 0.8407
    assert probabilistic_sharpe(0.1, 101, skew=0.0, kurtosis=3.0, sr_benchmark=0.0) == pytest.approx(0.8407, abs=1e-3)


def test_more_trials_deflate_the_same_track_record():
    returns = pd.Series(np.random.default_rng(1).normal(0.0006, 0.01, 1000))
    values = [deflated_sharpe_ratio(returns, n_trials=n, sr_variance=0.002) for n in (1, 10, 100)]
    assert values[0] > values[1] > values[2]
    assert 0.0 <= values[2] <= values[0] <= 1.0


from backtest.deflated_sharpe import memmel_z, sharpe_difference_test


def test_memmel_worked_example():
    # SR1=0.1, SR2=0.05, rho=0.5, T=1000:
    # V = (1/1000) * (2*(1-0.5) + 0.5*(0.01 + 0.0025 - 2*0.1*0.05*0.25)) = 0.001005
    # z = 0.05 / sqrt(0.001005) = 1.5772
    assert memmel_z(0.1, 0.05, rho=0.5, t=1000) == pytest.approx(1.5772, abs=1e-3)


def test_identical_strategies_have_no_difference():
    r = pd.Series(np.random.default_rng(2).normal(0.0005, 0.01, 500))
    z, p = sharpe_difference_test(r, r)
    assert z == 0.0 and p == 1.0


def test_a_clearly_better_strategy_is_detected():
    rng = np.random.default_rng(3)
    base = pd.Series(rng.normal(0.0, 0.01, 3000))
    better = base * 0.5 + 0.002 + pd.Series(rng.normal(0, 0.002, 3000))
    z, p = sharpe_difference_test(better, base)
    assert z > 0 and p < 0.01
