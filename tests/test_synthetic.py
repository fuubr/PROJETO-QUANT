import math

import numpy as np
import pandas as pd
import pytest

from config import SEED, SYNTHETIC_START_DATE
from data.synthetic import generate_random_walk_prices

ALPHA = 0.05
STATISTICAL_TEST_PERIODS = 100_000
LJUNG_BOX_MAX_LAG = 20


def _regularized_gamma_q(a: float, x: float) -> float:
    """Return Q(a, x), the upper regularized incomplete gamma function.

    This keeps the statistical tests self-contained in environments where
    scipy/statsmodels are unavailable.
    """

    if a <= 0:
        raise ValueError("a must be positive")
    if x < 0:
        raise ValueError("x must be non-negative")
    if x == 0:
        return 1.0

    epsilon = 1e-14
    max_iterations = 1_000

    if x < a + 1.0:
        term = 1.0 / a
        total = term
        ap = a
        for _ in range(max_iterations):
            ap += 1.0
            term *= x / ap
            total += term
            if abs(term) < abs(total) * epsilon:
                log_p = -x + a * math.log(x) - math.lgamma(a) + math.log(total)
                return 1.0 - float(math.exp(log_p))
        raise RuntimeError("gamma series did not converge")

    tiny = 1e-300
    b = x + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, max_iterations + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < epsilon:
            log_q = -x + a * math.log(x) - math.lgamma(a) + math.log(h)
            return float(math.exp(log_q))
    raise RuntimeError("gamma continued fraction did not converge")


def _chi_square_survival(statistic: float, degrees_of_freedom: int) -> float:
    return _regularized_gamma_q(degrees_of_freedom / 2.0, statistic / 2.0)


def _ljung_box_p_values(values: np.ndarray, max_lag: int) -> list[float]:
    centered = values - values.mean()
    denominator = np.dot(centered, centered)
    n = len(centered)
    p_values = []

    for lag in range(1, max_lag + 1):
        autocorrelations = []
        for k in range(1, lag + 1):
            numerator = np.dot(centered[k:], centered[:-k])
            autocorrelations.append(numerator / denominator)

        statistic = n * (n + 2.0) * sum(
            autocorrelation**2 / (n - k)
            for k, autocorrelation in enumerate(autocorrelations, start=1)
        )
        p_values.append(_chi_square_survival(statistic, lag))

    return p_values


def test_random_walk_is_reproducible_with_fixed_seed():
    first = generate_random_walk_prices(seed=SEED)
    second = generate_random_walk_prices(seed=SEED)

    assert first.equals(second)


def test_random_walk_changes_with_different_seed():
    first = generate_random_walk_prices(seed=SEED)
    second = generate_random_walk_prices(seed=SEED + 1)

    assert not np.allclose(first["close"].to_numpy(), second["close"].to_numpy())


def test_random_walk_schema_and_business_day_index():
    prices = generate_random_walk_prices(periods=10)

    assert list(prices.columns) == ["close", "return"]
    assert len(prices) == 10
    assert prices.index.is_monotonic_increasing
    assert prices.index.freqstr == "B"
    assert prices.index[0] == pd.Timestamp(SYNTHETIC_START_DATE)
    assert (prices["close"] > 0).all()
    assert prices["return"].iloc[0] == pytest.approx(0.0)


def test_random_walk_returns_match_close_prices():
    prices = generate_random_walk_prices(periods=100)
    expected_returns = prices["close"].pct_change().fillna(0.0)

    assert np.allclose(prices["return"].to_numpy(), expected_returns.to_numpy())


def test_random_walk_simple_returns_have_no_expected_drift():
    prices = generate_random_walk_prices(periods=STATISTICAL_TEST_PERIODS)
    returns = prices["return"].iloc[1:].to_numpy()
    standard_error = returns.std(ddof=1) / np.sqrt(len(returns))
    t_statistic = returns.mean() / standard_error

    assert abs(t_statistic) < 1.96


def test_random_walk_seed_has_no_ljung_box_autocorrelation_signal():
    prices = generate_random_walk_prices(periods=STATISTICAL_TEST_PERIODS)
    returns = prices["return"].iloc[1:].to_numpy()
    p_values = _ljung_box_p_values(returns, max_lag=LJUNG_BOX_MAX_LAG)
    bonferroni_alpha = ALPHA / LJUNG_BOX_MAX_LAG

    assert min(p_values) > bonferroni_alpha


def test_random_walk_validates_inputs():
    with pytest.raises(ValueError, match="periods"):
        generate_random_walk_prices(periods=1)
    with pytest.raises(ValueError, match="start_price"):
        generate_random_walk_prices(start_price=0)
    with pytest.raises(ValueError, match="daily_vol"):
        generate_random_walk_prices(daily_vol=0)
