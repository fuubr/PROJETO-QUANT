"""Statistical testing utilities shared across the project.

Centralizes two corrections identified as gaps after Stage 2:

1. Newey-West (HAC) standard errors: the naive t-test used throughout
   (returns.std(ddof=1) / sqrt(n)) assumes i.i.d. returns. That assumption
   holds for the synthetic control asset by construction (validated via
   Ljung-Box in Stage 0), but is NOT safe to assume for strategy returns in
   general -- persistence in a signal, volatility clustering, or overlapping
   trades can all induce autocorrelation that a naive standard error
   underestimates, inflating the t-statistic and creating false positives.

2. Bonferroni correction for multiple comparisons: reporting scripts test
   many (asset, baseline) combinations at once. Without correction, testing
   20 combinations at alpha=5% gives a ~64% chance of at least one false
   positive by chance alone ((1-0.05)^20 =~ 0.36 survival).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def default_max_lag(n: int) -> int:
    """Newey & West (1994) rule-of-thumb lag truncation."""

    return max(1, int(4.0 * (n / 100.0) ** (2.0 / 9.0)))


def newey_west_long_run_variance(returns: np.ndarray, max_lag: int) -> float:
    """HAC (Bartlett-kernel) long-run variance estimator."""

    n = len(returns)
    demeaned = returns - returns.mean()
    gamma_0 = float(np.dot(demeaned, demeaned) / n)

    long_run_variance = gamma_0
    for lag in range(1, max_lag + 1):
        gamma_lag = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        weight = 1.0 - lag / (max_lag + 1)
        long_run_variance += 2.0 * weight * gamma_lag

    return long_run_variance


def newey_west_t_stat(returns: np.ndarray, max_lag: int | None = None) -> float:
    """One-sample t-statistic for mean(returns) != 0, using a Newey-West HAC
    standard error instead of the naive i.i.d. standard error.

    Falls back to a t-statistic of 0.0 if the estimated variance is
    non-positive (degenerate input, e.g. all-zero returns).
    """

    n = len(returns)
    if n < 2:
        raise ValueError("need at least 2 observations")
    if max_lag is None:
        max_lag = default_max_lag(n)

    variance = newey_west_long_run_variance(returns, max_lag)
    if variance <= 0:
        return 0.0

    standard_error = math.sqrt(variance / n)
    return float(returns.mean() / standard_error)


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def two_sided_p_value(t_stat: float) -> float:
    """Normal-approximation two-sided p-value. Appropriate for the sample
    sizes used in this project (hundreds to tens of thousands of
    observations), where the t-distribution is indistinguishable from
    normal.
    """

    return 2.0 * (1.0 - normal_cdf(abs(t_stat)))


def bonferroni_alpha(alpha: float, num_tests: int) -> float:
    """Family-wise error rate correction: the per-test significance
    threshold required so that the probability of at least one false
    positive across num_tests independent tests stays at approximately
    alpha.
    """

    if num_tests < 1:
        raise ValueError("num_tests must be at least 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    return alpha / num_tests


def mean_return_significance(
    net_returns: pd.Series,
    num_tests: int = 1,
    alpha: float = 0.05,
    max_lag: int | None = None,
) -> dict:
    """Full significance assessment for whether mean net return differs
    from zero: Newey-West t-stat, its p-value, and whether it clears the
    Bonferroni-corrected threshold for a family of num_tests comparisons.

    num_tests=1 (default) applies no multiple-comparison correction --
    appropriate for a single pre-registered gate (e.g. one pytest
    assertion). Reporting scripts that test many assets/baselines at once
    should pass the total row count as num_tests.
    """

    # Drop the first return: the backtest engine always produces a 0.0 on
    # day 0 (no prior position to hold, shift(1) fills NaN with 0.0).
    # Including that structural zero would deflate the mean and inflate the
    # standard error, biasing the t-statistic toward zero. This matches the
    # identical slice used in the statistical gate tests (test_engine.py).
    returns = net_returns.iloc[1:].to_numpy()
    t_stat = newey_west_t_stat(returns, max_lag=max_lag)
    p_value = two_sided_p_value(t_stat)
    corrected_alpha = bonferroni_alpha(alpha, num_tests)

    return {
        "t_stat": t_stat,
        "p_value": p_value,
        "bonferroni_alpha": corrected_alpha,
        "significant": p_value < corrected_alpha,
    }
