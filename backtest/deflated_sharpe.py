"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

A Sharpe ratio picked as the best of many tries is inflated by the search.
DSR = probability that the true Sharpe exceeds what the best of `n_trials`
zero-skill strategies would show by luck, correcting also for track-record
length, skewness and fat tails. All Sharpe values here are PER PERIOD
(daily), not annualized.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew

EULER_MASCHERONI = 0.5772156649015329


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    """Expected maximum Sharpe among n_trials independent zero-skill
    strategies whose Sharpe estimates have variance sr_variance.
    """

    if n_trials <= 1:
        return 0.0
    g = EULER_MASCHERONI
    return float(np.sqrt(sr_variance) * (
        (1 - g) * norm.ppf(1 - 1 / n_trials) + g * norm.ppf(1 - 1 / (n_trials * np.e))
    ))


def probabilistic_sharpe(sr: float, t: int, skew: float, kurtosis: float, sr_benchmark: float) -> float:
    """P(true Sharpe > sr_benchmark), kurtosis NOT excess (normal = 3)."""

    denom = np.sqrt(1 - skew * sr + (kurtosis - 1) / 4 * sr ** 2)
    return float(norm.cdf((sr - sr_benchmark) * np.sqrt(t - 1) / denom))


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int, sr_variance: float) -> float:
    r = pd.Series(returns).dropna()
    sr = float(r.mean() / r.std(ddof=1))
    return probabilistic_sharpe(
        sr, len(r), skew=float(_skew(r)), kurtosis=float(_kurtosis(r, fisher=False)),
        sr_benchmark=expected_max_sharpe(n_trials, sr_variance),
    )


def memmel_z(sr1: float, sr2: float, rho: float, t: int) -> float:
    """Jobson-Korkie test statistic with Memmel's (2003) correction for
    H0: SR1 == SR2, per-period Sharpes of two return series with correlation
    rho over t periods. Assumes roughly normal iid returns -- an approximation
    for daily equity returns, so treat p-values near the threshold with care.
    """

    variance = (2 * (1 - rho) + 0.5 * (sr1 ** 2 + sr2 ** 2 - 2 * sr1 * sr2 * rho ** 2)) / t
    if variance <= 0:
        return 0.0
    return float((sr1 - sr2) / np.sqrt(variance))


def sharpe_difference_test(returns_a: pd.Series, returns_b: pd.Series) -> tuple[float, float]:
    """(z, two-sided p) for Sharpe(a) - Sharpe(b) on their common dates."""

    a, b = pd.Series(returns_a).align(pd.Series(returns_b), join="inner")
    sr_a, sr_b = a.mean() / a.std(ddof=1), b.mean() / b.std(ddof=1)
    z = memmel_z(float(sr_a), float(sr_b), rho=float(a.corr(b)), t=len(a))
    if z == 0.0:
        return 0.0, 1.0
    return z, float(2 * (1 - norm.cdf(abs(z))))
