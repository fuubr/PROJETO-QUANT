"""Drift detection for the live paper trading model.

Three deliberately separate questions, because they have different
causes and different remedies:

1. IMPLEMENTATION divergence (consistency_check.py): does live match the
   backtest engine with the same model? A mismatch is a BUG, not drift.
2. FEATURE drift (this module): do today's inputs look unlike the
   training data? Needs no labels, so it is available immediately -- but
   with few live days it has very little power, and says nothing by itself
   about whether predictions got worse.
3. PERFORMANCE drift (this module): are probabilities worse than a
   trivial base-rate predictor on matured outcomes? Needs 20-day labels to
   mature AND enough independent windows; before that the verdict is
   INSUFICIENTE, by design, rather than a noisy guess.

Statistical honesty notes (project-wide lessons applied here):
- 20-day labels overlap, so n_effective = matured_days / horizon, not
  matured_days. Verdicts are gated on n_effective.
- Naive p-values on autocorrelated features are anti-conservative. The
  feature test therefore uses an EMPIRICAL null built from rolling windows
  of the same length in the reference period, which keeps the
  autocorrelation intact. Its resolution is limited (about
  reference_days / window_length independent windows), so it reports
  "ATENCAO", never "confirmed drift".
- Performance uses a moving-block bootstrap (block = horizon) for the CI.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class FeatureDriftResult:
    feature: str
    standardized_shift: float  # (live mean - reference mean) / reference std
    empirical_p: float  # P(|shift| this large | rolling windows of the same length)
    alert: bool


def feature_drift(
    reference: pd.DataFrame, live: pd.DataFrame, alert_p: float, min_live_days: int
) -> list[FeatureDriftResult] | None:
    """Compare each feature's live mean to the reference mean, in
    reference-std units, against an empirical null from rolling windows of
    the SAME length in the reference. None if there is too little live data
    (or too little reference to build a null) for any statement.
    """

    window = len(live)
    if window < min_live_days or len(reference) < 3 * window:
        return None

    results = []
    for feature in reference.columns:
        ref = reference[feature]
        mean, std = ref.mean(), ref.std(ddof=1)
        if std == 0:
            continue
        live_shift = (live[feature].mean() - mean) / std
        rolling_means = ref.rolling(window).mean().dropna()
        null_shifts = ((rolling_means - mean) / std).abs()
        p = (float((null_shifts >= abs(live_shift)).sum()) + 1.0) / (len(null_shifts) + 1.0)
        results.append(FeatureDriftResult(feature, float(live_shift), p, p < alert_p))
    return results


def block_bootstrap_mean_ci(
    values: np.ndarray, block_length: int, n_resamples: int, alpha: float, seed: int
) -> tuple[float, float, float]:
    """Mean and (1-alpha) CI of a serially dependent series via the moving
    block bootstrap.
    """

    n = len(values)
    if n < 2 * block_length:
        # Fewer than two independent blocks: resampling would collapse to a
        # single point and look falsely precise. Say "unavailable" instead.
        return float(values.mean()), float("nan"), float("nan")
    block = block_length
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_resamples, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_resamples, -1)[:, :n]
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


@dataclass
class PerformanceDriftResult:
    verdict: str  # INSUFICIENTE | OK | DEGRADADO
    n_matured: int
    n_effective: float
    mean_skill: float  # mean per-day Brier advantage over the base-rate predictor (>0 = model better)
    ci_low: float
    ci_high: float


def performance_drift(
    probabilities: pd.Series, labels: pd.Series, base_rate: float, horizon_days: int,
    min_effective_samples: float, n_resamples: int, alpha: float, seed: int,
) -> PerformanceDriftResult:
    """Brier skill versus a trivial base-rate predictor on matured live
    outcomes. DEGRADADO only if the CI lies entirely below zero (the model
    is significantly WORSE than always predicting the base rate); OK means
    only "not significantly worse", not "good".
    """

    common = probabilities.index.intersection(labels.index)
    p, y = probabilities.loc[common].to_numpy(), labels.loc[common].to_numpy()
    n = len(common)
    n_eff = n / horizon_days
    if n == 0:
        return PerformanceDriftResult("INSUFICIENTE", 0, 0.0, float("nan"), float("nan"), float("nan"))

    skill = (base_rate - y) ** 2 - (p - y) ** 2
    if n < 2:
        return PerformanceDriftResult("INSUFICIENTE", n, n_eff, float(skill.mean()), float("nan"), float("nan"))
    mean, lo, hi = block_bootstrap_mean_ci(skill, horizon_days, n_resamples, alpha, seed)

    if n_eff < min_effective_samples:
        verdict = "INSUFICIENTE"
    elif hi < 0:
        verdict = "DEGRADADO"
    else:
        verdict = "OK"
    return PerformanceDriftResult(verdict, n, n_eff, mean, lo, hi)
