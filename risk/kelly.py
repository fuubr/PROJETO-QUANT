"""Fractional Kelly position sizing, driven by the calibrated probability
from Stage 3.

Unlike every baseline in this project (fixed 0/1 exposure), Kelly sizing
makes the bet size itself a function of edge: no edge (p close to 0.5)
should produce a position close to 0, not a full-size bet. This is the
built-in sanity property this module's own gate relies on (see
risk/kelly.py's synthetic-asset test): a model with no real skill should
make Kelly sizing converge toward flat, not toward any particular size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kelly_fraction(win_probability: float, win_loss_ratio: float) -> float:
    """Classic binary Kelly formula: f* = p - (1-p)/b.

    p: probability of a winning outcome (the calibrated probability from
       Stage 3).
    b: win/loss ratio -- average magnitude of a win divided by average
       magnitude of a loss. b <= 0 is degenerate (no informative
       win/loss data) and returns 0.0 rather than raising, so callers can
       treat "no confident bet available" uniformly with "no edge" (both
       result in position 0).

    Clipped to [0, 1]: negative Kelly (suggesting a short) and Kelly > 1
    (suggesting leverage) are both outside this project's existing
    position convention (0 = flat, 1 = fully invested, matching every
    other signal in the codebase). Shorting and leverage are explicitly
    out of scope here, not silently allowed through clipping.
    """

    if not 0.0 <= win_probability <= 1.0:
        raise ValueError("win_probability must be between 0 and 1")

    if win_loss_ratio <= 0:
        return 0.0

    raw_fraction = win_probability - (1.0 - win_probability) / win_loss_ratio
    return float(np.clip(raw_fraction, 0.0, 1.0))


def estimate_win_loss_ratio(
    horizon_returns: pd.Series,
    min_win_observations: int,
    min_loss_observations: int,
    horizon_days: int | None = None,
) -> float | None:
    """Estimate b (win/loss ratio) from a series of realized horizon
    returns (e.g. trailing 20-day returns, matching the model's own
    prediction horizon).

    Returns None -- not a noisy guess -- when there aren't enough win or
    loss observations to estimate reliably. Callers should treat None as
    "no confident bet available", not "assume some default ratio".

    When horizon_days is provided, applies shrinkage toward b=1.0 (neutral
    -- no asymmetry) based on the EFFECTIVE sample size, not the raw
    observation count. Consecutive horizon_days-return observations
    overlap heavily (each shares nearly its entire window with its
    neighbors), so a window of e.g. 252 daily observations with a 20-day
    horizon has only ~252/20 ≈ 12-13 genuinely independent data points --
    the same overlapping-window effect already corrected for with Newey-West
    elsewhere in this project, applied here to reduce day-to-day whiplash
    in the resulting bet size rather than to a significance test. Shrinkage
    strength: b_adjusted = (n_eff * b_raw + K) / (n_eff + K), K=10 -- with
    few effective samples, the estimate stays close to 1.0; with many, it
    converges to the raw sample estimate.
    """

    clean_returns = horizon_returns.dropna()
    wins = clean_returns[clean_returns > 0]
    losses = clean_returns[clean_returns < 0]

    if len(wins) < min_win_observations or len(losses) < min_loss_observations:
        return None

    avg_win = float(wins.mean())
    avg_loss = float(losses.mean())
    if avg_loss == 0:
        return None

    b_raw = avg_win / abs(avg_loss)

    if horizon_days is None or horizon_days < 1:
        return b_raw

    shrinkage_strength = 10.0
    effective_n = max(1.0, len(clean_returns) / horizon_days)
    b_shrunk = (effective_n * b_raw + shrinkage_strength * 1.0) / (effective_n + shrinkage_strength)
    return b_shrunk


def kelly_position_series(
    close: pd.Series,
    calibrated_probability: pd.Series,
    horizon_days: int,
    kelly_fraction_multiplier: float,
    win_loss_window_days: int,
    min_win_observations: int,
    min_loss_observations: int,
) -> pd.Series:
    """Fractional-Kelly position size for each date with a calibrated
    probability available.

    For each date t (using only information available at or before the
    close of day t, matching the causal convention used everywhere else
    in this project):
    1. b is estimated from realized horizon_days returns (close.pct_change
       (horizon_days), a backward-looking, already-realized quantity) over
       the trailing win_loss_window_days.
    2. p is the calibrated probability at t.
    3. position[t] = kelly_fraction_multiplier * kelly_fraction(p, b).

    Dates without enough trailing win/loss observations, or without a
    probability prediction, get position 0.0 -- "not enough information to
    bet confidently" defaults to flat, the same discipline used for
    feature warmup elsewhere in this project.
    """

    if not 0.0 < kelly_fraction_multiplier <= 1.0:
        raise ValueError("kelly_fraction_multiplier must be between 0 (exclusive) and 1")

    realized_horizon_returns = close.pct_change(horizon_days)
    position = pd.Series(0.0, index=calibrated_probability.index)

    for date in calibrated_probability.index:
        if date not in close.index:
            continue
        loc = close.index.get_loc(date)
        window_start = max(0, loc - win_loss_window_days + 1)
        trailing_window = realized_horizon_returns.iloc[window_start : loc + 1]

        win_loss_ratio = estimate_win_loss_ratio(
            trailing_window, min_win_observations, min_loss_observations, horizon_days=horizon_days
        )
        if win_loss_ratio is None:
            continue

        probability = float(calibrated_probability.loc[date])
        fraction = kelly_fraction(probability, win_loss_ratio)
        position.loc[date] = kelly_fraction_multiplier * fraction

    return position
