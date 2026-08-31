"""Mandatory baselines.

Every strategy introduced in future stages must be compared against these.
A model that cannot beat buy-and-hold and random entry, net of costs and
with statistical significance, has not demonstrated skill.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def buy_and_hold_signal(index: pd.Index) -> pd.Series:
    """Always fully invested."""

    return pd.Series(1.0, index=index)


def random_entry_signal(
    index: pd.Index,
    seed: int,
    probability: float = 0.5,
) -> pd.Series:
    """Random binary in/out position, independent each day.

    This is deliberately naive (no persistence, no momentum) so it serves as
    a pure noise floor: any strategy that cannot beat this by a statistically
    significant margin has no demonstrated edge.

    Turnover is ~50% of days by construction, which is unrealistically high
    for most real trading strategies -- see random_entry_persistent_signal
    for a lower-turnover alternative that still carries no directional edge.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")

    rng = np.random.default_rng(seed)
    positions = (rng.random(len(index)) < probability).astype(float)
    return pd.Series(positions, index=index)


def random_entry_persistent_signal(
    index: pd.Index,
    seed: int,
    probability: float = 0.5,
    avg_holding_days: float = 20.0,
) -> pd.Series:
    """Random in/out position with persistence, giving a more realistic
    turnover floor than the pure i.i.d. version above.

    Implemented as a symmetric two-state toggle chain: each day, the
    position flips to the opposite state with probability
    1 / avg_holding_days, and otherwise carries over unchanged. Every
    "switch" is therefore a real position change (unlike naively redrawing a
    fresh Bernoulli value, which would only actually change position about
    half the time and silently double the real holding period). This makes
    the expected holding period equal to avg_holding_days, and -- because
    the chain is symmetric -- the long-run fraction of time in the market
    converges to 50% regardless of the initial draw.

    `probability` only controls the initial day's starting state and is
    included for interface symmetry with random_entry_signal; it does not
    control the long-run in-market fraction for this persistent variant.
    Each new position, whether from the initial draw or a later toggle, is
    still a fair coin flip independent of price history, so this baseline
    carries no directional edge.
    """

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0 and 1")
    if avg_holding_days < 1:
        raise ValueError("avg_holding_days must be at least 1")

    rng = np.random.default_rng(seed)
    switch_probability = 1.0 / avg_holding_days
    switches = rng.random(len(index)) < switch_probability

    positions = np.empty(len(index))
    positions[0] = float(rng.random() < probability)
    for day in range(1, len(index)):
        positions[day] = 1.0 - positions[day - 1] if switches[day] else positions[day - 1]

    return pd.Series(positions, index=index)


def moving_average_crossover_signal(
    close: pd.Series,
    fast_window: int,
    slow_window: int,
) -> pd.Series:
    """Classic two-moving-average crossover: positioned when the fast SMA is
    above the slow SMA, flat otherwise.

    This is the "one simple non-AI strategy" required before any predictive
    model is introduced -- it exists to validate that the backtest engine
    produces sane results for a rule that actually reacts to price (unlike
    buy-and-hold, which is insensitive to most engine bugs since it never
    changes position).

    Both SMAs use pandas' trailing rolling window (no lookahead: SMA on day
    t only uses closes up to and including day t). During the warmup period,
    before slow_window observations are available, the comparison naturally
    evaluates to flat (NaN comparisons are False in pandas) rather than
    requiring special-cased warmup logic.
    """

    if fast_window < 1 or slow_window < 1:
        raise ValueError("fast_window and slow_window must be at least 1")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be smaller than slow_window")

    fast_sma = close.rolling(window=fast_window).mean()
    slow_sma = close.rolling(window=slow_window).mean()

    signal = (fast_sma > slow_sma).astype(float)
    return signal
