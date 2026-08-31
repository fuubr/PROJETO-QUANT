"""Label construction for Stage 3.

The label is the one place in this project that intentionally looks
forward in time (predicting the future is the entire point of a model).
Every other module in this project (features, backtest signals) is causal
by construction; this module is the deliberate, isolated exception, and is
kept separate specifically so that the forward-looking logic is easy to
audit in one place instead of scattered through feature code.
"""

from __future__ import annotations

import pandas as pd


def build_forward_return_label(close: pd.Series, horizon_days: int) -> pd.Series:
    """Binary label: 1 if close is higher horizon_days ahead, else 0.

    The final `horizon_days` rows have no valid label (their future price
    does not exist yet in the series) and are set to NaN rather than
    silently dropped or guessed -- callers must handle this explicitly.
    """

    if horizon_days < 1:
        raise ValueError("horizon_days must be at least 1")

    forward_return = close.shift(-horizon_days) / close - 1.0
    label = (forward_return > 0).astype(float)
    label[forward_return.isna()] = float("nan")

    return label
