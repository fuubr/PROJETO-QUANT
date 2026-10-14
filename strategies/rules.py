"""Model-free position rules, pre-registered as hypotheses H2/H3 in
docs/CRITERIOS.md. Every rule is causal: the position for date t uses only
prices up to and including t (it is applied to t+1's return by the engine).
Parameters live in config.py and were fixed before any live data existed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252
VOL_WINDOW = 20


def vol_target_position(close: pd.Series, target: float) -> float:
    """Exposure = target / annualized realized vol of the last 20 daily
    returns, clipped to [0, 1]. Flat (0.0) without 20 returns of history.
    """

    returns = close.pct_change().dropna().iloc[-VOL_WINDOW:]
    if len(returns) < VOL_WINDOW:
        return 0.0
    vol = float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))
    if vol <= 0:
        return 1.0
    return float(min(1.0, target / vol))


TREND_WINDOW = 252  # 12 months, Moskowitz-Ooi-Pedersen (2012) time-series momentum


def trend_vol_target_position(close: pd.Series, target: float) -> float:
    """vol_target_position while the trailing 12-month return is positive;
    cash otherwise. Flat without a full year of history. One combined rule
    (not two separate hypotheses), as pre-registered for H3.
    """

    if len(close) <= TREND_WINDOW:
        return 0.0
    if close.iloc[-1] / close.iloc[-1 - TREND_WINDOW] - 1.0 <= 0:
        return 0.0
    return vol_target_position(close, target)
