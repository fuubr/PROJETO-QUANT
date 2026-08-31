"""Synthetic control asset generation.

The synthetic asset is a pure geometric random walk. It intentionally has no
forecastable structure and must remain a permanent sanity check for future
models. Any statistically significant predictive performance on this asset is
treated as evidence of leakage, overfitting, or a backtest bug.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config import (
    SEED,
    SYNTHETIC_DAILY_VOL,
    SYNTHETIC_PERIODS,
    SYNTHETIC_START_DATE,
    SYNTHETIC_START_PRICE,
)

logger = logging.getLogger(__name__)


def generate_random_walk_prices(
    periods: int = SYNTHETIC_PERIODS,
    start_price: float = SYNTHETIC_START_PRICE,
    daily_vol: float = SYNTHETIC_DAILY_VOL,
    seed: int = SEED,
    start_date: str = SYNTHETIC_START_DATE,
) -> pd.DataFrame:
    """Generate a reproducible pure random-walk price series.

    Returns a DataFrame indexed by business day with columns:
    - close: synthetic close price
    - return: simple daily return
    """

    if periods < 2:
        raise ValueError("periods must be at least 2")
    if start_price <= 0:
        raise ValueError("start_price must be positive")
    if daily_vol <= 0:
        raise ValueError("daily_vol must be positive")

    rng = np.random.default_rng(seed)
    log_returns = rng.normal(loc=-0.5 * daily_vol**2, scale=daily_vol, size=periods)
    log_returns[0] = 0.0
    prices = start_price * np.exp(np.cumsum(log_returns))
    index = pd.bdate_range(start=start_date, periods=periods)

    frame = pd.DataFrame({"close": prices}, index=index)
    frame["return"] = frame["close"].pct_change().fillna(0.0)

    logger.info(
        "generated_synthetic_random_walk",
        extra={
            "periods": periods,
            "start_price": start_price,
            "daily_vol": daily_vol,
            "seed": seed,
        },
    )
    return frame
