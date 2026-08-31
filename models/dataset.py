"""Assembles features + label into a single clean dataset, with explicit
accounting of what gets dropped and why.
"""

from __future__ import annotations

import pandas as pd

from models.features import build_features
from models.labels import build_forward_return_label


def build_dataset(
    close: pd.Series,
    momentum_windows: tuple[int, ...],
    volatility_window: int,
    sma_distance_window: int,
    horizon_days: int,
) -> pd.DataFrame:
    """Build a dataset with feature columns + a `label` column, indexed by
    date, with incomplete rows dropped.

    Two distinct sources of missing data are both handled by a single
    dropna, but are conceptually different and worth naming:
    - Warmup NaNs at the START (not enough trailing history yet for the
      longest-window feature).
    - Unknown-future NaNs at the END (the label needs horizon_days of
      future price that does not exist yet for the most recent rows).
    Both are expected and correct to drop, not bugs.
    """

    features = build_features(
        close,
        momentum_windows=momentum_windows,
        volatility_window=volatility_window,
        sma_distance_window=sma_distance_window,
    )
    label = build_forward_return_label(close, horizon_days=horizon_days)

    dataset = features.copy()
    dataset["label"] = label
    dataset = dataset.dropna()

    return dataset


def feature_columns(dataset: pd.DataFrame) -> list[str]:
    return [column for column in dataset.columns if column != "label"]
