"""Version-aware inference: which model made which day's prediction.

Single place that turns (close prices, model versions) into probabilities,
shared by the daily script, the consistency check and the drift monitor,
so all three always agree on what each day's decision was.
"""

from __future__ import annotations

import pandas as pd

from config import FEATURE_MOMENTUM_WINDOWS, FEATURE_SMA_DISTANCE_WINDOW, FEATURE_VOLATILITY_WINDOW
from models.features import build_features
from paper_trading.model_store import ModelVersion


def inference_features(close: pd.Series) -> pd.DataFrame:
    """Feature rows for every date with enough trailing history. Uses
    build_features directly (no label, so no row is dropped for lacking a
    future price -- see the bug note in run_stage6_paper_trading_daily).
    """

    return build_features(
        close, momentum_windows=FEATURE_MOMENTUM_WINDOWS, volatility_window=FEATURE_VOLATILITY_WINDOW,
        sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW,
    ).dropna()


def versioned_probabilities(close: pd.Series, versions: list[ModelVersion]) -> pd.Series:
    """P(up) for every date >= the first version's effective_from, each
    predicted by the version that governed that date.
    """

    features = inference_features(close)
    pieces = []
    for i, version in enumerate(versions):
        start = version.effective_from
        end = versions[i + 1].effective_from if i + 1 < len(versions) else None
        mask = features.index >= start
        if end is not None:
            mask &= features.index < end
        rows = features[mask]
        if len(rows) == 0:
            continue
        probs = version.trained.xgboost_calibrated.predict_proba(rows)[:, 1]
        pieces.append(pd.Series(probs, index=rows.index))
    return pd.concat(pieces) if pieces else pd.Series(dtype=float)
