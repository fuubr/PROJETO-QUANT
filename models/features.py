"""Minimal, causal price-based features for Stage 3.

Deliberately few features, matching the project's stated overfitting
mitigation strategy ("poucas features"). Everything here uses pandas'
trailing rolling/shift operations exclusively, so a feature value on day t
never depends on data after day t -- the same discipline already enforced
structurally in backtest/engine.py for signals.

No volume, no fundamentals, no external data. If these price-only features
prove insufficient, that is itself useful information (documented, not
patched over by silently adding more features).
"""

from __future__ import annotations

import pandas as pd


def build_features(
    close: pd.Series,
    momentum_windows: tuple[int, ...],
    volatility_window: int,
    sma_distance_window: int,
) -> pd.DataFrame:
    """Build the Stage 3 feature set from a close-price series.

    Features (all causal, trailing-only):
    - momentum_{w}: simple return over the trailing w days. Captures
      trend/reversal signal at multiple horizons (short/medium/long).
    - volatility_{w}: rolling std of daily returns over w days. Captures
      current risk regime -- models may weigh momentum differently in
      calm vs turbulent periods.
    - sma_distance_{w}: (close - trailing SMA) / trailing SMA. Captures
      how stretched price is from its own recent trend, reusing the same
      concept already validated in the Stage 1/2 moving-average baseline.

    Rows with insufficient warmup history are NaN (not silently zero-filled
    or dropped here) -- the caller (models/dataset.py) is responsible for
    deciding how to handle warmup NaNs, keeping this function a pure,
    side-effect-free feature builder.
    """

    if not momentum_windows:
        raise ValueError("momentum_windows must not be empty")

    daily_returns = close.pct_change()
    features = {}

    for window in momentum_windows:
        features[f"momentum_{window}"] = close.pct_change(window)

    features[f"volatility_{volatility_window}"] = daily_returns.rolling(
        window=volatility_window
    ).std()

    sma = close.rolling(window=sma_distance_window).mean()
    features[f"sma_distance_{sma_distance_window}"] = (close - sma) / sma

    return pd.DataFrame(features, index=close.index)


def check_feature_correlation(features: pd.DataFrame, threshold: float = 0.7) -> pd.DataFrame:
    """Flag pairs of features with |correlation| above threshold.

    High correlation between features doesn't break tree models or logistic
    regression outright, but it distorts interpretation: SHAP importance
    gets arbitrarily split between correlated features (making any single
    feature's importance look smaller than its true combined signal), and
    logistic regression coefficients become unstable/hard to interpret
    individually. This is diagnostic only -- it does not drop or modify any
    feature, since removing one is a real modeling decision that should be
    made deliberately, not automatically.

    Returns a DataFrame with columns [feature_a, feature_b, correlation],
    one row per flagged pair, sorted by |correlation| descending. Empty if
    no pair exceeds the threshold.
    """

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    correlation_matrix = features.corr()
    flagged_pairs = []

    columns = correlation_matrix.columns
    for i, feature_a in enumerate(columns):
        for feature_b in columns[i + 1:]:
            correlation = correlation_matrix.loc[feature_a, feature_b]
            if abs(correlation) >= threshold:
                flagged_pairs.append(
                    {"feature_a": feature_a, "feature_b": feature_b, "correlation": round(float(correlation), 4)}
                )

    result = pd.DataFrame(flagged_pairs, columns=["feature_a", "feature_b", "correlation"])
    if not result.empty:
        result = result.reindex(result["correlation"].abs().sort_values(ascending=False).index)
        result = result.reset_index(drop=True)
    return result
