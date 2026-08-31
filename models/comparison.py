"""Shared helpers for turning model predictions into a backtest-ready
signal, restricted to a fair comparison window.

Extracted after these were found duplicated (and already cosmetically
drifting) between run_stage3_model.py (walk-forward folds) and
run_stage3_holdout_evaluation.py (final holdout) during an audit review.
"""

from __future__ import annotations

import pandas as pd


def comparison_window(close: pd.Series, predictions: pd.DataFrame) -> pd.Series:
    """Restrict a close-price series to the range where out-of-sample
    predictions actually exist, so every strategy in a baseline comparison
    (model included) is evaluated over the identical window.

    Fixes two real bugs found during development (see project README for
    the full history): a model backtested only where it has real
    predictions can otherwise look artificially worse if baselines are
    computed over a wider period the model was never actually able to
    trade in -- both at the start (walk-forward warmup) and the end
    (partial trailing test block with no coverage).
    """

    comparison_start = predictions.index.min()
    comparison_end = predictions.index.max()
    return close[(close.index >= comparison_start) & (close.index <= comparison_end)]


def predictions_to_backtest_signal(
    close: pd.Series, predictions: pd.DataFrame, model_column: str
) -> pd.Series:
    """Binary signal: 1 when predicted probability > 0.5, else 0. Aligned to
    the full close-price index (days without a prediction default to
    flat/0, not silently forward-filled).
    """

    signal = pd.Series(0.0, index=close.index)
    aligned_probs = predictions[model_column].reindex(close.index)
    signal[aligned_probs > 0.5] = 1.0
    return signal
