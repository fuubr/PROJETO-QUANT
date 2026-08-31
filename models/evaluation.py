"""Calibration evaluation: does a predicted "70% chance" actually correspond
to a 70% observed frequency? This is the explicit check the project spec
requires before trusting any probability the model outputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean squared error between predicted probability and the binary
    outcome. Lower is better; 0.25 is what a constant 0.5 predictor scores
    against a 50/50 base rate, useful as a rough sanity reference.
    """

    return float(np.mean((y_prob - y_true) ** 2))


def reliability_diagram_data(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> pd.DataFrame:
    """Bins predictions by predicted probability and compares mean predicted
    probability to observed frequency within each bin -- the data needed to
    plot (and to numerically check) a reliability diagram.

    Bins with zero observations are dropped from the output rather than
    shown as misleading zero/NaN rows.
    """

    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1], right=True)

    rows = []
    for bin_index in range(n_bins):
        mask = bin_indices == bin_index
        count = int(mask.sum())
        if count == 0:
            continue
        rows.append(
            {
                "bin_index": bin_index,
                "bin_range": f"[{bin_edges[bin_index]:.2f}, {bin_edges[bin_index + 1]:.2f}]",
                "count": count,
                "mean_predicted_probability": float(y_prob[mask].mean()),
                "observed_frequency": float(y_true[mask].mean()),
            }
        )

    return pd.DataFrame(rows)


def max_calibration_gap(reliability_data: pd.DataFrame) -> float:
    """Largest absolute gap between predicted probability and observed
    frequency across bins -- a single number summarizing how far off
    calibration is in the worst bin. 0.0 for a perfectly calibrated model.

    CAUTION: this is dominated by whichever bin has the fewest
    observations, since a bin with only 1-2 points can show an extreme gap
    (100% or 0% observed frequency) from pure sampling noise, not genuine
    miscalibration. Seen directly in a real SPY run: a bin with count=1
    showed a gap of 1.0 while every other bin was well-calibrated,
    making this metric alone misleading. Prefer
    expected_calibration_error for a bin-size-aware summary; use this
    metric only alongside the full reliability_data table, never alone.
    """

    if reliability_data.empty:
        return float("nan")

    gaps = (
        reliability_data["mean_predicted_probability"]
        - reliability_data["observed_frequency"]
    ).abs()
    return float(gaps.max())


def expected_calibration_error(reliability_data: pd.DataFrame) -> float:
    """Sample-size-weighted average calibration gap across bins (ECE) --
    the standard calibration summary metric, and NOT dominated by small
    bins the way max_calibration_gap can be: a bin with 1 observation
    contributes almost nothing to this weighted average, while a bin with
    hundreds of observations dominates appropriately.
    """

    if reliability_data.empty:
        return float("nan")

    total_count = reliability_data["count"].sum()
    gaps = (
        reliability_data["mean_predicted_probability"]
        - reliability_data["observed_frequency"]
    ).abs()
    weights = reliability_data["count"] / total_count
    return float((gaps * weights).sum())


def print_calibration_report(
    predictions: pd.DataFrame, model_column: str, label: str, n_bins: int
) -> None:
    """Print Brier score, ECE, max bin gap, and the full reliability
    diagram for a model's out-of-sample predictions.

    Shared by run_stage3_model.py (walk-forward folds) and
    run_stage3_holdout_evaluation.py (final holdout) -- previously
    duplicated in both, which had already cosmetically drifted apart
    (different warning wording) within days of being written. A single
    shared implementation removes that drift risk going forward.
    """

    y_true = predictions["label"].to_numpy()
    y_prob = predictions[model_column].to_numpy()

    score = brier_score(y_true, y_prob)
    diagram = reliability_diagram_data(y_true, y_prob, n_bins=n_bins)
    gap = max_calibration_gap(diagram)
    ece = expected_calibration_error(diagram)

    print(f"  [{label}] Brier score: {score:.4f} (0.25 = always predicting the 50% base rate)")
    print(f"  [{label}] ECE (media ponderada por bin, metrica robusta): {ece:.4f}")
    print(f"  [{label}] Maior gap entre bins (CUIDADO: pode ser dominado por bin "
          f"com poucas observacoes -- ver contagem de cada bin abaixo): {gap:.4f}")
    print(diagram.to_string(index=False))
