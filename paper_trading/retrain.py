"""Champion/challenger retraining.

Retraining is never automatic and never overwrites anything:
- A CHALLENGER is trained on data the current model has not seen, up to a
  cut-off, and both models are scored on the SAME later window that is
  out-of-sample for both (no leakage: challenger training labels only use
  prices up to the cut-off; evaluation labels only use prices after it).
- Promotion is a separate, explicit step that registers a NEW version
  (paper_trading/model_store.py), effective only from the next decision
  date, so logged history is never re-attributed to a different model.
- Evidence gate: promotion normally requires the challenger to be
  significantly better (block-bootstrap CI above zero) on at least
  DRIFT_MIN_EFFECTIVE_SAMPLES independent windows. Proving a retrain
  helped takes months of data -- that is a property of the problem, not a
  limitation to be tuned away. An explicit override exists, but it is loud.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import (
    CALIBRATION_FRACTION, CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC, DRIFT_BOOTSTRAP_RESAMPLES,
    DRIFT_BOOTSTRAP_SEED, DRIFT_CI_ALPHA, DRIFT_MIN_EFFECTIVE_SAMPLES, FEATURE_MOMENTUM_WINDOWS,
    FEATURE_SMA_DISTANCE_WINDOW, FEATURE_VOLATILITY_WINDOW, PREDICTION_HORIZON_DAYS,
    RETRAIN_EVAL_FRACTION, SEED,
)
from models.calibration import TrainedModels, train_models
from models.dataset import build_dataset, feature_columns
from models.labels import build_forward_return_label
from paper_trading.drift import block_bootstrap_mean_ci
from paper_trading.model_store import ModelVersion, register_version
from paper_trading.predictions import inference_features


def train_on_history(close: pd.Series, train_end: pd.Timestamp) -> TrainedModels:
    """Train using only prices up to train_end. build_dataset drops the
    last horizon rows (their labels need prices after train_end), which is
    exactly the purge -- no label ever uses a price past the cut-off.
    """

    dataset = build_dataset(
        close[close.index <= train_end], momentum_windows=FEATURE_MOMENTUM_WINDOWS,
        volatility_window=FEATURE_VOLATILITY_WINDOW, sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW,
        horizon_days=PREDICTION_HORIZON_DAYS,
    )
    cols = feature_columns(dataset)
    return train_models(
        dataset[cols], dataset["label"], calibration_fraction=CALIBRATION_FRACTION, seed=SEED,
        min_samples_for_isotonic=CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC,
    )


@dataclass
class ChallengerComparison:
    verdict: str  # INSUFICIENTE | CHALLENGER_MELHOR | SEM_DIFERENCA | INCUMBENTE_MELHOR
    train_end: pd.Timestamp | None
    eval_start: pd.Timestamp | None
    n_matured: int
    n_effective: float
    mean_advantage: float  # mean per-day Brier advantage of the challenger (>0 = challenger better)
    ci_low: float
    ci_high: float
    challenger: TrainedModels | None


def evaluate_challenger(close: pd.Series, incumbent: ModelVersion) -> ChallengerComparison:
    labels = build_forward_return_label(close, PREDICTION_HORIZON_DAYS).dropna()
    matured = labels[labels.index > incumbent.trained_through]
    m = len(matured)
    split = int(m * (1 - RETRAIN_EVAL_FRACTION))
    n_eval = m - split
    n_eff = n_eval / PREDICTION_HORIZON_DAYS
    nan = float("nan")

    # Not even enough days to form two parts with a meaningful evaluation.
    if split < 1 or n_eval < 2:
        return ChallengerComparison("INSUFICIENTE", None, None, n_eval, n_eff, nan, nan, nan, None)

    eval_start = matured.index[split]
    train_end = matured.index[split - 1]
    challenger = train_on_history(close, train_end)

    eval_labels = matured.iloc[split:]
    features = inference_features(close).loc[eval_labels.index]
    p_inc = incumbent.trained.xgboost_calibrated.predict_proba(features)[:, 1]
    p_chal = challenger.xgboost_calibrated.predict_proba(features)[:, 1]
    y = eval_labels.to_numpy()
    advantage = (p_inc - y) ** 2 - (p_chal - y) ** 2

    mean, lo, hi = block_bootstrap_mean_ci(
        advantage, PREDICTION_HORIZON_DAYS, DRIFT_BOOTSTRAP_RESAMPLES, DRIFT_CI_ALPHA, DRIFT_BOOTSTRAP_SEED
    )
    if n_eff < DRIFT_MIN_EFFECTIVE_SAMPLES:
        verdict = "INSUFICIENTE"
    elif lo > 0:
        verdict = "CHALLENGER_MELHOR"
    elif hi < 0:
        verdict = "INCUMBENTE_MELHOR"
    else:
        verdict = "SEM_DIFERENCA"
    return ChallengerComparison(verdict, train_end, eval_start, n_eval, n_eff, mean, lo, hi, challenger)


def promote(
    ticker: str, close: pd.Series, latest: ModelVersion, comparison: ChallengerComparison, force: bool
) -> ModelVersion:
    """Register a new version trained on ALL data to date. Refuses unless
    the evidence gate passed, or force=True (explicit, caller-visible).
    """

    if comparison.verdict != "CHALLENGER_MELHOR" and not force:
        raise PermissionError(
            f"Promocao recusada: veredito do challenger = {comparison.verdict}. "
            f"Use --force-promote para promover mesmo assim (decisao consciente, sem evidencia estatistica)."
        )
    as_of = close.index[-1]
    final_model = train_on_history(close, as_of)
    return register_version(
        ticker, final_model, trained_through=as_of, effective_from=as_of + pd.Timedelta(days=1)
    )
