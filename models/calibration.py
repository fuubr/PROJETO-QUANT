"""Model training and calibration.

Two models per the project spec: XGBoost as the primary model, logistic
regression as an interpretable internal baseline it must be compared
against -- not just beat the Stage 2 baselines, but justify its extra
complexity over the simplest reasonable classifier.

XGBoost's raw predict_proba output is not a calibrated probability by
default (tree-ensemble scores tend to be overconfident). It is calibrated
here using an isotonic regression fit on a chronologically LATER slice of
the training window, held out from the model-fitting step itself -- fitting
the base model and its calibration on the same rows would let the
calibration curve simply memorize the fit model's own training quirks
rather than measure genuine miscalibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class TrainedModels:
    logistic_regression: Pipeline  # StandardScaler + LogisticRegression
    xgboost_calibrated: CalibratedClassifierCV
    xgboost_raw: XGBClassifier  # uncalibrated, kept for SHAP (tree-based
    # explainers work directly on the raw booster; the calibration wrapper
    # is a post-hoc adjustment layered on top, not part of the tree
    # structure SHAP explains).
    calibration_method: str  # "isotonic" or "sigmoid" -- exposed for
    # transparency/logging, since the choice is made dynamically based on
    # calibration-slice size, not fixed.


def _choose_calibration_method(n_calibration_samples: int, min_samples_for_isotonic: int) -> str:
    """Isotonic regression is non-parametric and flexible, but can overfit
    ("memorize") noise with few calibration points -- exactly the failure
    mode visible in early walk-forward folds, which have the smallest
    training (and therefore calibration) windows. Sigmoid (Platt scaling)
    has only 2 parameters and degrades more gracefully with limited data.
    """

    return "isotonic" if n_calibration_samples >= min_samples_for_isotonic else "sigmoid"


def _build_xgboost_classifier(seed: int) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )


def train_xgboost_for_interpretation(X: pd.DataFrame, y: pd.Series, seed: int) -> XGBClassifier:
    """Fit an XGBoost model on ALL provided rows, with no fit/calibration
    split.

    Use this (not TrainedModels.xgboost_raw) when the goal is pure feature
    interpretation (e.g. SHAP), not evaluation -- there is no need to hold
    back a calibration slice when the model's probability outputs
    themselves are never used downstream. Using train_models' xgboost_raw
    for this purpose would silently train on only the ~80% "fit" portion
    (calibration_fraction held back), understating how much data the
    interpretation is actually based on.
    """

    model = _build_xgboost_classifier(seed)
    model.fit(X, y)
    return model


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    calibration_fraction: float,
    seed: int,
    min_samples_for_isotonic: int = 200,
) -> TrainedModels:
    """Train the logistic regression baseline on the full training window,
    and a calibrated XGBoost model whose base estimator is fit on the
    chronologically EARLIER portion of the window, calibrated on the
    chronologically LATER portion -- never shuffled, matching the same
    walk-forward discipline used everywhere else in this project.

    The calibration method (isotonic vs sigmoid) is chosen automatically
    based on how many observations end up in the calibration slice -- see
    _choose_calibration_method.
    """

    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between 0 and 1")

    # LogisticRegression with StandardScaler: features (momentum, volatility,
    # sma_distance) have different numeric scales. L2 regularization penalizes
    # coefficients in absolute terms, so without scaling, high-magnitude
    # features receive artificially larger penalties -- distorting relative
    # importances and slowing convergence. StandardScaler inside a Pipeline
    # ensures scaling is fit only on X_train (no data leakage from X_test).
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=1000)),
    ])
    logistic_model.fit(X_train, y_train)

    n = len(X_train)
    fit_size = int(n * (1.0 - calibration_fraction))
    if fit_size < 10 or (n - fit_size) < 10:
        raise ValueError(
            f"training window too small to split into fit/calibration "
            f"subsets (n={n}, fit_size={fit_size}); use a longer "
            f"WALK_FORWARD_MIN_TRAIN_DAYS"
        )

    X_fit, X_calib = X_train.iloc[:fit_size], X_train.iloc[fit_size:]
    y_fit, y_calib = y_train.iloc[:fit_size], y_train.iloc[fit_size:]

    xgboost_raw = _build_xgboost_classifier(seed)
    xgboost_raw.fit(X_fit, y_fit)

    calibration_method = _choose_calibration_method(len(X_calib), min_samples_for_isotonic)
    xgboost_calibrated = CalibratedClassifierCV(
        FrozenEstimator(xgboost_raw), method=calibration_method
    )
    xgboost_calibrated.fit(X_calib, y_calib)

    return TrainedModels(
        logistic_regression=logistic_model,
        xgboost_calibrated=xgboost_calibrated,
        xgboost_raw=xgboost_raw,
        calibration_method=calibration_method,
    )
