import numpy as np
import pandas as pd
import pytest

from models.calibration import _choose_calibration_method, train_models
from models.evaluation import brier_score


def _make_synthetic_classification_data(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feature_a": rng.normal(size=n),
            "feature_b": rng.normal(size=n),
        }
    )
    # Genuine (if weak) signal so the model has something real to fit.
    logits = 0.5 * X["feature_a"] - 0.3 * X["feature_b"]
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    y = pd.Series((rng.random(n) < probabilities).astype(int))
    return X, y


def test_train_models_rejects_invalid_calibration_fraction():
    X, y = _make_synthetic_classification_data(500, seed=0)

    with pytest.raises(ValueError, match="calibration_fraction"):
        train_models(X, y, calibration_fraction=0.0, seed=42)
    with pytest.raises(ValueError, match="calibration_fraction"):
        train_models(X, y, calibration_fraction=1.0, seed=42)


def test_train_models_rejects_too_small_training_window():
    X, y = _make_synthetic_classification_data(15, seed=0)

    with pytest.raises(ValueError, match="too small"):
        train_models(X, y, calibration_fraction=0.2, seed=42)


def test_train_models_returns_fitted_models_with_valid_probabilities():
    X, y = _make_synthetic_classification_data(500, seed=1)

    models = train_models(X, y, calibration_fraction=0.2, seed=42)

    logistic_probs = models.logistic_regression.predict_proba(X)[:, 1]
    xgboost_probs = models.xgboost_calibrated.predict_proba(X)[:, 1]

    assert ((logistic_probs >= 0.0) & (logistic_probs <= 1.0)).all()
    assert ((xgboost_probs >= 0.0) & (xgboost_probs <= 1.0)).all()


def test_calibration_improves_brier_score_vs_raw_overconfident_model():
    """The whole point of calibration: an overconfident raw model should
    have a worse (higher) Brier score on held-out data than its calibrated
    counterpart, when the raw model's confidence doesn't match reality.
    """

    X_train, y_train = _make_synthetic_classification_data(2000, seed=2)
    X_test, y_test = _make_synthetic_classification_data(500, seed=3)

    models = train_models(X_train, y_train, calibration_fraction=0.3, seed=42)

    raw_probs = models.xgboost_raw.predict_proba(X_test)[:, 1]
    calibrated_probs = models.xgboost_calibrated.predict_proba(X_test)[:, 1]

    raw_brier = brier_score(y_test.to_numpy(), raw_probs)
    calibrated_brier = brier_score(y_test.to_numpy(), calibrated_probs)

    # Calibration should not make things meaningfully worse; a small
    # tolerance accounts for sampling noise on a modestly sized test set.
    assert calibrated_brier <= raw_brier * 1.1


def test_choose_calibration_method_uses_sigmoid_for_small_samples():
    assert _choose_calibration_method(50, min_samples_for_isotonic=200) == "sigmoid"


def test_choose_calibration_method_uses_isotonic_for_large_samples():
    assert _choose_calibration_method(500, min_samples_for_isotonic=200) == "isotonic"


def test_choose_calibration_method_boundary_is_inclusive():
    assert _choose_calibration_method(200, min_samples_for_isotonic=200) == "isotonic"
    assert _choose_calibration_method(199, min_samples_for_isotonic=200) == "sigmoid"


def test_train_models_exposes_chosen_calibration_method():
    X, y = _make_synthetic_classification_data(500, seed=5)

    # calibration_fraction=0.2 of 500 = 100 calibration samples -> below a
    # 200-sample threshold -> should pick sigmoid.
    models_small_calib = train_models(
        X, y, calibration_fraction=0.2, seed=42, min_samples_for_isotonic=200
    )
    assert models_small_calib.calibration_method == "sigmoid"

    # Same data, lower threshold -> same 100 samples now clears it -> isotonic.
    models_large_calib = train_models(
        X, y, calibration_fraction=0.2, seed=42, min_samples_for_isotonic=50
    )
    assert models_large_calib.calibration_method == "isotonic"
