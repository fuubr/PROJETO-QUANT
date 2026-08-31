import numpy as np
import pandas as pd
import pytest

from models.evaluation import brier_score, expected_calibration_error, max_calibration_gap, reliability_diagram_data


def test_brier_score_is_zero_for_perfect_predictions():
    y_true = np.array([1.0, 0.0, 1.0, 0.0])
    y_prob = np.array([1.0, 0.0, 1.0, 0.0])

    assert brier_score(y_true, y_prob) == pytest.approx(0.0)


def test_brier_score_penalizes_confident_wrong_predictions_more():
    y_true = np.array([1.0, 1.0])
    slightly_wrong = np.array([0.4, 0.4])
    very_wrong = np.array([0.01, 0.01])

    assert brier_score(y_true, very_wrong) > brier_score(y_true, slightly_wrong)


def test_reliability_diagram_detects_perfect_calibration():
    rng = np.random.default_rng(0)
    n = 10_000
    y_prob = rng.uniform(0.0, 1.0, n)
    y_true = (rng.random(n) < y_prob).astype(float)

    diagram = reliability_diagram_data(y_true, y_prob, n_bins=10)
    gap = max_calibration_gap(diagram)

    assert gap < 0.05  # well-calibrated by construction, small sampling noise only


def test_reliability_diagram_detects_overconfidence():
    """A model that always predicts 0.9 regardless of the true 50% base
    rate should show a large calibration gap.
    """

    rng = np.random.default_rng(0)
    n = 2_000
    y_prob = np.full(n, 0.9)
    y_true = (rng.random(n) < 0.5).astype(float)  # true rate is 50%, not 90%

    diagram = reliability_diagram_data(y_true, y_prob, n_bins=10)
    gap = max_calibration_gap(diagram)

    assert gap > 0.3


def test_reliability_diagram_drops_empty_bins():
    y_true = np.array([1.0, 0.0])
    y_prob = np.array([0.05, 0.95])  # only the extreme bins are populated

    diagram = reliability_diagram_data(y_true, y_prob, n_bins=10)

    assert len(diagram) == 2


def test_max_calibration_gap_handles_empty_input():
    empty_diagram = pd.DataFrame(columns=["mean_predicted_probability", "observed_frequency"])

    assert np.isnan(max_calibration_gap(empty_diagram))


def test_ece_not_dominated_by_tiny_bin_unlike_max_gap():
    """Regression test for a real issue found in an actual SPY run: a bin
    with only 1 observation showed a calibration gap of 1.0 (100% observed
    vs ~0% predicted), making max_calibration_gap report 1.0 even though
    every other, well-populated bin was well-calibrated. ECE, weighted by
    bin size, should barely move from a well-calibrated baseline when one
    tiny noisy bin is added.
    """

    rng = np.random.default_rng(0)
    n = 2000
    # Predictions clustered in a narrow mid-range band -- mirrors a real
    # model whose outputs mostly cluster together, leaving some bins
    # (like the very low end) nearly empty. A uniform-over-[0,1] sample
    # would populate every bin and wouldn't reproduce the issue.
    y_prob = rng.uniform(0.4, 0.7, n)
    y_true = (rng.random(n) < y_prob).astype(float)  # well-calibrated by construction

    well_calibrated_diagram = reliability_diagram_data(y_true, y_prob, n_bins=10)
    baseline_ece = expected_calibration_error(well_calibrated_diagram)
    assert baseline_ece < 0.05

    # Inject a single wildly miscalibrated observation into the empty
    # low-probability bin.
    y_prob_with_outlier = np.concatenate([y_prob, [0.02]])
    y_true_with_outlier = np.concatenate([y_true, [1.0]])
    diagram_with_outlier = reliability_diagram_data(y_true_with_outlier, y_prob_with_outlier, n_bins=10)

    max_gap = max_calibration_gap(diagram_with_outlier)
    ece_with_outlier = expected_calibration_error(diagram_with_outlier)

    assert max_gap > 0.9  # dominated by the single-point bin, as expected
    assert ece_with_outlier < baseline_ece + 0.01  # barely moves -- not dominated


def test_expected_calibration_error_handles_empty_input():
    empty_diagram = pd.DataFrame(columns=["mean_predicted_probability", "observed_frequency", "count"])

    assert np.isnan(expected_calibration_error(empty_diagram))
