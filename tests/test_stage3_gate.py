import numpy as np
import pandas as pd

from run_stage3_model import _significance_gate_on_synthetic


def test_significance_gate_uses_hac_not_naive_se_on_autocorrelated_predictions():
    """Regression test for a real bug found during development: predictions
    within a walk-forward test block share heavily overlapping label
    horizons (each label looks PREDICTION_HORIZON_DAYS ahead), making the
    correct/incorrect sequence strongly autocorrelated. A naive t-test
    flagged a ~56% out-of-sample accuracy on the SYNTHETIC control asset as
    "significant" (t=2.38); the correct Newey-West-based test showed it was
    not (t=1.35), matching the fact that the synthetic asset has no
    predictable structure by construction.

    Reproduces that bug at the same effective scale (~19 independent
    20-day blocks, matching PREDICTION_HORIZON_DAYS): a deterministic
    pattern of 11 "correct" and 8 "incorrect" blocks gives ~58% accuracy,
    which a naive t-test flags as significant (t=3.11) but Newey-West
    correctly does not (t=1.28). If this test starts failing, it likely
    means the gate was reverted to a naive (non-HAC) test.
    """

    num_blocks = 19
    block_size = 20
    n = num_blocks * block_size

    block_labels = np.array([1] * 11 + [0] * 8, dtype=float)
    correct = np.repeat(block_labels, block_size)

    rng = np.random.default_rng(123)
    labels = rng.integers(0, 2, n).astype(float)
    # predicted probability reproduces the deterministic correct/incorrect
    # pattern relative to the (arbitrary) label column.
    prob = np.where(correct == 1.0, labels, 1.0 - labels)
    prob = np.clip(prob, 0.05, 0.95)

    predictions = pd.DataFrame({"label": labels, "prob_model": prob})

    gate_passed = _significance_gate_on_synthetic(predictions, "prob_model")

    assert gate_passed is True


def _make_block_correct_series(accuracy: float, num_blocks: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Deterministic-ish block-correlated correct/incorrect sequence at a
    target accuracy, mirroring the horizon-overlap autocorrelation of real
    walk-forward predictions.
    """

    num_correct_blocks = round(accuracy * num_blocks)
    block_labels = np.array([1] * num_correct_blocks + [0] * (num_blocks - num_correct_blocks), dtype=float)
    rng.shuffle(block_labels)
    return np.repeat(block_labels, block_size)


def _predictions_from_correct(correct: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    n = len(correct)
    labels = rng.integers(0, 2, n).astype(float)
    prob = np.where(correct == 1.0, labels, 1.0 - labels)
    prob = np.clip(prob, 0.05, 0.95)
    return pd.DataFrame({"label": labels, "prob_model": prob})


def test_pooling_seeds_has_more_power_than_requiring_each_seed_individually():
    """Regression test for a real design flaw found during development:
    requiring EVERY individual seed's gate to pass separately gives each
    test very few effective observations (here, ~19 independent blocks per
    seed), producing frequent spurious individual failures purely from
    sampling variance -- even when the pooled, properly-powered test across
    all seeds shows no real effect. This mirrors what happened with the
    real model: seed-level accuracies of [0.51, 0.53, 0.53, 0.32, 0.41]
    included one seed significant on its own, but the pooled test across
    all five was not significant, matching the true null (the synthetic
    asset has no real structure).
    """

    rng = np.random.default_rng(99)
    seed_accuracies = [0.51, 0.53, 0.53, 0.32, 0.41]
    num_blocks, block_size = 19, 20

    per_seed_predictions = []
    at_least_one_individual_failure = False
    for accuracy in seed_accuracies:
        correct = _make_block_correct_series(accuracy, num_blocks, block_size, rng)
        predictions = _predictions_from_correct(correct, rng)
        per_seed_predictions.append(predictions)

        individually_passed = _significance_gate_on_synthetic(predictions, "prob_model", num_tests=1)
        if not individually_passed:
            at_least_one_individual_failure = True

    assert at_least_one_individual_failure  # reproduces the spurious individual failure

    pooled = pd.concat(per_seed_predictions, ignore_index=True)
    pooled_passed = _significance_gate_on_synthetic(pooled, "prob_model", num_tests=1)

    assert pooled_passed  # the properly-powered pooled test does not reject the null
