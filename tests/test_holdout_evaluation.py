import pandas as pd

from run_stage3_holdout_evaluation import _build_training_and_holdout_datasets


def test_training_dataset_has_zero_labels_contaminated_by_holdout_jump():
    """Concrete, value-level proof that the training dataset for the final
    holdout evaluation never sees holdout information -- mirrors
    test_purge_integration.py's approach, applied to the dev/holdout
    boundary instead of a walk-forward fold boundary.

    Plants a permanent price jump exactly at the holdout boundary. Any
    training row whose label window (horizon_days forward) would reach the
    jump has a label that is almost entirely explained by information at or
    after the jump -- i.e., holdout information. If the training dataset is
    built correctly (from dev_close alone, not the full series), no such
    row should exist.
    """

    horizon_days = 20
    n = 2000
    index = pd.bdate_range("2015-01-01", periods=n)
    close = pd.Series(100.0, index=index)

    holdout_start_position = 1500
    holdout_start = index[holdout_start_position]
    close.iloc[holdout_start_position:] = 200.0  # permanent jump exactly at the holdout boundary

    training_dataset, holdout_dataset = _build_training_and_holdout_datasets(close, holdout_start)

    # No training row's date should be within horizon_days of the boundary
    # with a label that could see the jump -- structurally guaranteed by
    # building the training dataset from dev_close (truncated at the
    # boundary) rather than the full series.
    assert training_dataset.index.max() < holdout_start

    # The boundary rows (closest to holdout) must NOT show the stark,
    # jump-driven label pattern (mean label near 1.0) that would appear if
    # holdout data had leaked into their labels.
    boundary_rows = training_dataset.tail(horizon_days)
    assert boundary_rows["label"].mean() < 0.6  # not contaminated

    # The holdout dataset, by contrast, legitimately observes the jump.
    assert len(holdout_dataset) > 0
    assert holdout_dataset.index.min() >= holdout_start


def test_holdout_dataset_features_use_trailing_dev_data_for_warmup():
    """Holdout-period features are allowed to use trailing dev-period
    prices for warmup (that's ordinary causal feature computation, not
    leakage) -- confirmed by checking the first holdout row isn't dropped
    for lack of warmup, unlike what would happen if features were computed
    from holdout_close alone.
    """

    horizon_days = 20
    n = 2000
    index = pd.bdate_range("2015-01-01", periods=n)
    close = pd.Series(100.0 + pd.Series(range(n)).to_numpy() * 0.01, index=index)

    holdout_start_position = 1500
    holdout_start = index[holdout_start_position]

    _, holdout_dataset = _build_training_and_holdout_datasets(close, holdout_start)

    # The very first holdout date should be present (not dropped for
    # warmup), since features draw on trailing dev-period history.
    assert holdout_dataset.index.min() == holdout_start


def test_existing_holdout_results_finds_previous_runs(tmp_path, monkeypatch):
    """Regression coverage for a real process risk found during audit: the
    original 'run only once' instruction was just a print statement, with
    nothing in code actually preventing accidental re-runs (e.g. re-running
    a command from shell history). _existing_holdout_results is the
    detection half of a real guard in main() that now refuses to proceed
    without an explicit --force flag when a prior result exists.
    """

    from run_stage3_holdout_evaluation import _existing_holdout_results

    monkeypatch.setattr("run_stage3_holdout_evaluation.RESULTS_DIR", tmp_path)

    assert _existing_holdout_results() == []

    (tmp_path / "etapa3_SPY_HOLDOUT_FINAL_20260101_000000.csv").write_text("x")
    (tmp_path / "etapa3_SPY_dev_20260101_000000.csv").write_text("x")  # dev result, not holdout

    found = _existing_holdout_results()
    assert len(found) == 1
    assert "HOLDOUT_FINAL" in found[0].name
