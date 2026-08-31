import pandas as pd

from models.comparison import comparison_window, predictions_to_backtest_signal


def test_comparison_window_restricts_to_first_prediction_date():
    """Regression test for a real bug found during development: the model's
    backtest previously included years of "flat" days before its first
    walk-forward prediction existed, while baselines were backtested over
    the full period -- making the model look artificially worse. The fix
    restricts EVERY strategy's comparison window to start where
    out-of-sample predictions actually begin.
    """

    close = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        index=pd.bdate_range("2020-01-01", periods=6),
    )
    predictions = pd.DataFrame(
        {"label": [1.0, 0.0, 1.0]},
        index=close.index[3:],
    )

    restricted = comparison_window(close, predictions)

    assert restricted.index.min() == close.index[3]
    assert len(restricted) == 3


def test_comparison_window_also_restricts_trailing_dates_without_predictions():
    """Regression test for a second real bug found during development,
    discovered from an actual SPY run: walk-forward test blocks tile the
    dataset in fixed-size chunks, so a partial remainder at the end
    (shorter than one full test block -- 68 trailing days for SPY) is left
    without predictions. Without also capping the END of the comparison
    window, those trailing days would default the model's signal to flat
    while baselines stayed fully invested -- reintroducing the identical
    unfair-comparison bug at the tail instead of the head.
    """

    close = pd.Series(
        [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0],
        index=pd.bdate_range("2020-01-01", periods=8),
    )
    predictions = pd.DataFrame(
        {"label": [1.0, 0.0, 1.0]},
        index=close.index[2:5],
    )

    restricted = comparison_window(close, predictions)

    assert restricted.index.min() == close.index[2]
    assert restricted.index.max() == close.index[4]
    assert len(restricted) == 3


def test_predictions_to_backtest_signal_thresholds_at_half():
    index = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0] * 4, index=index)
    predictions = pd.DataFrame({"prob_model": [0.2, 0.5, 0.51, 0.9]}, index=index)

    signal = predictions_to_backtest_signal(close, predictions, "prob_model")

    assert signal.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_predictions_to_backtest_signal_defaults_to_flat_when_no_prediction():
    index = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([100.0] * 4, index=index)
    # Predictions only cover the first 2 days.
    predictions = pd.DataFrame({"prob_model": [0.9, 0.9]}, index=index[:2])

    signal = predictions_to_backtest_signal(close, predictions, "prob_model")

    assert signal.tolist() == [1.0, 1.0, 0.0, 0.0]
