import numpy as np
import pandas as pd
import pytest

from models.labels import build_forward_return_label


def test_label_matches_manual_forward_return_calculation():
    index = pd.bdate_range("2020-01-01", periods=6)
    # +10% every 2 days: 100 -> 110 -> 100 -> 110 -> 100 -> 110
    close = pd.Series([100.0, 110.0, 100.0, 110.0, 100.0, 110.0], index=index)

    label = build_forward_return_label(close, horizon_days=2)

    # day0 (100) -> day2 (100): flat, not > 0 -> label 0
    # day1 (110) -> day3 (110): flat -> label 0
    # day2 (100) -> day4 (100): flat -> label 0
    # day3 (110) -> day5 (110): flat -> label 0
    # day4, day5: no future data -> NaN
    assert label.iloc[:4].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert label.iloc[4:].isna().all()


def test_label_final_horizon_days_are_nan():
    index = pd.bdate_range("2020-01-01", periods=30)
    close = pd.Series(range(100, 130), index=index, dtype=float)

    label = build_forward_return_label(close, horizon_days=20)

    assert label.iloc[:10].isna().sum() == 0
    assert label.iloc[10:].isna().all()


def test_label_rejects_non_positive_horizon():
    close = pd.Series([100.0, 101.0], index=pd.bdate_range("2020-01-01", periods=2))

    with pytest.raises(ValueError, match="horizon_days"):
        build_forward_return_label(close, horizon_days=0)
