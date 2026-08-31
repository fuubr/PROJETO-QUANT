from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data.real import REQUIRED_COLUMNS, download_ohlcv


def _fake_yfinance_frame() -> pd.DataFrame:
    index = pd.bdate_range("2020-01-01", periods=5)
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 103.0, 104.0],
            "High": [101.0, 102.0, 103.0, 104.0, 105.0],
            "Low": [99.0, 100.0, 101.0, 102.0, 103.0],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1_000, 1_100, 1_200, 1_300, 1_400],
        },
        index=index,
    )


def test_download_ohlcv_normalizes_schema(tmp_path, monkeypatch):
    monkeypatch.setattr("data.real.CACHE_DIR", tmp_path)

    fake_module = MagicMock()
    fake_module.download.return_value = _fake_yfinance_frame()

    with patch.dict("sys.modules", {"yfinance": fake_module}):
        frame = download_ohlcv("FAKE", "2020-01-01", "2020-01-10", use_cache=False)

    assert list(frame.columns) == REQUIRED_COLUMNS
    assert len(frame) == 5
    assert (frame["close"] > 0).all()


def test_download_ohlcv_uses_cache_on_second_call(tmp_path, monkeypatch):
    monkeypatch.setattr("data.real.CACHE_DIR", tmp_path)

    fake_module = MagicMock()
    fake_module.download.return_value = _fake_yfinance_frame()

    with patch.dict("sys.modules", {"yfinance": fake_module}):
        first = download_ohlcv("FAKE", "2020-01-01", "2020-01-10", use_cache=True)
        second = download_ohlcv("FAKE", "2020-01-01", "2020-01-10", use_cache=True)

    # yfinance.download should only be hit once; second call reads cache.
    assert fake_module.download.call_count == 1
    # check_freq=False: parquet round-trips do not preserve DatetimeIndex.freq
    # metadata. This is a known pandas/parquet limitation, not data loss --
    # the actual index values and all columns are compared below regardless.
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_download_ohlcv_raises_on_empty_result(tmp_path, monkeypatch):
    monkeypatch.setattr("data.real.CACHE_DIR", tmp_path)

    fake_module = MagicMock()
    fake_module.download.return_value = pd.DataFrame()

    with patch.dict("sys.modules", {"yfinance": fake_module}):
        with pytest.raises(ValueError, match="no data"):
            download_ohlcv("FAKE", "2020-01-01", "2020-01-10", use_cache=False)


def test_download_ohlcv_forward_fills_internal_gaps(tmp_path, monkeypatch):
    monkeypatch.setattr("data.real.CACHE_DIR", tmp_path)

    frame_with_gap = _fake_yfinance_frame()
    # Simulate a missing observation in the middle (e.g. exchange holiday
    # mismatch between markets), not at the very start.
    frame_with_gap.iloc[2] = pd.NA

    fake_module = MagicMock()
    fake_module.download.return_value = frame_with_gap

    with patch.dict("sys.modules", {"yfinance": fake_module}):
        frame = download_ohlcv("FAKE", "2020-01-01", "2020-01-10", use_cache=False)

    assert not frame.isna().any().any()
    # Forward-filled value should equal the prior day's close, not be dropped.
    assert len(frame) == 5
    assert frame["close"].iloc[2] == frame["close"].iloc[1]


def test_download_ohlcv_drops_leading_rows_with_no_fill_source(tmp_path, monkeypatch):
    monkeypatch.setattr("data.real.CACHE_DIR", tmp_path)

    frame_with_leading_gap = _fake_yfinance_frame()
    # No prior value exists to forward-fill the very first row from.
    frame_with_leading_gap.iloc[0] = pd.NA

    fake_module = MagicMock()
    fake_module.download.return_value = frame_with_leading_gap

    with patch.dict("sys.modules", {"yfinance": fake_module}):
        frame = download_ohlcv("FAKE", "2020-01-01", "2020-01-10", use_cache=False)

    assert not frame.isna().any().any()
    assert len(frame) == 4  # first row dropped, no source to fill from
