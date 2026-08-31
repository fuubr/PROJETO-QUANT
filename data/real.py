"""Real market data loading with local caching.

Data is fetched via yfinance and cached to disk as parquet so that:
1. Backtests are reproducible without hitting the network every run.
2. We have an explicit, inspectable record of what data was used.

This module intentionally does NOT compute any features or signals. It only
loads clean OHLCV data. Feature engineering belongs to a later stage.
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "cache"

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def resolve_end_date(end_date: str | None) -> str:
    """Resolve None to today's date (ISO format); pass through explicit dates."""

    if end_date is None:
        return date.today().isoformat()
    return end_date


def _cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    safe_ticker = ticker.replace("/", "-").replace("^", "")
    filename = f"{safe_ticker}_{start_date}_{end_date}.parquet"
    return CACHE_DIR / filename


def download_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str | None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load OHLCV data for a single ticker, using local cache when available.

    end_date=None resolves to today's date, so backtests stay current
    without needing a manual config edit. Because the resolved date is part
    of the cache filename, this naturally creates a fresh cache entry each
    calendar day rather than silently serving stale data.

    Returns a DataFrame indexed by date with lowercase columns:
    open, high, low, close, volume.

    Raises ValueError if the resulting data is empty or missing required
    columns, so failures are loud rather than silently producing an empty
    backtest.
    """

    resolved_end_date = resolve_end_date(end_date)
    cache_file = _cache_path(ticker, start_date, resolved_end_date)

    if use_cache and cache_file.exists():
        logger.info("loading_cached_ohlcv", extra={"ticker": ticker, "path": str(cache_file)})
        frame = pd.read_parquet(cache_file)
        return frame

    frame = _fetch_from_yfinance(ticker, start_date, resolved_end_date)

    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_file)
        logger.info("cached_ohlcv", extra={"ticker": ticker, "path": str(cache_file)})

    return frame


def _fetch_from_yfinance(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required to download real data. "
            "Install it with: pip install yfinance"
        ) from exc

    raw = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"yfinance returned no data for ticker={ticker!r} "
            f"between {start_date} and {end_date}"
        )

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    frame = raw.rename(columns=str.lower)[REQUIRED_COLUMNS].copy()
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    frame.index.name = "date"

    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"downloaded data for {ticker!r} is missing columns: {missing}")

    frame = _handle_missing_data(frame, ticker)

    return frame


def _handle_missing_data(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Forward-fill internal gaps (e.g. exchange-specific holidays not
    aligned across markets) and drop any leading rows that still have NaN
    because there is no prior value to fill from.

    Silently leaving NaNs in would corrupt downstream return calculations
    (a NaN close produces a NaN return, which pandas operations can then
    propagate or silently drop in ways that are easy to miss). Handling it
    once here, explicitly and loudly logged, keeps that logic out of every
    consumer of this data.
    """

    missing_count = int(frame.isna().sum().sum())
    if missing_count == 0:
        return frame

    frame = frame.sort_index().ffill()

    remaining = int(frame.isna().sum().sum())
    # Calculate fills while we still know which NaNs were actually filled
    # (before dropna removes any un-fillable leading rows). Doing this after
    # dropna would make remaining == 0 and report that ALL original NaNs were
    # filled by ffill -- including the ones that were actually dropped.
    values_filled = missing_count - remaining

    if remaining > 0:
        rows_before = len(frame)
        frame = frame.dropna()
        logger.warning(
            "dropped_leading_rows_with_no_fill_source",
            extra={
                "ticker": ticker,
                "rows_dropped": rows_before - len(frame),
            },
        )

    logger.warning(
        "forward_filled_missing_ohlcv",
        extra={"ticker": ticker, "values_filled": values_filled},
    )

    return frame


def load_all(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
    use_cache: bool = True,
) -> dict[str, pd.DataFrame]:
    """Load OHLCV data for multiple tickers, keyed by ticker."""

    return {
        ticker: download_ohlcv(ticker, start_date, end_date, use_cache=use_cache)
        for ticker in tickers
    }
