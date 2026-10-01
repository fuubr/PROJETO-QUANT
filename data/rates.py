"""Risk-free (cash) rates from official sources, replacing assumed constants.

- Brazil: BCB SGS series 12 (CDI, % per day). Free, no key. The BCB limits
  daily-series queries to 10-year windows, so requests are chunked (5y).
- US: FRED DTB3 (3-month T-bill, annual %), via the public CSV endpoint (no
  key). Converted to a per-trading-day rate with 252 periods per year.

Both are sanity-checked (an unconverted percent or a unit error would
silently produce absurd cash returns) and cached on disk. If a source is
unreachable, `risk_free_daily` falls back to the ASSUMED constants in
config.py and says so in the returned source label -- it never silently
passes an assumption off as data.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

from config import CASH_ANNUAL_RATE_ASSUMED

BCB_CDI_SERIES = 12
FRED_US_SERIES = "DTB3"
TRADING_DAYS = 252
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "rates"
_CHUNK_YEARS = 5  # BCB rejects daily-series windows above 10 years


def _http_get(url: str, params: dict | None = None, retries: int = 3, timeout: int = 30) -> requests.Response:
    last = None
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code == 200:
                return response
            last = RuntimeError(f"HTTP {response.status_code} from {url}")
        except requests.RequestException as exc:
            last = exc
        time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not fetch {url}: {last}")


def annual_to_daily(annual_decimal: pd.Series | float) -> pd.Series | float:
    return (1.0 + annual_decimal) ** (1.0 / TRADING_DAYS) - 1.0


def daily_to_annual(daily_decimal: pd.Series | float) -> pd.Series | float:
    return (1.0 + daily_decimal) ** TRADING_DAYS - 1.0


def fetch_cdi_daily(start: str, end: str) -> pd.Series:
    """CDI as a per-business-day decimal rate (e.g. 0.0005 = 0.05%/day)."""

    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    pieces, cursor = [], start_ts
    while cursor <= end_ts:
        chunk_end = min(cursor + pd.DateOffset(years=_CHUNK_YEARS) - pd.Timedelta(days=1), end_ts)
        response = _http_get(
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.{BCB_CDI_SERIES}/dados",
            params={"formato": "json", "dataInicial": cursor.strftime("%d/%m/%Y"), "dataFinal": chunk_end.strftime("%d/%m/%Y")},
        )
        rows = response.json()
        if rows:
            pieces.append(pd.DataFrame(rows))
        cursor = chunk_end + pd.Timedelta(days=1)
    if not pieces:
        raise ValueError("BCB returned no CDI data for the requested period")

    table = pd.concat(pieces).drop_duplicates("data")
    series = pd.Series(
        table["valor"].astype(float).to_numpy() / 100.0,  # percent per day -> decimal
        index=pd.to_datetime(table["data"], format="%d/%m/%Y"),
    ).sort_index()
    # 0.2%/day would already be ~65% a.a.; larger means a unit mistake.
    if not ((series >= 0) & (series < 0.002)).all():
        raise ValueError("BCB CDI values outside the plausible 0-0.2%/day range -- unit/format changed?")
    return series


def fetch_us_tbill_daily(start: str, end: str) -> pd.Series:
    """3-month T-bill as a per-trading-day decimal rate."""

    response = _http_get(
        "https://fred.stlouisfed.org/graph/fredgraph.csv",
        params={"id": FRED_US_SERIES, "cosd": pd.Timestamp(start).date().isoformat(), "coed": pd.Timestamp(end).date().isoformat()},
    )
    table = pd.read_csv(io.StringIO(response.text))
    date_col, value_col = table.columns[0], FRED_US_SERIES
    if value_col not in table.columns:
        raise ValueError(f"FRED response missing column {value_col}: {list(table.columns)}")
    annual_pct = pd.to_numeric(table[value_col], errors="coerce")  # "." marks holidays
    series = pd.Series(annual_pct.to_numpy(), index=pd.to_datetime(table[date_col])).dropna().sort_index()
    if len(series) == 0:
        raise ValueError("FRED returned no T-bill data for the requested period")
    if not ((series > -1.0) & (series < 25.0)).all():
        raise ValueError("FRED T-bill values outside the plausible -1..25% a.a. range -- unit/format changed?")
    return annual_to_daily(series / 100.0)


def _cached_fetch(name: str, fetcher, start: str, end: str) -> pd.Series:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}_{start}_{end}.pkl"
    if path.exists():
        return pd.read_pickle(path)
    series = fetcher(start, end)
    series.to_pickle(path)
    return series


def risk_free_daily(ticker: str, index: pd.DatetimeIndex) -> tuple[pd.Series, str]:
    """Per-day cash rate aligned to `index` (forward-filled across days the
    source has no print, e.g. a holiday in one calendar but not the other),
    plus a label naming the source -- 'ASSUMIDA' when it fell back.
    """

    brazilian = ticker.endswith(".SA")
    start, end = index.min().date().isoformat(), index.max().date().isoformat()
    try:
        if brazilian:
            raw, label = _cached_fetch("cdi", fetch_cdi_daily, start, end), f"BCB CDI (SGS {BCB_CDI_SERIES})"
        else:
            raw, label = _cached_fetch("tbill3m", fetch_us_tbill_daily, start, end), f"FRED {FRED_US_SERIES}"
        aligned = raw.reindex(raw.index.union(index)).sort_index().ffill().bfill().reindex(index)
        return aligned, label
    except Exception as exc:  # noqa: BLE001 -- any source failure must degrade loudly, not crash the analysis
        rate = CASH_ANNUAL_RATE_ASSUMED["BR" if brazilian else "US"]
        print(f"  AVISO: taxa livre de risco real indisponivel para {ticker} ({exc}); usando taxa ASSUMIDA de {rate:.1%} a.a.")
        return pd.Series(annual_to_daily(rate), index=index), f"ASSUMIDA {rate:.1%} a.a. (fonte oficial indisponivel)"
