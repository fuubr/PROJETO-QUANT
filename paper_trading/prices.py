"""Canonical, append-only, chain-linked price series for paper trading.

Why this exists (real defect found by comparing logged vs. re-downloaded
prices: up to 2.7% apart for the same dates): yfinance adjusts history
retroactively (dividends, revisions). If decisions, logs and replays each
re-download history, (a) nothing is reproducible -- the consistency check
recomputes from data the decision never saw -- and (b) mixing a stale logged
price with a fresh one books a dividend as a price drop.

The canonical series fixes both. It is built ONCE, then only ever APPENDED:
each new bar = previous canonical value x (fresh[today] / fresh[previous
bar]), a ratio taken inside a single download so it is immune to later
re-adjustment and includes dividends (total return). Old bars are never
rewritten, so every day's inputs can be reproduced exactly, forever.
The level is an index, not a quotable price; only ratios matter.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICES_DIR = Path(__file__).resolve().parent / "prices"


def prices_path(ticker: str) -> Path:
    return PRICES_DIR / f"{ticker.replace('/', '-').replace('^', '')}.csv"


def load_canonical(ticker: str) -> pd.Series | None:
    path = prices_path(ticker)
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)["close"]


def _save(ticker: str, series: pd.Series) -> None:
    PRICES_DIR.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_csv(prices_path(ticker), index_label="date")


def bootstrap_canonical(fresh: pd.Series, log: pd.DataFrame | None) -> pd.Series:
    """First-time construction. With no paper-trading log yet, it is simply
    the fresh history. With an existing log (migration), the logged closes
    are kept verbatim for the live period -- they are exactly what past
    decisions used -- and the older history is rescaled to join them.
    """

    if log is None or len(log) == 0:
        return fresh.copy()
    first = log.index[0]
    pre = fresh[fresh.index < first]
    scale = float(log["close"].iloc[0]) / float(fresh.loc[first])
    return pd.concat([pre * scale, log["close"]]).sort_index()


def update_canonical(ticker: str, fresh: pd.Series, log: pd.DataFrame | None = None) -> pd.Series:
    """Append bars newer than the stored series using same-download ratios.
    Returns the (possibly extended) canonical series and persists it.
    """

    canonical = load_canonical(ticker)
    if canonical is None:
        canonical = bootstrap_canonical(fresh, log)
        _save(ticker, canonical)
        # Do NOT return yet: when migrating from an existing log, bars newer
        # than the last logged day must be chained on right away, otherwise
        # the first run would silently skip a trading day.

    last = canonical.index[-1]
    if last not in fresh.index:
        raise ValueError(
            f"{ticker}: last canonical date {last.date()} is missing from the fresh download; "
            f"refusing to guess the link between old and new data"
        )
    new_dates = fresh.index[fresh.index > last]
    if len(new_dates) == 0:
        return canonical

    value, prev_date, rows = float(canonical.iloc[-1]), last, {}
    for date in new_dates:
        value *= float(fresh.loc[date]) / float(fresh.loc[prev_date])
        rows[date] = value
        prev_date = date
    canonical = pd.concat([canonical, pd.Series(rows)]).sort_index().rename_axis("date").rename("close")
    _save(ticker, canonical)
    return canonical
