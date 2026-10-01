"""Disk cache for walk-forward out-of-sample predictions.

Stages 3-5 and the baseline/ablation analysis all recompute the same
expensive walk-forward (many XGBoost fits per asset). The key hashes the
exact price data AND every config value that changes the result, so a
cache hit is only possible when the output would be identical -- it can
never serve a stale answer after a config or data change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "cache" / "oos"


def cache_key(close: pd.Series, config_values: dict) -> str:
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(close, index=True).to_numpy().tobytes())
    h.update(json.dumps(config_values, sort_keys=True, default=str).encode())
    return h.hexdigest()[:24]


def load_cached(key: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{key}.pkl"
    return pd.read_pickle(path) if path.exists() else None


def save_cached(key: str, predictions: pd.DataFrame) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    predictions.to_pickle(CACHE_DIR / f"{key}.pkl")
