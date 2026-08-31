"""Persists backtest summary tables to disk, so results from different runs
(and different stages) can be compared later without relying on manually
copying console output.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def save_summary(summary: pd.DataFrame, stage_name: str) -> Path:
    """Save a summary DataFrame to results/<stage_name>_<timestamp>.csv and
    return the path written.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"{stage_name}_{timestamp}.csv"
    summary.to_csv(path)
    return path
