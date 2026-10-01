"""Persists the FROZEN model used for paper trading.

Trained exactly once, on the first day paper trading runs for a given
ticker, using all real data available up to that point -- there is no
"future" left to protect at that point (today IS the prospective
boundary), unlike every earlier stage's dev/holdout split. Never
retrained by this module; periodic retraining is Stage 7's job, kept
deliberately separate so it happens only as a conscious decision, not
silently every time the daily script runs.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from models.calibration import TrainedModels

MODEL_DIR = Path(__file__).resolve().parent / "models"


def model_path(ticker: str) -> Path:
    safe_ticker = ticker.replace("/", "-").replace("^", "")
    return MODEL_DIR / f"{safe_ticker}.joblib"


def save_frozen_model(ticker: str, trained: TrainedModels, frozen_date: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = model_path(ticker)
    joblib.dump({"trained": trained, "frozen_date": frozen_date}, path)
    return path


def load_frozen_model(ticker: str) -> tuple[TrainedModels, str] | None:
    """Returns (trained_models, frozen_date), or None if no frozen model
    exists yet for this ticker (first-run case).
    """

    path = model_path(ticker)
    if not path.exists():
        return None
    payload = joblib.load(path)
    return payload["trained"], payload["frozen_date"]


# --- Stage 7: model versioning -------------------------------------------
#
# The original frozen model (above) is version 1 and is NEVER overwritten.
# A retrained model becomes version 2, 3, ... only through an explicit
# promotion (see paper_trading/retrain.py) and is recorded in
# registry.csv with the first date it governs. Each decision date uses the
# latest version whose effective_from <= that date, so history is never
# rewritten and the backtest-vs-paper-trading consistency check can
# reproduce exactly which model made each day's decision.

import csv
from dataclasses import dataclass

import pandas as pd

REGISTRY_COLUMNS = ["ticker", "version", "effective_from", "trained_through", "filename"]


@dataclass
class ModelVersion:
    version: int
    effective_from: pd.Timestamp
    trained_through: pd.Timestamp
    trained: TrainedModels


def _registry_path() -> Path:
    return MODEL_DIR / "registry.csv"


def _safe(ticker: str) -> str:
    return ticker.replace("/", "-").replace("^", "")


def list_versions(ticker: str) -> list[ModelVersion]:
    """All versions for a ticker, oldest first. Empty if the ticker was
    never frozen (first-run case).
    """

    base = load_frozen_model(ticker)
    if base is None:
        return []
    trained, frozen_date = base
    frozen_ts = pd.Timestamp(frozen_date)
    versions = [ModelVersion(1, frozen_ts, frozen_ts, trained)]

    registry = _registry_path()
    if registry.exists():
        table = pd.read_csv(registry)
        for _, row in table[table["ticker"] == ticker].sort_values("version").iterrows():
            payload = joblib.load(MODEL_DIR / row["filename"])
            versions.append(
                ModelVersion(
                    int(row["version"]), pd.Timestamp(row["effective_from"]),
                    pd.Timestamp(row["trained_through"]), payload["trained"],
                )
            )
    return versions


def model_for_date(versions: list[ModelVersion], date: pd.Timestamp) -> ModelVersion | None:
    """The version governing a decision made on `date`: the latest one
    with effective_from <= date.
    """

    eligible = [v for v in versions if v.effective_from <= date]
    return eligible[-1] if eligible else None


def register_version(
    ticker: str, trained: TrainedModels, trained_through: pd.Timestamp, effective_from: pd.Timestamp
) -> ModelVersion:
    """Persist a new model version. Refuses to backdate: effective_from
    must be later than every existing version's, so already-logged
    decisions can never be silently re-attributed to a different model.
    """

    existing = list_versions(ticker)
    if not existing:
        raise ValueError(f"{ticker} has no frozen base model; cannot register a new version")
    if effective_from <= existing[-1].effective_from:
        raise ValueError("effective_from must be later than the current latest version's")

    version = existing[-1].version + 1
    filename = f"{_safe(ticker)}_v{version}.joblib"
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"trained": trained}, MODEL_DIR / filename)

    registry = _registry_path()
    new_file = not registry.exists()
    with registry.open("a", newline="") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(REGISTRY_COLUMNS)
        writer.writerow([ticker, version, effective_from.date(), trained_through.date(), filename])

    return ModelVersion(version, effective_from, trained_through, trained)
