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
