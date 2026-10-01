"""Runs ONE day of paper trading for every configured ticker.

Meant to be run once per real trading day (e.g. via a scheduled task,
after market close so today's close price is available). Each run:

1. First time for a ticker: freezes a model (trained on all real data
   available up to today -- there is no future left to protect at this
   point, today IS the prospective boundary) and records day 1's state
   (flat, no prior position).
2. Every later run: reconstructs yesterday's RiskState from the log,
   advances it by exactly one day using step_risk_managed_backtest (the
   SAME function the batch backtest uses internally -- see
   risk/managed_backtest.py's module docstring for why that matters),
   computes tomorrow's target position from the frozen model, and appends
   one new row.

Idempotent: if today's date is already the log's last row, this is a
no-op (safe to run the scheduled task more than once on the same day by
accident).

The model is NEVER retrained here. Comparing this log's realized returns
against what a normal Stage 4/5 backtest would have predicted for the
same dates (once enough days accumulate) is the project's core Stage 6
metric -- large divergence matters more than the return itself, since it
usually means a bug slipped through backtesting, not that "the market
changed".
"""

from __future__ import annotations

import sys

import pandas as pd

from config import (
    ASSET_CIRCUIT_BREAKER_OVERRIDES,
    ASSET_COST_OVERRIDES,
    ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES,
    ASSET_LEVEL_CIRCUIT_BREAKER_PCT,
    BACKTEST_START_DATE,
    CALIBRATION_FRACTION,
    CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC,
    CIRCUIT_BREAKER_DRAWDOWN_PCT,
    FEATURE_MOMENTUM_WINDOWS,
    FEATURE_SMA_DISTANCE_WINDOW,
    FEATURE_VOLATILITY_WINDOW,
    KELLY_FRACTION,
    KELLY_MIN_LOSS_OBSERVATIONS,
    KELLY_MIN_WIN_OBSERVATIONS,
    KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
    PAPER_TRADING_INITIAL_CAPITAL,
    PAPER_TRADING_TICKERS,
    PREDICTION_HORIZON_DAYS,
    SEED,
    SLIPPAGE_BPS,
    STALENESS_WARNING_DAYS,
    STOP_LOSS_PCT,
    TRANSACTION_FEE_BPS,
)
from data.real import load_all
from models.calibration import train_models
from models.dataset import build_dataset, feature_columns
from models.features import build_features
from paper_trading.model_store import list_versions, model_for_date, save_frozen_model
from paper_trading.state import append_row, load_log, state_from_last_row
from risk.kelly import kelly_position_series
from risk.managed_backtest import RiskState, step_risk_managed_backtest


def _asset_costs(ticker: str) -> tuple[float, float]:
    return ASSET_COST_OVERRIDES.get(ticker, (TRANSACTION_FEE_BPS, SLIPPAGE_BPS))


def _circuit_breaker_thresholds(ticker: str) -> tuple[float, float | None]:
    portfolio_pct = ASSET_CIRCUIT_BREAKER_OVERRIDES.get(ticker, CIRCUIT_BREAKER_DRAWDOWN_PCT)
    asset_pct = ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES.get(ticker, ASSET_LEVEL_CIRCUIT_BREAKER_PCT)
    return portfolio_pct, asset_pct


def _today_target_position(close: pd.Series, trained) -> float:
    """Compute the Kelly-sized target position for the LATEST date in
    close, using the frozen model. Reuses kelly_position_series with a
    single-date probability series -- same causal logic as every other
    stage, just evaluated at one point instead of a whole history.

    Uses build_features directly (NOT build_dataset). Real bug found via
    the Stage 6 consistency check (paper_trading/consistency_check.py):
    build_dataset always requires a label, and build_forward_return_label
    sets the label to NaN for the last horizon_days rows since their
    future price doesn't exist yet -- build_dataset's dropna then silently
    drops those rows, INCLUDING today's, even with horizon_days=1 (today's
    row still needs at least 1 day of future price it doesn't have yet).
    The original code took dataset_features.index[-1] believing it was
    today, but it was actually YESTERDAY -- every daily target position
    was computed one full day stale from the day this script launched
    until this fix. build_features has no label and therefore no such
    drop: its last row is genuinely today, confirmed by test.
    """

    features = build_features(
        close, momentum_windows=FEATURE_MOMENTUM_WINDOWS, volatility_window=FEATURE_VOLATILITY_WINDOW,
        sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW,
    ).dropna()
    feature_cols = list(features.columns)
    today_date = features.index[-1]
    today_features = features.loc[[today_date], feature_cols]

    probability = float(trained.xgboost_calibrated.predict_proba(today_features)[:, 1][0])
    probability_series = pd.Series([probability], index=[today_date])

    positions = kelly_position_series(
        close, probability_series, horizon_days=PREDICTION_HORIZON_DAYS,
        kelly_fraction_multiplier=KELLY_FRACTION, win_loss_window_days=KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
        min_win_observations=KELLY_MIN_WIN_OBSERVATIONS, min_loss_observations=KELLY_MIN_LOSS_OBSERVATIONS,
    )
    return float(positions.iloc[-1])


def run_one_day_for_ticker(ticker: str, close: pd.Series) -> None:
    print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")

    fee_bps, slippage_bps = _asset_costs(ticker)
    circuit_breaker_pct, asset_circuit_breaker_pct = _circuit_breaker_thresholds(ticker)

    versions = list_versions(ticker)
    today_date = close.index[-1]
    today_close = float(close.iloc[-1])

    if not versions:
        print(f"  Primeira execucao para {ticker}. Congelando modelo com dados ate {today_date.date()}.")

        full_dataset = build_dataset(
            close, momentum_windows=FEATURE_MOMENTUM_WINDOWS, volatility_window=FEATURE_VOLATILITY_WINDOW,
            sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW, horizon_days=PREDICTION_HORIZON_DAYS,
        )
        feature_cols = feature_columns(full_dataset)
        trained = train_models(
            full_dataset[feature_cols], full_dataset["label"], calibration_fraction=CALIBRATION_FRACTION,
            seed=SEED, min_samples_for_isotonic=CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC,
        )
        save_frozen_model(ticker, trained, frozen_date=str(today_date.date()))

        target_position = _today_target_position(close, trained)

        state = RiskState.initial(PAPER_TRADING_INITIAL_CAPITAL, today_close)
        append_row(ticker, today_date, today_close, target_position, net_return=0.0, state=state)

        print(f"  Dia 1 registrado. Posicao alvo para amanha: {target_position:.4f}")
        return

    active = model_for_date(versions, today_date)
    trained = active.trained
    frozen_date = f"{versions[0].effective_from.date()} (versao ativa: v{active.version})"
    log = load_log(ticker)

    if today_date in log.index:
        print(f"  Ja processado hoje ({today_date.date()}) -- nada a fazer (script e idempotente).")
        return

    state, last_close, target_position_yesterday = state_from_last_row(log)
    last_date = log.index[-1]

    if today_date <= last_date:
        gap_days = (pd.Timestamp.now().normalize() - last_date.normalize()).days
        if gap_days > STALENESS_WARNING_DAYS:
            print(
                f"  *** AVISO: sem dado novo ha {gap_days} dias corridos (ultimo: "
                f"{last_date.date()}) -- isso passa do que um fim de semana ou feriado "
                f"isolado explicaria. Verificar se o feed de dados (yfinance) esta "
                f"funcionando para {ticker}. ***"
            )
        else:
            print(f"  Nenhum dado novo desde {last_date.date()} -- nada a fazer.")
        return

    new_state, net_return = step_risk_managed_backtest(
        state, price_yesterday=last_close, price_today=today_close,
        target_position_yesterday=target_position_yesterday, fee_bps=fee_bps, slippage_bps=slippage_bps,
        stop_loss_pct=STOP_LOSS_PCT, circuit_breaker_drawdown_pct=circuit_breaker_pct,
        today_date=today_date, asset_circuit_breaker_drawdown_pct=asset_circuit_breaker_pct,
    )

    new_target_position = _today_target_position(close, trained)
    append_row(ticker, today_date, today_close, new_target_position, net_return, new_state)

    print(f"  Modelo congelado em: {frozen_date}")
    print(f"  Retorno de hoje: {net_return:.4%}, equity: {new_state.equity:,.2f}")
    print(f"  Posicao mantida hoje: {new_state.held_position:.4f}, "
          f"posicao alvo para amanha: {new_target_position:.4f}")
    if new_state.circuit_breaker_triggered and not state.circuit_breaker_triggered:
        print(f"  *** CIRCUIT BREAKER DISPAROU HOJE ({new_state.circuit_breaker_reason}) ***")
    if new_state.stop_loss_triggers > state.stop_loss_triggers:
        print(f"  *** STOP-LOSS DISPAROU HOJE ***")


def main() -> None:
    real_data = load_all(PAPER_TRADING_TICKERS, BACKTEST_START_DATE, None)
    for ticker in PAPER_TRADING_TICKERS:
        if ticker not in real_data:
            print(f"AVISO: sem dados para {ticker}, pulando.")
            continue
        run_one_day_for_ticker(ticker, real_data[ticker]["close"])


if __name__ == "__main__":
    main()
