"""Runs Stage 5: integrated backtest with regime segmentation (bull vs
named crisis windows) and the full metric set the spec requires (return,
Sharpe, Sortino, max drawdown, win rate), compared against the 3 Stage 2
baselines.

This does NOT re-implement the pipeline -- it reuses the exact same
building blocks Stage 4 already validated (_get_oos_predictions,
kelly_position_series, comparison_window, run_risk_managed_backtest) and
adds one new thing: slicing the resulting REALIZED returns by regime
(backtest/regime.py) for separate reporting, without re-running the
sequential engine on a discontinuous date range (see that module's
docstring for why that would be wrong).
"""

from __future__ import annotations

import pandas as pd

from backtest.baselines import buy_and_hold_signal
from backtest.plotting import plot_equity_curves
from backtest.regime import segment_returns_by_regime, summarize_regime_segment
from backtest.reporting import RESULTS_DIR, save_summary
from config import (
    ASSET_LEVEL_CIRCUIT_BREAKER_PCT,
    BACKTEST_START_DATE,
    CIRCUIT_BREAKER_DRAWDOWN_PCT,
    CRISIS_PERIODS,
    INITIAL_CAPITAL,
    KELLY_FRACTION,
    KELLY_MIN_LOSS_OBSERVATIONS,
    KELLY_MIN_WIN_OBSERVATIONS,
    KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
    PREDICTION_HORIZON_DAYS,
    SLIPPAGE_BPS,
    STAGE3_TICKERS,
    STOP_LOSS_PCT,
    SYNTHETIC_PERIODS,
    TRANSACTION_FEE_BPS,
    VAR_CONFIDENCE_LEVEL,
)
from data.real import load_all
from data.synthetic import generate_random_walk_prices
from models.comparison import comparison_window, predictions_to_backtest_signal
from risk.kelly import kelly_position_series
from risk.managed_backtest import run_risk_managed_backtest
from run_stage4_risk_management import (
    _asset_costs,
    _asset_level_circuit_breaker_threshold,
    _circuit_breaker_threshold,
    _get_oos_predictions,
)


def _run_regime_segmented_backtest(
    name: str, close: pd.Series, fee_bps: float, slippage_bps: float,
    circuit_breaker_pct: float, asset_circuit_breaker_pct: float | None,
    respect_holdout: bool,
) -> None:
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    predictions = _get_oos_predictions(close, respect_holdout=respect_holdout)
    comparison_close = comparison_window(close, predictions)

    kelly_positions = kelly_position_series(
        close, predictions["prob_xgboost"], horizon_days=PREDICTION_HORIZON_DAYS,
        kelly_fraction_multiplier=KELLY_FRACTION, win_loss_window_days=KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
        min_win_observations=KELLY_MIN_WIN_OBSERVATIONS, min_loss_observations=KELLY_MIN_LOSS_OBSERVATIONS,
    )
    kelly_signal_full = pd.Series(0.0, index=comparison_close.index)
    kelly_signal_full.update(kelly_positions.reindex(comparison_close.index))
    binary_signal = predictions_to_backtest_signal(comparison_close, predictions, "prob_xgboost")

    strategies = {
        "kelly_risk_managed": (kelly_signal_full, STOP_LOSS_PCT),
        "binario_risk_managed": (binary_signal, STOP_LOSS_PCT),
        "buy_and_hold_risk_managed": (buy_and_hold_signal(comparison_close.index), 0.99),
    }

    equity_curves = {}
    breaker_dates = {}
    regime_rows = []

    for strategy_name, (signal, stop_loss) in strategies.items():
        result = run_risk_managed_backtest(
            comparison_close, signal, fee_bps=fee_bps, slippage_bps=slippage_bps,
            initial_capital=INITIAL_CAPITAL, stop_loss_pct=stop_loss,
            circuit_breaker_drawdown_pct=circuit_breaker_pct,
            asset_circuit_breaker_drawdown_pct=asset_circuit_breaker_pct,
            var_confidence_level=VAR_CONFIDENCE_LEVEL,
        )
        equity_curves[strategy_name] = result.equity_curve
        breaker_dates[strategy_name] = result.circuit_breaker_date

        segments = segment_returns_by_regime(result.net_returns.iloc[1:], CRISIS_PERIODS)
        for regime_name, regime_returns in segments.items():
            metrics = summarize_regime_segment(regime_returns)
            metrics["estrategia"] = strategy_name
            metrics["regime"] = regime_name
            regime_rows.append(metrics)

        print(f"  [{strategy_name}] retorno total: {result.total_return:.4f}, "
              f"Sharpe: {result.annualized_sharpe:.3f}, Sortino: {result.sortino_ratio:.3f}, "
              f"max_drawdown: {result.max_drawdown:.4f}, win_rate: {result.win_rate:.4f}")

    regime_table = pd.DataFrame(regime_rows).set_index(["regime", "estrategia"]).sort_index()
    print(f"\n  Metricas por regime (bull vs janelas de crise conhecidas):")
    print(regime_table)

    plot_path = RESULTS_DIR / f"etapa5_equity_curve_{name}.png"
    plot_equity_curves(equity_curves, breaker_dates, f"Etapa 5 -- {name}", plot_path)
    print(f"  Grafico salvo em: {plot_path}")

    saved_path = save_summary(regime_table, stage_name=f"etapa5_{name}_regime")
    print(f"  Tabela por regime salva em: {saved_path}")


def main() -> None:
    synthetic_prices = generate_random_walk_prices(periods=SYNTHETIC_PERIODS)
    _run_regime_segmented_backtest(
        "synthetic", synthetic_prices["close"], TRANSACTION_FEE_BPS, SLIPPAGE_BPS,
        CIRCUIT_BREAKER_DRAWDOWN_PCT, ASSET_LEVEL_CIRCUIT_BREAKER_PCT, respect_holdout=False,
    )

    real_data = load_all(STAGE3_TICKERS, BACKTEST_START_DATE, None)
    for ticker, frame in real_data.items():
        fee_bps, slippage_bps = _asset_costs(ticker)
        circuit_breaker_pct = _circuit_breaker_threshold(ticker)
        asset_circuit_breaker_pct = _asset_level_circuit_breaker_threshold(ticker)
        _run_regime_segmented_backtest(
            ticker, frame["close"], fee_bps, slippage_bps,
            circuit_breaker_pct, asset_circuit_breaker_pct, respect_holdout=True,
        )


if __name__ == "__main__":
    main()
