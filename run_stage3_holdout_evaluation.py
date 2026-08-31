"""Runs the FINAL holdout evaluation for Stage 3 -- the single, terminal
check of whether the model's development-period performance generalizes to
genuinely unseen data.

*** THIS SHOULD ONLY BE RUN ONCE, WHEN YOU ARE CONFIDENT THE MODEL IS DONE
BEING DEVELOPED. *** Repeatedly running this script and adjusting the model
based on its output defeats the entire purpose of a holdout: it becomes
just another validation set the model gets implicitly tuned against, no
different from the walk-forward folds. If you find yourself running this
more than once for the same model configuration, stop and treat that as a
signal to reconsider the process, not just the model.

Design (leak-free by construction):
1. Load the FULL close-price history for each Stage 3 ticker.
2. Split into dev (< STAGE3_HOLDOUT_START_DATE) and holdout (>=) exactly as
   run_stage3_model.py does.
3. Build the TRAINING dataset from dev_close ALONE (not the full series).
   Because dev_close is truncated at the holdout boundary, any row whose
   forward-looking label would need to peek past that boundary is
   automatically NaN (build_dataset's own dropna removes it) -- the same
   mechanism already relied on throughout this project, just applied at the
   dev/holdout boundary instead of a walk-forward fold boundary. This
   dataset never sees a single holdout price, directly or via its labels.
4. Build a SEPARATE dataset from the FULL close series (dev + holdout) for
   evaluation purposes only. Its features for holdout dates may legitimately
   use trailing dev-period prices (that's just normal warmup, not leakage);
   its labels for holdout dates use real subsequent prices, which is
   exactly what an honest evaluation requires. Only rows dated on/after the
   holdout boundary are kept from this dataset -- it is never used for
   training.
5. Train the final model ONCE on the full training dataset (all of it, no
   walk-forward folds -- there is only one holdout, evaluated only once).
6. Predict on the holdout dataset, evaluate calibration, and backtest the
   resulting signal against the mandatory baselines, restricted to the same
   comparison window on both ends (same fix applied to the dev-period
   comparison in run_stage3_model.py).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from backtest.baselines import buy_and_hold_signal, moving_average_crossover_signal, random_entry_persistent_signal
from backtest.engine import run_backtest
from backtest.reporting import RESULTS_DIR, save_summary
from backtest.statistics import mean_return_significance
from config import (
    ASSET_CIRCUIT_BREAKER_OVERRIDES as ASSET_CIRCUIT_BREAKER_OVERRIDES_,
    ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES as ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES_,
    ASSET_LEVEL_CIRCUIT_BREAKER_PCT,
    CIRCUIT_BREAKER_DRAWDOWN_PCT,
    KELLY_FRACTION,
    KELLY_MIN_LOSS_OBSERVATIONS,
    KELLY_MIN_WIN_OBSERVATIONS,
    KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
    STOP_LOSS_PCT,
    VAR_CONFIDENCE_LEVEL,
    ASSET_COST_OVERRIDES,
    BACKTEST_START_DATE,
    CALIBRATION_DIAGRAM_BINS,
    CALIBRATION_FRACTION,
    CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC,
    FEATURE_MOMENTUM_WINDOWS,
    FEATURE_SMA_DISTANCE_WINDOW,
    FEATURE_VOLATILITY_WINDOW,
    INITIAL_CAPITAL,
    PREDICTION_HORIZON_DAYS,
    RANDOM_ENTRY_AVG_HOLDING_DAYS,
    RANDOM_ENTRY_PROBABILITY,
    SEED,
    SLIPPAGE_BPS,
    SMA_FAST_WINDOW,
    SMA_SLOW_WINDOW,
    STAGE3_HOLDOUT_START_DATE,
    STAGE3_TICKERS,
    TRANSACTION_FEE_BPS,
)
from data.real import load_all
from risk.kelly import kelly_position_series
from risk.managed_backtest import run_risk_managed_backtest
from models.calibration import train_models
from models.comparison import comparison_window, predictions_to_backtest_signal
from models.dataset import build_dataset, feature_columns
from models.evaluation import (
    brier_score,
    expected_calibration_error,
    max_calibration_gap,
    print_calibration_report,
    reliability_diagram_data,
)


def _asset_costs(ticker: str) -> tuple[float, float]:
    return ASSET_COST_OVERRIDES.get(ticker, (TRANSACTION_FEE_BPS, SLIPPAGE_BPS))


def _build_training_and_holdout_datasets(
    close: pd.Series, holdout_start: pd.Timestamp
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (training_dataset, holdout_dataset), built with zero leakage
    between them -- see module docstring for the full reasoning.
    """

    dev_close = close[close.index < holdout_start]

    training_dataset = build_dataset(
        dev_close,
        momentum_windows=FEATURE_MOMENTUM_WINDOWS,
        volatility_window=FEATURE_VOLATILITY_WINDOW,
        sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW,
        horizon_days=PREDICTION_HORIZON_DAYS,
    )

    full_dataset = build_dataset(
        close,
        momentum_windows=FEATURE_MOMENTUM_WINDOWS,
        volatility_window=FEATURE_VOLATILITY_WINDOW,
        sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW,
        horizon_days=PREDICTION_HORIZON_DAYS,
    )
    holdout_dataset = full_dataset[full_dataset.index >= holdout_start]

    return training_dataset, holdout_dataset


def _evaluate_calibration(predictions: pd.DataFrame, model_column: str, label: str) -> None:
    print_calibration_report(predictions, model_column, label, n_bins=CALIBRATION_DIAGRAM_BINS)


def run_holdout_evaluation_for_asset(name: str, close: pd.Series, fee_bps: float, slippage_bps: float) -> None:
    print(f"\n{'=' * 70}\n{name} -- AVALIACAO FINAL DE HOLDOUT\n{'=' * 70}")

    holdout_start = pd.Timestamp(STAGE3_HOLDOUT_START_DATE)
    training_dataset, holdout_dataset = _build_training_and_holdout_datasets(close, holdout_start)

    print(f"Dataset de treino (dev, {training_dataset.index.min().date()} a "
          f"{training_dataset.index.max().date()}): {len(training_dataset)} linhas")
    print(f"Dataset de holdout ({holdout_dataset.index.min().date() if len(holdout_dataset) else 'N/A'} a "
          f"{holdout_dataset.index.max().date() if len(holdout_dataset) else 'N/A'}): {len(holdout_dataset)} linhas")

    if len(holdout_dataset) < 30:
        print("*** Holdout com menos de 30 observacoes -- insuficiente para "
              "avaliacao com algum poder estatistico. Abortando. ***")
        return

    feature_cols = feature_columns(training_dataset)
    X_train = training_dataset[feature_cols]
    y_train = training_dataset["label"]
    X_holdout = holdout_dataset[feature_cols]

    trained = train_models(
        X_train, y_train, calibration_fraction=CALIBRATION_FRACTION, seed=SEED,
        min_samples_for_isotonic=CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC,
    )
    print(f"Metodo de calibracao usado: {trained.calibration_method}")

    predictions = pd.DataFrame(
        {
            "label": holdout_dataset["label"].to_numpy(),
            "prob_logistic": trained.logistic_regression.predict_proba(X_holdout)[:, 1],
            "prob_xgboost": trained.xgboost_calibrated.predict_proba(X_holdout)[:, 1],
        },
        index=holdout_dataset.index,
    )

    print("\nCalibracao no holdout:")
    _evaluate_calibration(predictions, "prob_logistic", "logistic_regression")
    _evaluate_calibration(predictions, "prob_xgboost", "xgboost_calibrado")

    print("\nBacktest no holdout vs baselines (mesma janela nos dois lados):")
    comparison_close = comparison_window(close, predictions)
    print(f"Janela: {comparison_close.index.min().date()} a {comparison_close.index.max().date()} "
          f"({len(comparison_close)} dias)")

    xgboost_signal = predictions_to_backtest_signal(comparison_close, predictions, "prob_xgboost")
    logistic_signal = predictions_to_backtest_signal(comparison_close, predictions, "prob_logistic")

    kelly_signal = kelly_position_series(
        close, predictions["prob_xgboost"], horizon_days=PREDICTION_HORIZON_DAYS,
        kelly_fraction_multiplier=KELLY_FRACTION, win_loss_window_days=KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
        min_win_observations=KELLY_MIN_WIN_OBSERVATIONS, min_loss_observations=KELLY_MIN_LOSS_OBSERVATIONS,
    ).reindex(comparison_close.index).fillna(0.0)
    circuit_breaker_pct = ASSET_CIRCUIT_BREAKER_OVERRIDES_.get(name, CIRCUIT_BREAKER_DRAWDOWN_PCT)
    asset_breaker_pct = ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES_.get(name, ASSET_LEVEL_CIRCUIT_BREAKER_PCT)
    kelly_result = run_risk_managed_backtest(
        comparison_close, kelly_signal, fee_bps=fee_bps, slippage_bps=slippage_bps,
        initial_capital=INITIAL_CAPITAL, stop_loss_pct=STOP_LOSS_PCT,
        circuit_breaker_drawdown_pct=circuit_breaker_pct, var_confidence_level=VAR_CONFIDENCE_LEVEL,
        asset_circuit_breaker_drawdown_pct=asset_breaker_pct,
    )
    kelly_significance = mean_return_significance(kelly_result.net_returns, num_tests=6)
    rows = [{
        "estrategia": "kelly_risk_managed (Etapa 4, pipeline final)",
        "total_return": round(kelly_result.total_return, 4),
        "annualized_sharpe": round(kelly_result.annualized_sharpe, 3),
        "max_drawdown": round(kelly_result.max_drawdown, 4),
        "total_costs_paid": round(kelly_result.total_costs_paid, 2),
        "t_stat_newey_west": round(kelly_significance["t_stat"], 3),
        "significant_bonferroni": kelly_significance["significant"],
    }]
    for strategy_name, signal in [
        ("modelo (xgboost calibrado)", xgboost_signal),
        ("modelo (logistic_regression)", logistic_signal),
        ("buy_and_hold", buy_and_hold_signal(comparison_close.index)),
        ("aleatorio", random_entry_persistent_signal(
            comparison_close.index, seed=SEED, probability=RANDOM_ENTRY_PROBABILITY,
            avg_holding_days=RANDOM_ENTRY_AVG_HOLDING_DAYS,
        )),
        ("media_movel", moving_average_crossover_signal(comparison_close, fast_window=SMA_FAST_WINDOW, slow_window=SMA_SLOW_WINDOW)),
    ]:
        result = run_backtest(
            comparison_close, signal, fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
        )
        significance = mean_return_significance(result.net_returns, num_tests=6)
        rows.append(
            {
                "estrategia": strategy_name,
                "total_return": round(result.total_return, 4),
                "annualized_sharpe": round(result.annualized_sharpe, 3),
                "max_drawdown": round(result.max_drawdown, 4),
                "total_costs_paid": round(result.total_costs_paid, 2),
                "t_stat_newey_west": round(significance["t_stat"], 3),
                "significant_bonferroni": significance["significant"],
            }
        )

    summary = pd.DataFrame(rows).set_index("estrategia")
    print(summary)

    saved_path = save_summary(summary, stage_name=f"etapa3_{name}_HOLDOUT_FINAL")
    print(f"\nResultado salvo em: {saved_path}")
    print(
        "\nLembrete: compare esta tabela com a tabela de desenvolvimento "
        "(results/etapa3_*_dev_*.csv) do mesmo modelo. Divergencia grande "
        "entre dev e holdout e mais importante de investigar do que o "
        "retorno do holdout isoladamente -- e o sintoma classico de "
        "overfitting que passou despercebido pelo walk-forward."
    )


def _existing_holdout_results() -> list[Path]:
    """Find any previously saved holdout result files, across all tickers
    and timestamps.
    """

    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("etapa3_*_HOLDOUT_FINAL_*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the FINAL Stage 3 holdout evaluation. Intended to run exactly once."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Required to re-run after a holdout result already exists. "
            "Re-running and adjusting the model based on the output defeats "
            "the purpose of a holdout -- this flag exists so that happens "
            "only on purpose, never by accident (e.g. re-running the same "
            "command from shell history)."
        ),
    )
    args = parser.parse_args()

    existing = _existing_holdout_results()
    if existing and not args.force:
        print("=" * 70)
        print("PARANDO: ja existe(m) resultado(s) de avaliacao de holdout salvos:")
        for path in existing:
            print(f"  {path}")
        print()
        print(
            "Rodar de novo sem revisar isso primeiro corre o risco de "
            "transformar o holdout em mais um conjunto de validacao "
            "implicito -- o que anula o proposito dele. Se voce tem certeza "
            "de que quer uma nova avaliacao final (ex: modelo genuinamente "
            "re-desenhado, nao so re-ajustado por causa do resultado "
            "anterior), rode de novo com --force."
        )
        print("=" * 70)
        sys.exit(1)

    print("=" * 70)
    print("AVISO: esta e a avaliacao FINAL de holdout. So deve ser rodada")
    print("uma vez, quando o modelo estiver pronto. Rodar repetidamente e")
    print("ajustar o modelo com base neste resultado anula o proposito do")
    print("holdout -- ele viraria so mais um conjunto de validacao.")
    print("=" * 70)

    real_data = load_all(STAGE3_TICKERS, BACKTEST_START_DATE, None)
    for ticker, frame in real_data.items():
        fee_bps, slippage_bps = _asset_costs(ticker)
        run_holdout_evaluation_for_asset(ticker, frame["close"], fee_bps, slippage_bps)


if __name__ == "__main__":
    main()
