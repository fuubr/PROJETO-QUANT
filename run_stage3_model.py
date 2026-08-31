"""Runs Stage 3: calibrated-probability AI model with purged walk-forward
validation.

Per the project spec, this script:
1. Builds a dataset (price-only features, forward-return label) for each
   configured ticker.
2. Runs purged walk-forward validation: for every fold, fit logistic
   regression + calibrated XGBoost on the (purged) training window,
   predict on the held-out test block. Aggregates ALL out-of-sample
   predictions across folds for evaluation -- no fold's test predictions
   are ever produced by a model that saw that fold's data.
3. Evaluates calibration (Brier score + reliability diagram) on the
   aggregated out-of-sample predictions.
4. Runs the same pipeline on the synthetic control asset: the gate for
   this stage is that out-of-sample performance there must be statistically
   equivalent to random. If it is not, this script stops and prints a
   warning instead of continuing to the real-asset comparison -- exactly as
   the spec requires ("pare e corrija antes de continuar").
5. Converts predicted probabilities into a binary trading signal
   (prob > 0.5) and backtests it through the same engine used in Stages
   1-2, for direct comparison against the 3 mandatory baselines.
6. Computes SHAP values on a final model refit on all non-holdout data, to
   see which features actually drive predictions.
7. Reserves everything from STAGE3_HOLDOUT_START_DATE onward as a final,
   untouched holdout -- walk-forward folds never include it. It is
   evaluated once, at the end, exactly as the spec requires.

Fractional/confidence-weighted position sizing (turning probability into
bet size) is explicitly OUT of scope here -- that's Stage 4 (Kelly). This
script uses a flat binary signal so the comparison against Stage 2's
baselines (also binary) is apples-to-apples.
"""

from __future__ import annotations

import shap
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.baselines import buy_and_hold_signal, moving_average_crossover_signal, random_entry_persistent_signal
from backtest.engine import run_backtest
from backtest.reporting import RESULTS_DIR, save_summary
from backtest.statistics import bonferroni_alpha, mean_return_significance, newey_west_t_stat, two_sided_p_value
from config import (
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
    STAGE3_SYNTHETIC_GATE_SEEDS,
    STAGE3_TICKERS,
    SYNTHETIC_PERIODS,
    TRANSACTION_FEE_BPS,
    WALK_FORWARD_MIN_TRAIN_DAYS,
    WALK_FORWARD_PURGE_DAYS,
    WALK_FORWARD_TEST_WINDOW_DAYS,
)
from data.real import load_all, resolve_end_date
from data.synthetic import generate_random_walk_prices
from models.calibration import train_models, train_xgboost_for_interpretation
from models.comparison import comparison_window, predictions_to_backtest_signal
from models.dataset import build_dataset, feature_columns
from models.evaluation import (
    brier_score,
    expected_calibration_error,
    max_calibration_gap,
    print_calibration_report,
    reliability_diagram_data,
)
from models.features import check_feature_correlation
from models.walk_forward import generate_walk_forward_splits


def _asset_costs(ticker: str) -> tuple[float, float]:
    return ASSET_COST_OVERRIDES.get(ticker, (TRANSACTION_FEE_BPS, SLIPPAGE_BPS))


def _run_walk_forward(dataset: pd.DataFrame) -> pd.DataFrame:
    """Run purged walk-forward validation over a dataset, returning
    out-of-sample predictions from every fold concatenated together (each
    row's prediction always comes from a model that never saw that row
    during training or calibration).
    """

    feature_cols = feature_columns(dataset)
    splits = generate_walk_forward_splits(
        n=len(dataset),
        min_train_size=WALK_FORWARD_MIN_TRAIN_DAYS,
        test_size=WALK_FORWARD_TEST_WINDOW_DAYS,
        purge_size=WALK_FORWARD_PURGE_DAYS,
    )

    if not splits:
        raise ValueError(
            f"not enough data for even one walk-forward fold "
            f"(have {len(dataset)} rows, need at least "
            f"{WALK_FORWARD_MIN_TRAIN_DAYS + WALK_FORWARD_PURGE_DAYS + WALK_FORWARD_TEST_WINDOW_DAYS})"
        )

    all_predictions = []
    for fold_number, (train_idx, test_idx) in enumerate(splits):
        X_train = dataset.iloc[train_idx][feature_cols]
        y_train = dataset.iloc[train_idx]["label"]
        X_test = dataset.iloc[test_idx][feature_cols]
        y_test = dataset.iloc[test_idx]["label"]

        trained = train_models(
            X_train, y_train, calibration_fraction=CALIBRATION_FRACTION, seed=SEED,
            min_samples_for_isotonic=CALIBRATION_METHOD_MIN_SAMPLES_FOR_ISOTONIC,
        )

        fold_predictions = pd.DataFrame(
            {
                "label": y_test.to_numpy(),
                "prob_logistic": trained.logistic_regression.predict_proba(X_test)[:, 1],
                "prob_xgboost": trained.xgboost_calibrated.predict_proba(X_test)[:, 1],
                "fold": fold_number,
            },
            index=X_test.index,
        )
        all_predictions.append(fold_predictions)
        print(f"  Fold {fold_number}: treino={len(train_idx)} dias, teste={len(test_idx)} dias, "
              f"calibracao={trained.calibration_method}")

    return pd.concat(all_predictions)


def _evaluate_calibration(predictions: pd.DataFrame, model_column: str, label: str) -> None:
    print_calibration_report(predictions, model_column, label, n_bins=CALIBRATION_DIAGRAM_BINS)


def _significance_gate_on_synthetic(
    predictions: pd.DataFrame, model_column: str, num_tests: int = 1
) -> bool:
    """Gate required by the spec: out-of-sample performance on the
    synthetic control must be statistically equivalent to random. Uses
    accuracy vs the 50% base rate as the simplest, most direct test of "is
    this model finding real structure in an asset that has none by
    construction".

    Uses Newey-West (HAC), not a naive t-test: consecutive rows within the
    same walk-forward test block have heavily OVERLAPPING label windows
    (each label looks PREDICTION_HORIZON_DAYS ahead, so neighboring labels
    share almost their entire horizon). That makes the correct/incorrect
    sequence strongly autocorrelated, and a naive i.i.d. standard error
    underestimates the true variance -- the same mistake already identified
    and fixed for the Stage 1/2 baseline comparisons, which had to be
    corrected here too.

    num_tests: pass the total number of simultaneous gate checks (e.g.
    num_seeds * num_models) to apply a Bonferroni correction across them.
    Running the gate against multiple seeds without this correction repeats
    the exact same multiple-comparisons mistake already fixed twice
    elsewhere in this project -- found here too during development, when a
    single seed's "failure" needed to be checked against whether it was
    just one false positive among many simultaneous tests (it was not, in
    the case that triggered this fix; see the Obsidian decision note).
    """

    y_true = predictions["label"].to_numpy()
    y_pred = (predictions[model_column].to_numpy() > 0.5).astype(float)
    correct = (y_pred == y_true).astype(float)

    accuracy = correct.mean()
    # Center at 0 (correct - 0.5) so newey_west_t_stat, which tests
    # mean != 0, tests accuracy != 0.5.
    t_statistic = newey_west_t_stat(correct - 0.5)
    p_value = two_sided_p_value(t_statistic)
    corrected_alpha = bonferroni_alpha(0.05, num_tests)

    print(f"  Acuracia fora da amostra no sintetico: {accuracy:.4f} (esperado: ~0.50)")
    print(f"  t-stat (Newey-West) vs 50%: {t_statistic:.3f}, p={p_value:.5f}, "
          f"alpha corrigido (Bonferroni, {num_tests} testes): {corrected_alpha:.5f}")

    return bool(p_value >= corrected_alpha)


def _run_shap_analysis(dataset: pd.DataFrame, ticker: str) -> None:
    feature_cols = feature_columns(dataset)
    X = dataset[feature_cols]
    y = dataset["label"]

    correlated_pairs = check_feature_correlation(X, threshold=0.7)
    if not correlated_pairs.empty:
        print(f"  AVISO: pares de features com |correlacao| >= 0.7 (distorce leitura do SHAP "
              f"e desestabiliza coeficientes da regressao logistica):")
        print(correlated_pairs.to_string(index=False))
    else:
        print("  Nenhum par de features com |correlacao| >= 0.7.")

    # Trained on 100% of dev data -- unlike train_models' xgboost_raw (fit
    # on the ~80% "fit" portion, holding back a calibration slice), which
    # would understate how much data this interpretation is based on.
    xgboost_model = train_xgboost_for_interpretation(X, y, seed=SEED)

    explainer = shap.TreeExplainer(xgboost_model)
    shap_values = explainer.shap_values(X)

    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_cols).sort_values(ascending=False)
    print(f"  Importancia media (|SHAP|) por feature, {ticker}:")
    print(mean_abs_shap.to_string())

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # shap.summary_plot creates its own internal figure when called without
    # an explicit ax. Pre-creating a fig/ax and then calling plt.close(fig)
    # would close the empty pre-created figure -- NOT the SHAP figure --
    # leaving SHAP's figure open and accumulating across multiple tickers.
    # Instead, capture the current figure AFTER summary_plot with plt.gcf().
    shap.summary_plot(shap_values, X, show=False, plot_size=(8, 5))
    fig = plt.gcf()
    plt.tight_layout()
    output_path = RESULTS_DIR / f"shap_summary_{ticker}.png"
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    print(f"  Grafico SHAP salvo em: {output_path}")


def run_for_asset(
    name: str, close: pd.Series, fee_bps: float, slippage_bps: float, is_synthetic: bool,
) -> pd.DataFrame | None:
    """Returns the out-of-sample predictions DataFrame for synthetic runs
    (so main() can pool predictions across seeds into one well-powered
    test), or None for real-asset runs.
    """

    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")

    if not is_synthetic:
        holdout_start = pd.Timestamp(STAGE3_HOLDOUT_START_DATE)
        dev_close = close[close.index < holdout_start]
        holdout_close = close[close.index >= holdout_start]
        print(f"Holdout final reservado a partir de {STAGE3_HOLDOUT_START_DATE}: "
              f"{len(holdout_close)} dias nunca usados no walk-forward.")
    else:
        dev_close = close
        holdout_close = None

    dataset = build_dataset(
        dev_close,
        momentum_windows=FEATURE_MOMENTUM_WINDOWS,
        volatility_window=FEATURE_VOLATILITY_WINDOW,
        sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW,
        horizon_days=PREDICTION_HORIZON_DAYS,
    )
    print(f"Dataset (dev, pos-warmup, pos-drop de rotulo futuro indisponivel): {len(dataset)} linhas")

    predictions = _run_walk_forward(dataset)
    print(f"Predicoes fora da amostra agregadas de todos os folds: {len(predictions)} linhas")

    print("\nCalibracao (fora da amostra, todos os folds agregados):")
    _evaluate_calibration(predictions, "prob_logistic", "logistic_regression")
    _evaluate_calibration(predictions, "prob_xgboost", "xgboost_calibrado")

    if is_synthetic:
        print("\nDiagnostico por seed (informativo -- a decisao do gate usa TODAS "
              "as seeds agrupadas, nao cada seed isoladamente; ver justificativa "
              "no README: testar cada seed separadamente com poucas observacoes "
              "da baixo poder estatistico e alta variancia por acaso, o que "
              "produz falhas espurias frequentes sem indicar problema real):")
        for column, label in [("prob_logistic", "logistic_regression"), ("prob_xgboost", "xgboost_calibrado")]:
            y_true = predictions["label"].to_numpy()
            y_pred = (predictions[column].to_numpy() > 0.5).astype(float)
            accuracy = (y_pred == y_true).mean()
            print(f"  [{label}] acuracia fora da amostra nesta seed: {accuracy:.4f}")
        return predictions

    print("\nBacktest do sinal do modelo vs baselines da Etapa 2 (mesma janela -- "
          "apenas o periodo com predicoes fora da amostra disponiveis, para "
          "nao penalizar o modelo com o periodo de aquecimento OU com o "
          "resto de dias sem predicao no final do walk-forward):")
    comparison_close = comparison_window(dev_close, predictions)
    print(f"Janela de comparacao: {comparison_close.index.min().date()} a {comparison_close.index.max().date()} "
          f"({len(comparison_close)} dias -- baselines tambem restritos a esta janela, "
          f"nao ao periodo de desenvolvimento inteiro)")

    xgboost_signal = predictions_to_backtest_signal(comparison_close, predictions, "prob_xgboost")
    logistic_signal = predictions_to_backtest_signal(comparison_close, predictions, "prob_logistic")

    baseline_rows = []
    for baseline_name, signal in [
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
        significance = mean_return_significance(result.net_returns, num_tests=5)
        baseline_rows.append(
            {
                "estrategia": baseline_name,
                "total_return": round(result.total_return, 4),
                "annualized_sharpe": round(result.annualized_sharpe, 3),
                "max_drawdown": round(result.max_drawdown, 4),
                "total_costs_paid": round(result.total_costs_paid, 2),
                "t_stat_newey_west": round(significance["t_stat"], 3),
                "significant_bonferroni": significance["significant"],
            }
        )

    summary = pd.DataFrame(baseline_rows).set_index("estrategia")
    print(summary)
    saved_path = save_summary(summary, stage_name=f"etapa3_{name}_dev")
    print(f"Resultado salvo em: {saved_path}")

    if holdout_close is not None and len(holdout_close) > PREDICTION_HORIZON_DAYS:
        print(f"\nHoldout final (nunca visto ate agora, {STAGE3_HOLDOUT_START_DATE} em diante):")
        print(
            "Avaliacao formal do holdout requer re-treinar o modelo com "
            "TODO o periodo de desenvolvimento (ja feito para o SHAP "
            "abaixo) e rodar uma unica vez aqui -- deixado como proximo "
            "passo explicito, nao automatizado neste script, para que o "
            "holdout so seja tocado quando o time estiver certo de que o "
            "modelo esta pronto para avaliacao final (consultar antes de "
            "reexecutar repetidamente contra o holdout)."
        )

    print("\nSHAP (modelo final, refit em todo o periodo de desenvolvimento):")
    _run_shap_analysis(dataset, name)


def _validate_stage3_config() -> None:
    """Guards against a silent reintroduction of label leakage: if someone
    edits config.py and WALK_FORWARD_PURGE_DAYS ends up smaller than
    PREDICTION_HORIZON_DAYS, the purge would no longer fully remove
    training rows whose label window overlaps the test period -- the exact
    leakage this project's walk-forward purging exists to prevent. Fails
    loudly at startup instead of producing a silently-too-good result
    later.
    """

    if WALK_FORWARD_PURGE_DAYS < PREDICTION_HORIZON_DAYS:
        raise ValueError(
            f"WALK_FORWARD_PURGE_DAYS ({WALK_FORWARD_PURGE_DAYS}) is smaller "
            f"than PREDICTION_HORIZON_DAYS ({PREDICTION_HORIZON_DAYS}). This "
            f"would let training labels near the train/test boundary leak "
            f"information from the test period. Set "
            f"WALK_FORWARD_PURGE_DAYS >= PREDICTION_HORIZON_DAYS in config.py."
        )


def main() -> None:
    _validate_stage3_config()

    print(f"\nGate estatistico no ativo sintetico, agregando "
          f"{len(STAGE3_SYNTHETIC_GATE_SEEDS)} seeds: {list(STAGE3_SYNTHETIC_GATE_SEEDS)}")
    print(
        "Design: as predicoes fora da amostra de todas as seeds sao "
        "AGRUPADAS antes do teste de significancia (nao testadas seed por "
        "seed com exigencia de que todas passem). Cada seed isolada tem "
        "poucas observacoes efetivas (autocorrelacao por sobreposicao de "
        "horizonte reduz ainda mais a amostra independente), entao testar "
        "cada uma separadamente tem baixo poder estatistico e alta chance "
        "de falha espuria por acaso -- analogo a exigir que cada estudo "
        "pequeno de uma meta-analise seja individualmente significativo, "
        "em vez de agregar a evidencia. Seeds sao series independentes, "
        "entao agrupar e estatisticamente valido."
    )

    all_predictions = []
    for seed in STAGE3_SYNTHETIC_GATE_SEEDS:
        synthetic_prices = generate_random_walk_prices(periods=SYNTHETIC_PERIODS, seed=seed)
        predictions = run_for_asset(
            f"synthetic (seed={seed})", synthetic_prices["close"],
            TRANSACTION_FEE_BPS, SLIPPAGE_BPS, is_synthetic=True,
        )
        predictions = predictions.copy()
        predictions["seed"] = seed
        all_predictions.append(predictions)

    pooled_predictions = pd.concat(all_predictions, ignore_index=False)
    print(f"\n{'=' * 70}\nGate agregado ({len(pooled_predictions)} predicoes fora da amostra, "
          f"todas as seeds)\n{'=' * 70}")

    gate_passed_logistic = _significance_gate_on_synthetic(pooled_predictions, "prob_logistic", num_tests=2)
    gate_passed_xgboost = _significance_gate_on_synthetic(pooled_predictions, "prob_xgboost", num_tests=2)
    gate_passed = gate_passed_logistic and gate_passed_xgboost

    if not gate_passed:
        print(
            "\n*** PARANDO: gate agregado falhou. Por definicao do projeto, "
            "isso e tratado como BUG (vazamento de dados ou overfitting) "
            "ate prova em contrario. Nao avaliar ativos reais ate isso ser "
            "investigado e corrigido. ***"
        )
        return

    print("\nGate agregado passou. Seguindo para ativos reais.")

    real_data = load_all(STAGE3_TICKERS, BACKTEST_START_DATE, None)
    for ticker, frame in real_data.items():
        fee_bps, slippage_bps = _asset_costs(ticker)
        run_for_asset(ticker, frame["close"], fee_bps, slippage_bps, is_synthetic=False)


if __name__ == "__main__":
    main()
