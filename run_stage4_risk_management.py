"""Runs Stage 4: dynamic (Kelly-fractional) position sizing with stop-loss
and a portfolio-level circuit breaker, built on top of Stage 3's calibrated
probabilities.

Design notes specific to this stage:

1. Kelly sizing is NOT guaranteed to output 0 exposure when the model has
   no real edge (p ~= 0.5). The classic formula f* = p - (1-p)/b depends
   on b (win/loss ratio) too -- and b can drift away from 1.0 purely from
   sampling noise, even on the synthetic control asset, which has zero
   directional edge by construction. Verified empirically during
   development: at p=0.5 exactly, kelly_fraction(0.5, b) was nonzero (up
   to ~4%) for some random-walk seeds, purely from asymmetry in realized
   win/loss magnitudes. Because of this, the gate below tests the
   resulting BACKTEST RETURNS for statistical significance (the same
   Newey-West + Bonferroni, pooled-across-seeds methodology as Stage 3),
   not "is the average position size exactly zero".
2. Reuses Stage 3's walk-forward machinery (_run_walk_forward) directly
   rather than re-implementing it, avoiding the kind of duplication (and
   drift) already found and fixed once between run_stage3_model.py and
   run_stage3_holdout_evaluation.py.
"""

from __future__ import annotations

import pandas as pd

from backtest.baselines import buy_and_hold_signal
from backtest.plotting import plot_equity_curves
from backtest.reporting import RESULTS_DIR, save_summary
from backtest.statistics import mean_return_significance
from config import (
    ASSET_CIRCUIT_BREAKER_OVERRIDES,
    ASSET_COST_OVERRIDES,
    ASSET_LEVEL_CIRCUIT_BREAKER_PCT,
    ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES,
    BACKTEST_START_DATE,
    CIRCUIT_BREAKER_DRAWDOWN_PCT,
    INITIAL_CAPITAL,
    KELLY_FRACTION,
    KELLY_MIN_LOSS_OBSERVATIONS,
    KELLY_MIN_WIN_OBSERVATIONS,
    KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
    PREDICTION_HORIZON_DAYS,
    SLIPPAGE_BPS,
    STAGE3_HOLDOUT_START_DATE,
    STAGE3_TICKERS,
    STAGE3_SYNTHETIC_GATE_SEEDS,
    STOP_LOSS_PCT,
    SYNTHETIC_PERIODS,
    TRANSACTION_FEE_BPS,
    VAR_CONFIDENCE_LEVEL,
)
from data.real import load_all
from data.synthetic import generate_random_walk_prices
from models.comparison import comparison_window, predictions_to_backtest_signal
from models.dataset import build_dataset
from risk.kelly import kelly_position_series
from risk.managed_backtest import run_risk_managed_backtest
from run_stage3_model import FEATURE_MOMENTUM_WINDOWS, FEATURE_SMA_DISTANCE_WINDOW, FEATURE_VOLATILITY_WINDOW, _run_walk_forward


def _asset_costs(ticker: str) -> tuple[float, float]:
    return ASSET_COST_OVERRIDES.get(ticker, (TRANSACTION_FEE_BPS, SLIPPAGE_BPS))


def _circuit_breaker_threshold(ticker: str) -> float:
    return ASSET_CIRCUIT_BREAKER_OVERRIDES.get(ticker, CIRCUIT_BREAKER_DRAWDOWN_PCT)


def _asset_level_circuit_breaker_threshold(ticker: str) -> float | None:
    return ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES.get(ticker, ASSET_LEVEL_CIRCUIT_BREAKER_PCT)


def _get_oos_predictions(close: pd.Series, respect_holdout: bool) -> pd.DataFrame:
    """Rebuild the Stage 3 dataset and run purged walk-forward on it,
    returning aggregated out-of-sample predictions -- same pipeline Stage 3
    uses, factored out here so Stage 4 reuses it exactly rather than
    duplicating it.

    respect_holdout=True truncates close to the dev period (< STAGE3_
    HOLDOUT_START_DATE) before building anything -- required for real
    assets, since the walk-forward's expanding window otherwise keeps
    consuming data until it runs out, silently reaching into the reserved
    holdout period. This was a real bug found during development: the
    first version of this function used the full close series directly,
    letting Stage 4's walk-forward folds extend into 2024+ data that Stage
    3 had deliberately reserved and never touched.

    respect_holdout=False is used only for the synthetic control asset,
    which has no holdout concept (it's freshly generated, not a real
    historical series with a real future to protect).
    """

    if respect_holdout:
        holdout_start = pd.Timestamp(STAGE3_HOLDOUT_START_DATE)
        close = close[close.index < holdout_start]

    dataset = build_dataset(
        close,
        momentum_windows=FEATURE_MOMENTUM_WINDOWS,
        volatility_window=FEATURE_VOLATILITY_WINDOW,
        sma_distance_window=FEATURE_SMA_DISTANCE_WINDOW,
        horizon_days=PREDICTION_HORIZON_DAYS,
    )
    return _run_walk_forward(dataset)


def _active_window_signal_stats(signal: pd.Series, active_days: int) -> tuple[float, float]:
    """Mean and std of a signal restricted to its first active_days
    observations -- extracted as a pure function so the "measure only the
    active window, not the full series" logic (see the investigation note
    in _run_kelly_risk_managed) is directly testable.
    """

    window = signal.iloc[:active_days]
    mean = float(window.mean()) if active_days > 0 else float("nan")
    std = float(window.std()) if active_days > 1 else 0.0
    return mean, std


def _run_kelly_risk_managed(
    name: str, close: pd.Series, fee_bps: float, slippage_bps: float, circuit_breaker_pct: float,
    asset_circuit_breaker_pct: float | None,
) -> tuple[pd.DataFrame, dict]:
    """Returns (summary_table, gate_significance_dict) for one asset."""

    predictions = _get_oos_predictions(close, respect_holdout=True)

    kelly_positions = kelly_position_series(
        close,
        predictions["prob_xgboost"],
        horizon_days=PREDICTION_HORIZON_DAYS,
        kelly_fraction_multiplier=KELLY_FRACTION,
        win_loss_window_days=KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
        min_win_observations=KELLY_MIN_WIN_OBSERVATIONS,
        min_loss_observations=KELLY_MIN_LOSS_OBSERVATIONS,
    )

    comparison_close = comparison_window(close, predictions)
    kelly_signal_full = pd.Series(0.0, index=comparison_close.index)
    kelly_signal_full.update(kelly_positions.reindex(comparison_close.index))

    kelly_result = run_risk_managed_backtest(
        comparison_close, kelly_signal_full,
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=STOP_LOSS_PCT, circuit_breaker_drawdown_pct=circuit_breaker_pct,
        asset_circuit_breaker_drawdown_pct=asset_circuit_breaker_pct,
        var_confidence_level=VAR_CONFIDENCE_LEVEL,
    )

    # Etapa 3's plain binary model signal (prob > 0.5 -> fully invested),
    # for direct comparison: does dynamic sizing + risk overlays actually
    # help, or just add complexity?
    binary_signal = predictions_to_backtest_signal(comparison_close, predictions, "prob_xgboost")

    binary_result = run_risk_managed_backtest(
        comparison_close, binary_signal,
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
        stop_loss_pct=STOP_LOSS_PCT, circuit_breaker_drawdown_pct=circuit_breaker_pct,
        asset_circuit_breaker_drawdown_pct=asset_circuit_breaker_pct,
        var_confidence_level=VAR_CONFIDENCE_LEVEL,
    )

    buy_hold_result = run_risk_managed_backtest(
        comparison_close, buy_and_hold_signal(comparison_close.index),
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
        # Buy-and-hold gets the same circuit breaker (a fair risk-adjusted
        # comparison), but an unreachably loose stop-loss -- a "stop-loss"
        # on a strategy with no re-entry logic isn't a meaningful concept.
        stop_loss_pct=0.99, circuit_breaker_drawdown_pct=circuit_breaker_pct,
        asset_circuit_breaker_drawdown_pct=asset_circuit_breaker_pct,
        var_confidence_level=VAR_CONFIDENCE_LEVEL,
    )

    # Diagnostic, corrected after an investigation: measuring the signal
    # over the FULL comparison window was misleading whenever the circuit
    # breaker cuts activity short (e.g. AAPL: only the first 169 of 1386
    # days; PETR4.SA: only 37). The signal can vary substantially LATER in
    # the series -- in a period neither the binary nor buy-and-hold
    # strategy ever reaches once flattened -- while being constant during
    # the short ACTIVE window that actually determined both strategies'
    # returns. Measuring only the active window (up to
    # binary_result.active_trading_days, the shorter of the two
    # breaker-trigger points) directly tests the real explanation instead
    # of requiring the reader to infer it.
    active_days = min(binary_result.active_trading_days, buy_hold_result.active_trading_days)
    active_signal_mean, active_signal_std = _active_window_signal_stats(binary_signal, active_days)
    print(f"  Diagnostico [{name}]: sinal binario do modelo DURANTE A JANELA ATIVA "
          f"({active_days} dias, ate o circuit breaker mais cedo entre binario/buy_and_hold) "
          f"-- media={active_signal_mean:.4f}, desvio={active_signal_std:.4f} "
          f"({'CONSTANTE nesta janela -- explica retorno identico a buy-and-hold, nao e coincidencia' if active_signal_std < 1e-9 else 'varia mesmo dentro da janela ativa -- retorno identico exigiria outra explicacao'})")

    # Same "measure only the active window" correction, applied to the
    # Kelly position itself. The avg_position_size column in the summary
    # table averages over the FULL comparison window, including the flat
    # tail after Kelly's own circuit breaker fires -- which can make real
    # exposure right before a crash look deceptively tiny. Found during an
    # investigation of PETR4.SA: the table showed avg_position_size=0.0087
    # (looks negligible), but restricted to Kelly's own 46-day active
    # window, the real average exposure was ~0.26 (26%) -- not tiny at
    # all, and enough to still accumulate a real loss during a severe
    # crash even with substantially reduced sizing versus buy-and-hold's
    # 100%.
    kelly_active_mean, kelly_active_std = _active_window_signal_stats(
        kelly_signal_full, kelly_result.active_trading_days
    )
    kelly_active_max = float(kelly_signal_full.iloc[: kelly_result.active_trading_days].max()) if kelly_result.active_trading_days > 0 else float("nan")
    print(f"  Diagnostico [{name}]: exposicao Kelly DURANTE SUA PROPRIA JANELA ATIVA "
          f"({kelly_result.active_trading_days} dias) -- media={kelly_active_mean:.4f}, "
          f"maximo={kelly_active_max:.4f} (a media_de_posicao na tabela abaixo e sobre TODO "
          f"o periodo, incluindo a cauda flat -- pode parecer pequena sem ser)")

    rows = []
    for strategy_name, result in [
        ("kelly_risk_managed", kelly_result),
        ("binario_risk_managed", binary_result),
        ("buy_and_hold_risk_managed", buy_hold_result),
    ]:
        significance = mean_return_significance(result.net_returns, num_tests=3)
        active_pct = round(100.0 * result.active_trading_days / result.total_days, 1)
        rows.append(
            {
                "estrategia": strategy_name,
                "total_return": round(result.total_return, 4),
                "annualized_sharpe": round(result.annualized_sharpe, 3),
                "max_drawdown": round(result.max_drawdown, 4),
                "historical_var_95": round(result.historical_var_95, 4),
                "annualized_volatility": round(result.annualized_volatility, 4),
                "stop_loss_triggers": result.stop_loss_triggers,
                "circuit_breaker_triggered": result.circuit_breaker_triggered,
                "pct_dias_ativo": active_pct,
                "avg_position_size": round(float(result.actual_position.mean()), 4),
                "t_stat_newey_west": round(significance["t_stat"], 3),
                "significant_bonferroni": significance["significant"],
            }
        )
        if result.circuit_breaker_triggered:
            print(
                f"  AVISO [{strategy_name}]: circuit breaker ({result.circuit_breaker_reason}) "
                f"disparou em {result.circuit_breaker_date.date()}, ativo por apenas "
                f"{active_pct}% do periodo ({result.active_trading_days} de "
                f"{result.total_days} dias). As metricas acima (Sharpe, "
                f"retorno total, volatilidade) misturam o desempenho real "
                f"durante o periodo ativo com a 'cauda plana' de dias "
                f"desligado -- nao interprete como desempenho fraco e "
                f"consistente quando pode ser uma perda concentrada seguida "
                f"de inatividade."
            )

    summary = pd.DataFrame(rows).set_index("estrategia")

    plot_path = RESULTS_DIR / f"etapa4_equity_curve_{name}.png"
    plot_equity_curves(
        curves={
            "kelly_risk_managed": kelly_result.equity_curve,
            "binario_risk_managed": binary_result.equity_curve,
            "buy_and_hold_risk_managed": buy_hold_result.equity_curve,
        },
        circuit_breaker_dates={
            "kelly_risk_managed": kelly_result.circuit_breaker_date,
            "binario_risk_managed": binary_result.circuit_breaker_date,
            "buy_and_hold_risk_managed": buy_hold_result.circuit_breaker_date,
        },
        title=f"Curva de capital -- {name}",
        output_path=plot_path,
    )
    print(f"  Grafico de curva de capital salvo em: {plot_path}")

    kelly_significance = mean_return_significance(kelly_result.net_returns, num_tests=1)
    return summary, kelly_significance


def _run_synthetic_gate() -> bool:
    """Statistical gate: pooled across STAGE3_SYNTHETIC_GATE_SEEDS (same
    methodology as Stage 3's gate, for the same reason -- testing each seed
    individually has too little power and produces frequent spurious
    failures). Tests whether the Kelly-sized, risk-managed strategy shows
    a statistically significant return on an asset with zero true edge by
    construction.
    """

    print(f"\nGate estatistico (Etapa 4) no ativo sintetico, agregando "
          f"{len(STAGE3_SYNTHETIC_GATE_SEEDS)} seeds...")

    all_net_returns = []
    for seed in STAGE3_SYNTHETIC_GATE_SEEDS:
        prices = generate_random_walk_prices(periods=SYNTHETIC_PERIODS, seed=seed)
        predictions = _get_oos_predictions(prices["close"], respect_holdout=False)

        kelly_positions = kelly_position_series(
            prices["close"], predictions["prob_xgboost"],
            horizon_days=PREDICTION_HORIZON_DAYS, kelly_fraction_multiplier=KELLY_FRACTION,
            win_loss_window_days=KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS,
            min_win_observations=KELLY_MIN_WIN_OBSERVATIONS,
            min_loss_observations=KELLY_MIN_LOSS_OBSERVATIONS,
        )
        comparison_close = comparison_window(prices["close"], predictions)
        kelly_signal_full = pd.Series(0.0, index=comparison_close.index)
        kelly_signal_full.update(kelly_positions.reindex(comparison_close.index))

        result = run_risk_managed_backtest(
            comparison_close, kelly_signal_full,
            fee_bps=TRANSACTION_FEE_BPS, slippage_bps=SLIPPAGE_BPS, initial_capital=INITIAL_CAPITAL,
            stop_loss_pct=STOP_LOSS_PCT, circuit_breaker_drawdown_pct=CIRCUIT_BREAKER_DRAWDOWN_PCT,
            var_confidence_level=VAR_CONFIDENCE_LEVEL,
        )
        avg_position = float(result.actual_position.mean())
        print(f"  seed={seed}: posicao media={avg_position:.4f}, "
              f"retorno total={result.total_return:.4f}, "
              f"stop_loss_triggers={result.stop_loss_triggers}, "
              f"circuit_breaker={result.circuit_breaker_triggered}")
        all_net_returns.append(result.net_returns.iloc[1:])

    pooled_returns = pd.concat(all_net_returns, ignore_index=True)
    significance = mean_return_significance(pooled_returns, num_tests=1)
    print(f"\nGate agregado: t-stat={significance['t_stat']:.3f}, "
          f"p={significance['p_value']:.5f}, "
          f"significativo={significance['significant']}")

    if significance["significant"]:
        print(
            "\n*** GATE FALHOU: o sizing dinamico produziu retorno "
            "estatisticamente significativo no ativo sintetico, que nao "
            "tem edge real por construcao. Tratar como BUG ate prova em "
            "contrario -- NAO confiar nos resultados dos ativos reais "
            "abaixo. ***"
        )
        return False

    print("Gate passou: retorno da estrategia Kelly no sintetico e "
          "estatisticamente equivalente a zero.")
    return True


def main() -> None:
    gate_passed = _run_synthetic_gate()
    if not gate_passed:
        return

    real_data = load_all(STAGE3_TICKERS, BACKTEST_START_DATE, None)
    for ticker, frame in real_data.items():
        fee_bps, slippage_bps = _asset_costs(ticker)
        circuit_breaker_pct = _circuit_breaker_threshold(ticker)
        asset_circuit_breaker_pct = _asset_level_circuit_breaker_threshold(ticker)
        print(f"\n{'=' * 70}\n{ticker}\n{'=' * 70}")
        if ticker in ASSET_CIRCUIT_BREAKER_OVERRIDES:
            print(f"  Circuit breaker (carteira) customizado para {ticker}: {circuit_breaker_pct:.1%} "
                  f"(default global: {CIRCUIT_BREAKER_DRAWDOWN_PCT:.1%})")
        if ticker in ASSET_LEVEL_CIRCUIT_BREAKER_OVERRIDES:
            print(f"  Circuit breaker (ATIVO) customizado para {ticker}: "
                  f"{asset_circuit_breaker_pct:.1%} -- desligado para os demais ativos")
        summary, _ = _run_kelly_risk_managed(
            ticker, frame["close"], fee_bps, slippage_bps, circuit_breaker_pct, asset_circuit_breaker_pct,
        )
        print(summary)
        saved_path = save_summary(summary, stage_name=f"etapa4_{ticker}")
        print(f"Resultado salvo em: {saved_path}")


if __name__ == "__main__":
    main()
