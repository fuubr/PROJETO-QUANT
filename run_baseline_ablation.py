"""Fair baselines + ablation for the Kelly/risk strategy.

Answers two questions the earlier comparisons could not:

1. Is the Kelly result better than FAIR alternatives? The earlier
   "buy-and-hold" column ran with the circuit breaker on, so it sat out the
   recovery after a crash while Kelly (small exposure) stayed in -- an
   apples-to-oranges comparison. Here:
     - buy_and_hold_puro: no overlays at all
     - exposicao_igualada: constant exposure equal to Kelly's average, no overlays
       (same dilution, zero information)
     - vol_target: exposure = target vol / realized vol, no overlays
2. ABLATION: does the MODEL add anything? Same Kelly sizing and overlays,
   but the model's probability replaced by the causal base rate of matured
   labels (what a no-information forecaster would say). If this performs
   like the real model, the AI contributes nothing and the result comes
   from sizing and stops. Tested directly: Newey-West on the daily return
   difference (real model minus ablation).

Out-of-sample only (dev period; holdout untouched). Results are printed
and written to docs/analise_baselines.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.baselines import buy_and_hold_signal
from backtest.statistics import mean_return_significance
from config import (
    BACKTEST_START_DATE, CASH_ANNUAL_RATE_ASSUMED, INITIAL_CAPITAL, KELLY_FRACTION, KELLY_MIN_LOSS_OBSERVATIONS,
    KELLY_MIN_WIN_OBSERVATIONS, KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS, PREDICTION_HORIZON_DAYS, STAGE3_TICKERS,
    STOP_LOSS_PCT, VAR_CONFIDENCE_LEVEL, VOL_TARGET_ANNUAL,
)
from data.real import load_all
from models.comparison import comparison_window
from models.labels import build_forward_return_label
from risk.kelly import kelly_position_series
from risk.managed_backtest import run_risk_managed_backtest
from risk.metrics import TRADING_DAYS_PER_YEAR
from run_stage4_risk_management import (
    _asset_costs, _asset_level_circuit_breaker_threshold, _circuit_breaker_threshold, _get_oos_predictions,
)

OUT_PATH = Path(__file__).resolve().parent / "docs" / "analise_baselines.md"
OFF = 0.99  # effectively disables stop-loss / circuit breaker


def causal_base_rate(close: pd.Series, index: pd.Index) -> pd.Series:
    """P(up) a no-information forecaster would state: the mean of labels
    whose outcome is already known (a label at s is known at s + horizon).
    """

    labels = build_forward_return_label(close, PREDICTION_HORIZON_DAYS).dropna()
    known = labels.shift(PREDICTION_HORIZON_DAYS).expanding().mean()
    return known.reindex(index).fillna(0.5)


def _kelly_signal(close, probs, window):
    positions = kelly_position_series(
        close, probs, horizon_days=PREDICTION_HORIZON_DAYS, kelly_fraction_multiplier=KELLY_FRACTION,
        win_loss_window_days=KELLY_WIN_LOSS_ESTIMATION_WINDOW_DAYS, min_win_observations=KELLY_MIN_WIN_OBSERVATIONS,
        min_loss_observations=KELLY_MIN_LOSS_OBSERVATIONS,
    )
    signal = pd.Series(0.0, index=window.index)
    signal.update(positions.reindex(window.index))
    return signal


def _metrics(result, ticker: str) -> dict:
    r = result.net_returns.iloc[1:]
    held = result.actual_position.iloc[1:]
    cash_daily = (1 + CASH_ANNUAL_RATE_ASSUMED["BR" if ticker.endswith(".SA") else "US"]) ** (1 / TRADING_DAYS_PER_YEAR) - 1
    rc = r + (1 - held) * cash_daily
    sharpe = lambda x: float(x.mean() / x.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR)) if x.std(ddof=1) > 0 else 0.0
    return {
        "retorno": round(result.total_return, 4), "sharpe": round(result.annualized_sharpe, 3),
        "sortino": round(result.sortino_ratio, 3), "max_dd": round(result.max_drawdown, 4),
        "exposicao_media": round(float(held.mean()), 3), "dias_ativo_%": round(100 * result.active_trading_days / result.total_days, 1),
        "retorno_c/_juros_caixa": round(float((1 + rc).prod() - 1), 4), "sharpe_c/_juros_caixa": round(sharpe(rc), 3),
    }


def analyze(name, close, fee, slip, cb_pct, acb_pct, respect_holdout):
    predictions = _get_oos_predictions(close, respect_holdout=respect_holdout)
    window = comparison_window(close, predictions)
    common = dict(fee_bps=fee, slippage_bps=slip, initial_capital=INITIAL_CAPITAL, var_confidence_level=VAR_CONFIDENCE_LEVEL)
    on = dict(stop_loss_pct=STOP_LOSS_PCT, circuit_breaker_drawdown_pct=cb_pct, asset_circuit_breaker_drawdown_pct=acb_pct)
    off = dict(stop_loss_pct=OFF, circuit_breaker_drawdown_pct=OFF)

    kelly = run_risk_managed_backtest(window, _kelly_signal(close, predictions["prob_xgboost"], window), **common, **on)
    ablation = run_risk_managed_backtest(
        window, _kelly_signal(close, causal_base_rate(close, predictions.index), window), **common, **on)
    bh_pure = run_risk_managed_backtest(window, buy_and_hold_signal(window.index), **common, **off)
    bh_overlay = run_risk_managed_backtest(window, buy_and_hold_signal(window.index), **common, **on)
    f = float(kelly.actual_position.iloc[1:].mean())
    matched = run_risk_managed_backtest(window, pd.Series(f, index=window.index), **common, **off)
    vol = close.pct_change().rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    vt_signal = (VOL_TARGET_ANNUAL / vol).clip(0, 1).reindex(window.index).fillna(0.0)
    vol_target = run_risk_managed_backtest(window, vt_signal, **common, **off)

    runs = {
        "kelly_modelo (c/ stops)": kelly, "ablacao_sem_modelo (c/ stops)": ablation,
        "exposicao_igualada (s/ stops)": matched, "vol_target (s/ stops)": vol_target,
        "buy_and_hold_puro (s/ stops)": bh_pure, "buy_and_hold (c/ breaker)": bh_overlay,
    }
    table = pd.DataFrame({k: _metrics(v, name) for k, v in runs.items()}).T

    tests = {}
    for label, other in (("modelo - ablacao", ablation), ("modelo - exposicao_igualada", matched)):
        diff = kelly.net_returns.iloc[1:] - other.net_returns.iloc[1:]
        sig = mean_return_significance(diff, num_tests=2)
        tests[label] = (float(diff.mean() * TRADING_DAYS_PER_YEAR), sig["t_stat"], sig["p_value"], sig["significant"])
    return table, tests, (window.index[0].date(), window.index[-1].date(), len(window))


def main() -> None:
    real = load_all(STAGE3_TICKERS, BACKTEST_START_DATE, None)
    md = ["# Analise de baselines justos e ablacao", "",
          "Fora da amostra (walk-forward purgado), periodo de desenvolvimento; holdout intocado. "
          "Gerado por `run_baseline_ablation.py`. Caixa a 0% nas colunas principais; as colunas "
          "`c/ juros caixa` usam taxas ASSUMIDAS (config), nao dados de mercado.", ""]
    for ticker, frame in real.items():
        fee, slip = _asset_costs(ticker)
        table, tests, (start, end, n) = analyze(
            ticker, frame["close"], fee, slip, _circuit_breaker_threshold(ticker),
            _asset_level_circuit_breaker_threshold(ticker), respect_holdout=True)
        print(f"\n=== {ticker} ({start} a {end}, {n} dias) ===\n{table.to_string()}")
        md += [f"## {ticker} ({start} a {end}, {n} dias)", "", table.to_markdown(), "",
               "Teste de diferenca (Newey-West, Bonferroni x2): retorno anualizado da diferenca, t, p", ""]
        for label, (ann, t, p, sig) in tests.items():
            line = f"- {label}: {ann:+.2%} a.a., t={t:.2f}, p={p:.3f}, significativo={sig}"
            print(line); md.append(line)
        md.append("")
    OUT_PATH.parent.mkdir(exist_ok=True)
    OUT_PATH.write_text("\n".join(md), encoding="utf-8")
    print(f"\nSalvo em {OUT_PATH}")


if __name__ == "__main__":
    main()
