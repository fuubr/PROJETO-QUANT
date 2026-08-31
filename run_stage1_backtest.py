"""Runs the Stage 1 backtest: mandatory baselines on real assets and on the
synthetic control asset, side by side.

This script does not implement any strategy. It exists to produce the
comparison table that closes out the stage: real-asset baselines next to the
synthetic-asset baselines, so any suspiciously good real-asset number can be
sanity-checked against what the same engine produces on an asset with no
real structure.

Statistical methodology (added after a post-Stage-2 review): significance is
assessed with a Newey-West (HAC) standard error, robust to autocorrelation
in strategy returns, and corrected for multiple comparisons across every row
in this table via Bonferroni.
"""

from __future__ import annotations

import pandas as pd

from backtest.baselines import buy_and_hold_signal, random_entry_persistent_signal, random_entry_signal
from backtest.engine import BacktestResult, run_backtest
from backtest.reporting import save_summary
from backtest.statistics import mean_return_significance
from config import (
    ASSET_COST_OVERRIDES,
    BACKTEST_END_DATE,
    BACKTEST_START_DATE,
    INITIAL_CAPITAL,
    RANDOM_ENTRY_AVG_HOLDING_DAYS,
    RANDOM_ENTRY_PROBABILITY,
    REAL_ASSET_TICKERS,
    SEED,
    SLIPPAGE_BPS,
    SYNTHETIC_PERIODS,
    TRANSACTION_FEE_BPS,
)
from data.real import load_all, resolve_end_date
from data.synthetic import generate_random_walk_prices


def _asset_costs(ticker: str) -> tuple[float, float]:
    """Resolve (fee_bps, slippage_bps) for a ticker, falling back to the
    global defaults when no override is configured. This lets the same
    engine run against different brokers/markets without code changes.
    """

    return ASSET_COST_OVERRIDES.get(ticker, (TRANSACTION_FEE_BPS, SLIPPAGE_BPS))


def _run_all_baselines(
    name: str, close: pd.Series, fee_bps: float, slippage_bps: float
) -> list[tuple[str, BacktestResult]]:
    results = []

    bh_result = run_backtest(
        close, buy_and_hold_signal(close.index),
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
    )
    results.append((f"{name} (buy_and_hold)", bh_result))

    iid_result = run_backtest(
        close, random_entry_signal(close.index, seed=SEED, probability=RANDOM_ENTRY_PROBABILITY),
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
    )
    results.append((f"{name} (random_entry_iid)", iid_result))

    persistent_result = run_backtest(
        close,
        random_entry_persistent_signal(
            close.index,
            seed=SEED,
            probability=RANDOM_ENTRY_PROBABILITY,
            avg_holding_days=RANDOM_ENTRY_AVG_HOLDING_DAYS,
        ),
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
    )
    results.append((f"{name} (random_entry_persistent)", persistent_result))

    return results


def _build_summary(named_results: list[tuple[str, BacktestResult]]) -> pd.DataFrame:
    num_tests = len(named_results)
    rows = []

    for name, result in named_results:
        significance = mean_return_significance(result.net_returns, num_tests=num_tests)
        rows.append(
            {
                "asset": name,
                "total_return": round(result.total_return, 4),
                "annualized_sharpe": round(result.annualized_sharpe, 3),
                "max_drawdown": round(result.max_drawdown, 4),
                "win_rate": round(result.win_rate, 4),
                "total_costs_paid": round(result.total_costs_paid, 2),
                "t_stat_newey_west": round(significance["t_stat"], 3),
                "p_value": round(significance["p_value"], 4),
                "significant_bonferroni": significance["significant"],
            }
        )

    return pd.DataFrame(rows).set_index("asset")


def main() -> None:
    named_results: list[tuple[str, BacktestResult]] = []

    # Synthetic control asset first, as the reference point for "no edge".
    synthetic_prices = generate_random_walk_prices(periods=SYNTHETIC_PERIODS)
    named_results.extend(
        _run_all_baselines(
            "synthetic", synthetic_prices["close"], TRANSACTION_FEE_BPS, SLIPPAGE_BPS
        )
    )

    # Real assets, each with its own resolved cost profile.
    resolved_end_date = resolve_end_date(BACKTEST_END_DATE)
    real_data = load_all(REAL_ASSET_TICKERS, BACKTEST_START_DATE, resolved_end_date)
    for ticker, frame in real_data.items():
        fee_bps, slippage_bps = _asset_costs(ticker)
        named_results.extend(_run_all_baselines(ticker, frame["close"], fee_bps, slippage_bps))

    summary = _build_summary(named_results)
    pd.set_option("display.width", 120)
    print(f"Janela: {BACKTEST_START_DATE} a {resolved_end_date} (fim resolvido dinamicamente)")
    print(f"Correcao: Newey-West (HAC) + Bonferroni para {len(named_results)} comparacoes simultaneas")
    print()
    print(summary)
    print()
    print(
        "Nota: 'significant_bonferroni' usa erro-padrao robusto a "
        "autocorrelacao (Newey-West) e corrige o limiar de significancia "
        "para multiplas comparacoes simultaneas (Bonferroni). Para o ativo "
        "sintetico, significancia no buy_and_hold indicaria possivel bug no "
        "motor. Para os baselines de entrada aleatoria, significancia "
        "costuma refletir custo de transacao real (turnover), nao edge "
        "espurio -- confirme olhando total_costs_paid antes de suspeitar do "
        "motor."
    )

    saved_path = save_summary(summary, stage_name="etapa1_backtest")
    print()
    print(f"Resultado salvo em: {saved_path}")


if __name__ == "__main__":
    main()
