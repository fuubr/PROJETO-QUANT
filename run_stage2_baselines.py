"""Runs the Stage 2 baseline comparison: the 3 mandatory baselines required
before any AI model is introduced (aleatorio, buy-and-hold, media movel
simples), on real assets and on the synthetic control asset.

Per the project's own restrictions, this script does not implement anything
beyond what Stage 2 explicitly requires. random_entry_persistent_signal is
used as the primary "aleatorio" baseline (more realistic turnover than pure
i.i.d.); random_entry_signal (i.i.d.) is kept alongside as an extra stress
test carried over from Stage 1, not a Stage-2 requirement on its own.

Statistical methodology (added after a post-Stage-2 review): significance is
assessed with a Newey-West (HAC) standard error, robust to autocorrelation
in strategy returns, and corrected for multiple comparisons across every
(asset, baseline) row in this table via Bonferroni -- testing 20 things at
once at an uncorrected 5% level would give a large chance of at least one
false positive by chance alone.

Going forward, any AI model (Stage 3+) only has a claim to being useful if
it beats all three of these baselines consistently -- not just once, and
not just on the assets where it happens to look best.
"""

from __future__ import annotations

import pandas as pd

from backtest.baselines import (
    buy_and_hold_signal,
    moving_average_crossover_signal,
    random_entry_persistent_signal,
    random_entry_signal,
)
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
    SMA_FAST_WINDOW,
    SMA_SLOW_WINDOW,
    SYNTHETIC_PERIODS,
    TRANSACTION_FEE_BPS,
)
from data.real import load_all, resolve_end_date
from data.synthetic import generate_random_walk_prices


def _asset_costs(ticker: str) -> tuple[float, float]:
    return ASSET_COST_OVERRIDES.get(ticker, (TRANSACTION_FEE_BPS, SLIPPAGE_BPS))


def _run_mandatory_baselines(
    name: str, close: pd.Series, fee_bps: float, slippage_bps: float
) -> list[tuple[str, BacktestResult]]:
    results = []

    bh_result = run_backtest(
        close, buy_and_hold_signal(close.index),
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
    )
    results.append((f"{name} (buy_and_hold)", bh_result))

    random_result = run_backtest(
        close,
        random_entry_persistent_signal(
            close.index, seed=SEED, probability=RANDOM_ENTRY_PROBABILITY,
            avg_holding_days=RANDOM_ENTRY_AVG_HOLDING_DAYS,
        ),
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
    )
    results.append((f"{name} (aleatorio)", random_result))

    sma_result = run_backtest(
        close,
        moving_average_crossover_signal(close, fast_window=SMA_FAST_WINDOW, slow_window=SMA_SLOW_WINDOW),
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
    )
    results.append((f"{name} (media_movel)", sma_result))

    # Extra stress test carried over from Stage 1, not a Stage 2 requirement.
    iid_result = run_backtest(
        close, random_entry_signal(close.index, seed=SEED, probability=RANDOM_ENTRY_PROBABILITY),
        fee_bps=fee_bps, slippage_bps=slippage_bps, initial_capital=INITIAL_CAPITAL,
    )
    results.append((f"{name} (aleatorio_iid_extra)", iid_result))

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

    synthetic_prices = generate_random_walk_prices(periods=SYNTHETIC_PERIODS)
    named_results.extend(
        _run_mandatory_baselines(
            "synthetic", synthetic_prices["close"], TRANSACTION_FEE_BPS, SLIPPAGE_BPS
        )
    )

    resolved_end_date = resolve_end_date(BACKTEST_END_DATE)
    real_data = load_all(REAL_ASSET_TICKERS, BACKTEST_START_DATE, resolved_end_date)
    for ticker, frame in real_data.items():
        fee_bps, slippage_bps = _asset_costs(ticker)
        named_results.extend(_run_mandatory_baselines(ticker, frame["close"], fee_bps, slippage_bps))

    summary = _build_summary(named_results)
    pd.set_option("display.width", 120)
    print(f"Janela: {BACKTEST_START_DATE} a {resolved_end_date} (fim resolvido dinamicamente)")
    print(f"SMA crossover: fast={SMA_FAST_WINDOW}, slow={SMA_SLOW_WINDOW}")
    print(f"Correcao: Newey-West (HAC) + Bonferroni para {len(named_results)} comparacoes simultaneas")
    print()
    print(summary)
    print()
    print(
        "Gate da Etapa 2: no ativo sintetico, NENHUM dos 3 baselines "
        "obrigatorios deveria ser significativo com custo zero (ja validado "
        "nos testes automatizados, que usam Newey-West sem correcao de "
        "Bonferroni por serem gates individuais pre-registrados, nao uma "
        "tabela exploratoria). Nesta tabela, com custo E correcao de "
        "Bonferroni para todas as comparacoes simultaneas, a barra para "
        "'significant_bonferroni=True' e bem mais alta que os 1.96 usados "
        "antes -- menos linhas devem aparecer como significativas do que na "
        "primeira versao deste script."
    )

    saved_path = save_summary(summary, stage_name="etapa2_baselines")
    print()
    print(f"Resultado salvo em: {saved_path}")


if __name__ == "__main__":
    main()
