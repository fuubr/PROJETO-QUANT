"""H4 -- Teste 1 (desenvolvimento, 2012-2023) + controle sintetico.

Executa EXATAMENTE o pre-registro de docs/CRITERIOS.md. Nao toca 2024+ (holdout da H4):
os precos sao cortados em H4_DEV_END antes de qualquer calculo.
Saida: docs/h4_resultado.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.statistics import mean_return_significance
from config import H4_COST_BPS, H4_DEV_END, H4_DEV_START, H4_UNIVERSE
from data.rates import risk_free_daily
from strategies.cross_section import (
    equal_weight_select,
    h4_select,
    run_monthly_portfolio,
)

OUT = Path(__file__).resolve().parent / "docs" / "h4_resultado.md"
CACHE = Path(__file__).resolve().parent / "data" / "cache" / f"h4_universe_2010_{H4_DEV_END}.parquet"
NUM_TESTS = 4  # Bonferroni: H1, H2, H3, H4
N_SYNTHETIC = 50


def load_closes() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    import yfinance as yf
    end = (pd.Timestamp(H4_DEV_END) + pd.Timedelta(days=1)).date().isoformat()
    closes = yf.download(H4_UNIVERSE, start="2010-01-01", end=end, auto_adjust=True, progress=False)["Close"]
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    closes.to_parquet(CACHE)
    return closes


def evaluate(closes: pd.DataFrame, rf_monthly: pd.Series | None) -> dict:
    h4 = run_monthly_portfolio(closes, h4_select, H4_COST_BPS)
    ew = run_monthly_portfolio(closes, equal_weight_select, H4_COST_BPS)
    h4, ew = h4.loc[H4_DEV_START:], ew.loc[H4_DEV_START:]
    diff = h4 - ew
    sig = mean_return_significance(diff, num_tests=NUM_TESTS)
    rf = rf_monthly.reindex(h4.index).fillna(0.0) if rf_monthly is not None else 0.0

    def sharpe(r):
        ex = r - rf
        return float(ex.mean() / ex.std(ddof=1) * np.sqrt(12))

    def cagr(r):
        return float((1 + r).prod() ** (12 / len(r)) - 1)

    passed = diff.mean() > 0 and sig["significant"] and sharpe(h4) > sharpe(ew)
    return {"meses": len(diff), "h4_cagr": cagr(h4), "ew_cagr": cagr(ew), "diff_aa": float(diff.mean() * 12),
                "t": sig["t_stat"], "p": sig["p_value"], "sharpe_h4": sharpe(h4), "sharpe_ew": sharpe(ew),
                "dd_h4": float(((1 + h4).cumprod() / (1 + h4).cumprod().cummax() - 1).min()),
                "dd_ew": float(((1 + ew).cumprod() / (1 + ew).cumprod().cummax() - 1).min()), "passou": bool(passed)}


def synthetic_like(closes: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Random walks com a MESMA vol e a mesma janela de existencia de cada acao,
    drift zero: nenhuma informacao cross-section por construcao."""

    rng = np.random.default_rng(seed)
    vols = closes.pct_change().std()
    noise = pd.DataFrame(rng.standard_normal(closes.shape) * vols.values, index=closes.index, columns=closes.columns)
    # Retorno SIMPLES com media zero. (Subtrair 0,5*vol^2 aqui criava um premio
    # de baixa vol falso -- exatamente o que a H4 compra. Pego pelo controle.)
    fake = 100 * (1 + noise).cumprod()
    return fake.where(closes.notna())


def main() -> None:
    closes = load_closes().loc[:H4_DEV_END]
    rf_daily, rf_source = risk_free_daily("PETR4.SA", closes.index)
    rf_monthly = (1 + rf_daily).groupby(rf_daily.index.to_period("M")).prod() - 1
    rf_monthly.index = rf_monthly.index.to_timestamp("M")
    rf_monthly = rf_monthly.reindex(pd.to_datetime(rf_monthly.index)).rename(None)
    # alinhar ao ultimo pregao do mes
    month_last = pd.Series(closes.index, index=closes.index).groupby(closes.index.to_period("M")).max()
    rf_monthly.index = month_last.reindex(rf_monthly.index.to_period("M")).values

    real = evaluate(closes, rf_monthly)
    synth = [evaluate(synthetic_like(closes, s), None) for s in range(N_SYNTHETIC)]
    synth_pass = sum(s["passou"] for s in synth)

    md = ["# H4 -- Teste 1 (desenvolvimento)", "",
          (f"Pre-registro: `docs/CRITERIOS.md`. Periodo {H4_DEV_START} a {H4_DEV_END}, {real['meses']} meses, "
          f"universo de {closes.shape[1]} acoes, custo {H4_COST_BPS:.0f} bps, caixa: {rf_source}."), "",
          "| | H4 (top 10) | Benchmark (todas, peso igual) |", "|---|---|---|",
          f"| Retorno a.a. | {real['h4_cagr']:+.1%} | {real['ew_cagr']:+.1%} |",
          f"| Sharpe de excesso (CDI) | {real['sharpe_h4']:.2f} | {real['sharpe_ew']:.2f} |",
          f"| Drawdown maximo | {real['dd_h4']:.0%} | {real['dd_ew']:.0%} |", "",
          (f"Diferenca H4 - benchmark: **{real['diff_aa']:+.1%} a.a.**, Newey-West t={real['t']:.2f}, "
          f"p={real['p']:.4f} (limite {0.05 / NUM_TESTS:.4f})."), "",
          f"**Teste 1: {'PASSOU' if real['passou'] else 'NAO PASSOU'}.**", "",
          (f"Controle sintetico: {synth_pass} de {N_SYNTHETIC} random walks passaram "
          f"(esperado por acaso ~{N_SYNTHETIC * 0.05 / NUM_TESTS / 2:.1f})."), ""]
    OUT.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
