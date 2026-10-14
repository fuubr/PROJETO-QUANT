"""H5 -- Teste 1 (desenvolvimento, 2012-2023) + controle sintetico.

Executa o pre-registro (com a emenda) de docs/CRITERIOS.md. Precos cortados em
H4_DEV_END: o holdout 2024+ nao e tocado. Saida: docs/h5_resultado.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.tseries.offsets import BDay

from config import H4_COST_BPS, H4_DEV_END, H4_DEV_START, H4_UNIVERSE
from data.cvm import load_dfp_year
from run_h4_momentum import N_SYNTHETIC, evaluate, monthly_rf, synthetic_like
from strategies.fundamental import earnings_yield, h5_scores, point_in_time

OUT = Path(__file__).resolve().parent / "docs" / "h5_resultado.md"
CACHE = Path(__file__).resolve().parent / "data" / "cache" / f"h5_precos_2010_{H4_DEV_END}.parquet"
SPLITS_CACHE = CACHE.with_name("h5_desdobramentos_ate_hoje.parquet")
NUM_TESTS = 5  # Bonferroni: H1..H5
UNITS = {"SANB11.SA", "TAEE11.SA", "KLBN11.SA", "ENGI11.SA"}
H5_UNIVERSE = [t for t in H4_UNIVERSE if t not in UNITS and t != "PETR3.SA"]

# CNPJ e estavel quando o codigo CVM muda; a TIM trocou de entidade em 2018.
CNPJ = {
    "ABEV3": ["07.526.557/0001-00"], "B3SA3": ["09.346.601/0001-25"], "BBAS3": ["00.000.000/0001-91"],
    "BBDC4": ["60.746.948/0001-12"], "BBSE3": ["17.344.597/0001-94"], "BRAP4": ["03.847.461/0001-92"],
    "BRKM5": ["42.150.391/0001-70"], "CMIG4": ["17.155.730/0001-64"], "CMIN3": ["08.902.291/0001-15"],
    "CPFE3": ["02.429.144/0001-93"], "CSAN3": ["50.746.577/0001-15"], "CSNA3": ["33.042.730/0001-04"],
    "CYRE3": ["73.178.600/0001-18"], "EGIE3": ["02.474.103/0001-19"], "ENEV3": ["04.423.567/0001-21"],
    "EQTL3": ["03.220.438/0001-73"], "GGBR4": ["33.611.500/0001-19"], "GOAU4": ["92.690.783/0001-09"],
    "HYPE3": ["02.932.074/0001-91"], "ITSA4": ["61.532.644/0001-15"], "ITUB4": ["60.872.504/0001-23"],
    "LREN3": ["92.754.738/0001-62"], "MRVE3": ["08.343.492/0001-20"], "MULT3": ["07.816.890/0001-53"],
    "PETR4": ["33.000.167/0001-01"], "PRIO3": ["10.629.105/0001-68"], "RADL3": ["61.585.865/0001-51"],
    "RENT3": ["16.670.085/0001-55"], "SBSP3": ["43.776.517/0001-80"], "SUZB3": ["16.404.287/0001-55"],
    "TIMS3": ["02.558.115/0001-21", "02.421.421/0001-11"], "TOTS3": ["53.113.791/0001-22"],
    "UGPA3": ["33.256.439/0001-39"], "USIM5": ["60.894.730/0001-05"], "VALE3": ["33.592.510/0001-54"],
    "VIVT3": ["02.558.157/0001-62"], "WEGE3": ["84.429.695/0001-11"], "YDUQ3": ["08.807.432/0001-10"],
}
CLASS = {"3": "ON", "4": "PN", "5": "PNA", "6": "PNB"}


def eps_for(ticker: str, eps: dict) -> float | None:
    cls = CLASS[ticker.removesuffix(".SA")[-1]]
    if cls in eps:
        return eps[cls]
    if cls != "ON":  # PN sem letra / PNA / PNB variam entre anos
        return next((v for k, v in eps.items() if k.startswith("PN")), None)
    return None


def build_fundamentals(years: range) -> pd.DataFrame:
    rows = []
    for y in years:
        d = load_dfp_year(y).set_index("cnpj")
        for t in H5_UNIVERSE:
            for cnpj in CNPJ[t.removesuffix(".SA")]:
                if cnpj in d.index:
                    r = d.loc[cnpj]
                    rows.append([t, r.received + BDay(1), r.fy_end, eps_for(t, r.eps if isinstance(r.eps, dict) else {}), r.net_income, r.equity])
    return pd.DataFrame(rows, columns=["ticker", "available", "fy_end", "eps", "net_income", "equity"])


def load_prices() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """(fechamento ajustado total -> retornos, fechamento ajustado so por
    desdobramentos -> E/P, eventos de desdobramento por ticker)."""

    if not CACHE.exists():
        import yfinance as yf
        end = (pd.Timestamp(H4_DEV_END) + pd.Timedelta(days=1)).date().isoformat()
        raw = yf.download(H5_UNIVERSE, start="2010-01-01", end=end, auto_adjust=False, actions=True, progress=False)
        raw[["Adj Close", "Close", "Stock Splits"]].to_parquet(CACHE)
    raw = pd.read_parquet(CACHE)
    # O yfinance ajusta TODO o historico pelos desdobramentos ate HOJE (inclusive os
    # posteriores ao corte): o fator do LPA precisa dos mesmos eventos, senao o E/P
    # sai errado (BBAS3, desdobramento 2:1 em 2024, aparecia com E/P dobrado). So
    # datas de eventos societarios; nenhum preco depois do corte e usado.
    if not SPLITS_CACHE.exists():
        import yfinance as yf
        ev = yf.download(H5_UNIVERSE, start="2010-01-01", auto_adjust=False, actions=True, progress=False)
        ev[["Stock Splits"]].to_parquet(SPLITS_CACHE)
    ev = pd.read_parquet(SPLITS_CACHE)["Stock Splits"]
    splits = {t: s[s > 0] for t, s in ev.items()}
    return raw["Adj Close"], raw["Close"], splits


def make_selectors(fund, split_close, splits):
    def scores(hist: pd.DataFrame) -> pd.Series:
        t = hist.index[-1]
        pit = point_in_time(fund, t)
        pit = pit[pit.index.isin(hist.columns[hist.iloc[-1].notna()])].copy()
        if pit.empty:
            return pd.Series(dtype=float)
        pit["ep"] = [earnings_yield(r.eps, r.fy_end, split_close.at[t, tk], splits.get(tk, pd.Series(dtype=float)))
                     if pd.notna(r.eps) else float("nan") for tk, r in pit.iterrows()]
        return h5_scores(pit)

    return (lambda h: list(scores(h).nlargest(10).index)), (lambda h: list(scores(h).index))


def main() -> None:
    adj, split_close, splits = load_prices()
    adj, split_close = adj.loc[:H4_DEV_END], split_close.loc[:H4_DEV_END]
    fund = build_fundamentals(range(2010, 2024))
    rf_monthly, rf_source = monthly_rf(adj)

    sel, bench = make_selectors(fund, split_close, splits)
    real = evaluate(adj, rf_monthly, sel, bench, NUM_TESTS)
    synth_pass = 0
    for s in range(N_SYNTHETIC):
        fake = synthetic_like(adj, s)
        fsel, fbench = make_selectors(fund, fake, {})
        synth_pass += evaluate(fake, None, fsel, fbench, NUM_TESTS)["passou"]

    md = ["# H5 -- Teste 1 (desenvolvimento)", "",
          (f"Pre-registro + emenda: `docs/CRITERIOS.md`. Periodo {H4_DEV_START} a {H4_DEV_END}, {real['meses']} meses, "
          f"universo de {len(H5_UNIVERSE)} acoes, balancos DFP da CVM point-in-time, custo {H4_COST_BPS:.0f} bps, "
          f"caixa: {rf_source}."), "",
          "| | H5 (top 10) | Benchmark (elegiveis, peso igual) |", "|---|---|---|",
          f"| Retorno a.a. | {real['h4_cagr']:+.1%} | {real['ew_cagr']:+.1%} |",
          f"| Sharpe de excesso (CDI) | {real['sharpe_h4']:.2f} | {real['sharpe_ew']:.2f} |",
          f"| Drawdown maximo | {real['dd_h4']:.0%} | {real['dd_ew']:.0%} |", "",
          (f"Diferenca H5 - benchmark: **{real['diff_aa']:+.1%} a.a.**, Newey-West t={real['t']:.2f}, "
          f"p={real['p']:.4f} (limite {0.05 / NUM_TESTS:.4f})."), "",
          f"**Teste 1: {'PASSOU' if real['passou'] else 'NAO PASSOU'}.**", "",
          (f"Controle sintetico (precos aleatorios, balancos reais): {synth_pass} de {N_SYNTHETIC} passaram "
          f"(esperado por acaso ~{N_SYNTHETIC * 0.05 / NUM_TESTS / 2:.1f})."), ""]
    OUT.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
