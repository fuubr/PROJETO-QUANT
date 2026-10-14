"""H4: ranking cross-section (momentum 12-1 + baixa volatilidade), mensal.

Parametros congelados em config.py / docs/CRITERIOS.md. Toda funcao recebe so o
historico ate a data de decisao -- quem garante isso e run_monthly_portfolio.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable

import pandas as pd
from pandas.tseries.offsets import BMonthEnd

from config import H4_MOMENTUM_LOOKBACK, H4_MOMENTUM_SKIP, H4_TOP_N, H4_VOL_WINDOW

MIN_HISTORY = H4_MOMENTUM_LOOKBACK + H4_MOMENTUM_SKIP  # 273 pregoes


def _eligible(closes: pd.DataFrame) -> pd.DataFrame:
    ok = (closes.notna().sum() >= MIN_HISTORY) & closes.iloc[-1].notna()
    return closes.loc[:, ok]


def momentum_12_1(closes: pd.DataFrame) -> pd.Series:
    """Retorno de t-252 a t-21: ignora o ultimo mes (reversao de curto prazo)."""

    c = _eligible(closes)
    if c.shape[1] == 0:
        return pd.Series(dtype=float)
    return c.iloc[-1 - H4_MOMENTUM_SKIP] / c.iloc[-1 - H4_MOMENTUM_LOOKBACK] - 1


def h4_scores(closes: pd.DataFrame) -> pd.Series:
    c = _eligible(closes)
    if c.shape[1] == 0:
        return pd.Series(dtype=float)
    mom = momentum_12_1(c)
    vol = c.iloc[-H4_VOL_WINDOW - 1:].pct_change().iloc[1:].std(ddof=1)
    return mom.rank(pct=True) + (-vol).rank(pct=True)


def h4_select(closes: pd.DataFrame) -> list[str]:
    return list(h4_scores(closes).nlargest(H4_TOP_N).index)


def equal_weight_select(closes: pd.DataFrame) -> list[str]:
    """Benchmark justo: todas as elegiveis do mesmo universo."""

    return list(_eligible(closes).columns)


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Ultimo pregao de cada mes; o mes final so entra se ja terminou."""

    last = pd.Series(index, index=index).groupby(index.to_period("M")).max()
    if len(last) and last.iloc[-1] != last.iloc[-1] + BMonthEnd(0):
        last = last.iloc[:-1]
    return pd.DatetimeIndex(last.values)


def run_monthly_portfolio(
    closes: pd.DataFrame, select: Callable[[pd.DataFrame], list[str]], cost_bps: float,
) -> pd.Series:
    """Retorno liquido mensal: pesos iguais nos escolhidos no fechamento de t,
    mantidos ate o fechamento do proximo fim de mes. Custo = cost_bps sobre o
    valor negociado (diferenca entre alvo e pesos ja derivados pelo mercado).
    """

    dates = month_end_dates(closes.index)
    cost = cost_bps / 10_000
    held = pd.Series(dtype=float)
    out = {}
    for t, t_next in itertools.pairwise(dates):
        picks = select(closes.loc[:t])
        target = pd.Series(1.0 / len(picks), index=picks) if picks else pd.Series(dtype=float)
        traded = target.sub(held, fill_value=0).abs().sum()
        growth = (closes.loc[t_next, target.index] / closes.loc[t, target.index]).fillna(1.0)
        gross = float((target * growth).sum()) if len(target) else 1.0
        out[t_next] = gross * (1 - cost * traded) - 1
        held = target * growth / gross if len(target) else target
    return pd.Series(out, dtype=float)
