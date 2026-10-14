"""H5: valor (E/P) + qualidade (ROE), point-in-time. Regras em docs/CRITERIOS.md."""

from __future__ import annotations

import pandas as pd

MAX_ABS_EP = 1.0  # acima disso e erro de dado (emenda pre-registrada)


def point_in_time(fund: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    """Ultimo balanco de cada ticker ja publicado (available <= t)."""

    known = fund[fund.available <= t]
    if known.empty:
        return known.set_index("ticker")
    return known.sort_values("available").groupby("ticker").last()


def earnings_yield(eps: float, fy_end: pd.Timestamp, price: float, splits: pd.Series) -> float:
    """LPA do exercicio / preco ajustado so por desdobramentos (base de hoje).
    Eventos depois do fim do exercicio levam o LPA para a mesma base do preco."""

    factor = float(splits[splits.index > fy_end].prod()) if len(splits) else 1.0
    return eps / factor / price


def h5_scores(pit: pd.DataFrame) -> pd.Series:
    """pit: index ticker, colunas ep, net_income, equity, eps."""

    if pit.empty:
        return pd.Series(dtype=float)
    ok = ((pit.equity > 0) & pit.ep.notna() & (pit.ep.abs() <= MAX_ABS_EP)
          & ~((pit.eps == 0) & (pit.net_income != 0)))
    p = pit[ok]
    roe = p.net_income / p.equity
    return p.ep.rank(pct=True) + roe.rank(pct=True)
