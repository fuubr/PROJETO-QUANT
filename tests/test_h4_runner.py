import numpy as np
import pandas as pd

from run_h4_momentum import synthetic_like


def test_synthetic_has_no_volatility_premium():
    """Controle sintetico nao pode recompensar baixa vol: retorno medio ~0 para todos."""

    idx = pd.bdate_range("2000-01-01", periods=20_000)
    template = pd.DataFrame({"calma": 100.0, "agitada": 100.0}, index=idx)
    template["agitada"] *= np.exp(np.random.default_rng(1).normal(0, 0.06, len(idx)).cumsum())
    template["calma"] *= np.exp(np.random.default_rng(2).normal(0, 0.01, len(idx)).cumsum())
    mean = synthetic_like(template, seed=0).pct_change().mean()
    assert abs(mean["agitada"]) < 3 * 0.06 / np.sqrt(len(idx))
    assert abs(mean["calma"]) < 3 * 0.01 / np.sqrt(len(idx))


def test_synthetic_keeps_listing_window():
    idx = pd.bdate_range("2020-01-01", periods=10)
    template = pd.DataFrame({"A": 1.0, "B": 1.0}, index=idx)
    template.iloc[:4, 1] = np.nan
    assert synthetic_like(template, seed=0)["B"].isna().sum() == 4
