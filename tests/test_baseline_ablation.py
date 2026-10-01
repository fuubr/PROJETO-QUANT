import numpy as np
import pandas as pd
import pytest

from config import PREDICTION_HORIZON_DAYS
from run_baseline_ablation import analyze, causal_base_rate


def _close(n, seed):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))), index=pd.bdate_range("2015-01-01", periods=n))


def test_causal_base_rate_ignores_outcomes_not_yet_known():
    """The ablation's 'no information' forecaster must not peek: changing
    prices AFTER day t must not change its estimate at t (a label at s only
    becomes known at s + horizon).
    """

    close = _close(600, 1)
    t = close.index[400]
    base = causal_base_rate(close, close.index)
    tampered = close.copy()
    tampered.iloc[401:] *= np.linspace(1, 3, len(tampered.iloc[401:]))  # rewrite the entire future
    assert causal_base_rate(tampered, close.index)[t] == pytest.approx(base[t])


def test_causal_base_rate_only_uses_labels_matured_by_t():
    close = _close(600, 2)
    base = causal_base_rate(close, close.index)
    assert base.iloc[: PREDICTION_HORIZON_DAYS].eq(0.5).all()  # nothing matured yet -> no information
    assert ((base >= 0) & (base <= 1)).all()


def test_analyze_returns_all_strategies_and_tests():
    table, tests, (start, end, n) = analyze("FAKE", _close(1500, 3), 5.0, 2.0, 0.20, None, respect_holdout=False)
    assert len(table) == 6 and n > 100
    assert {"retorno", "sharpe", "max_dd", "exposicao_media", "retorno_c/_juros_caixa"} <= set(table.columns)
    assert set(tests) == {"modelo - ablacao", "modelo - exposicao_igualada"}
    # the matched-exposure baseline is built to have the same average exposure as the model strategy
    assert table.loc["exposicao_igualada (s/ stops)", "exposicao_media"] == pytest.approx(
        table.loc["kelly_modelo (c/ stops)", "exposicao_media"], abs=0.005)
