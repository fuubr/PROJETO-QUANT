import numpy as np
import pandas as pd
import pytest

from strategies.cross_section import h4_scores, month_end_dates, run_monthly_portfolio


def _prices(n_days: int, daily: dict[str, np.ndarray]) -> pd.DataFrame:
    idx = pd.bdate_range("2015-01-01", periods=n_days)
    return pd.DataFrame({k: 100 * np.cumprod(1 + v) for k, v in daily.items()}, index=idx)


def test_scores_prefer_high_momentum_and_low_vol():
    rng = np.random.default_rng(0)
    n = 300
    closes = _prices(n, {
        "BOM": 0.002 + 0.005 * rng.standard_normal(n),   # sobe, pouca vol
        "RUIM": -0.002 + 0.03 * rng.standard_normal(n),  # cai, muita vol
        "MEIO": 0.0 + 0.015 * rng.standard_normal(n),
    })
    s = h4_scores(closes)
    assert s.idxmax() == "BOM" and s.idxmin() == "RUIM"


def test_short_history_is_not_eligible():
    n = 300
    closes = _prices(n, {"A": np.full(n, 0.001), "B": np.full(n, 0.0005)})
    closes.loc[closes.index[:100], "B"] = np.nan  # so 200 pregoes
    assert list(h4_scores(closes).index) == ["A"]


def test_no_eligible_stock_means_empty_scores():
    closes = _prices(50, {"A": np.full(50, 0.001)})
    assert h4_scores(closes).empty


def test_momentum_skips_last_month():
    n = 300
    base = _prices(n, {"A": np.full(n, 0.001), "B": np.full(n, 0.0005)})
    spiked = base.copy()
    spiked.iloc[-10:, 1] *= 3  # B dispara nos ultimos 10 dias
    # momentum 12-1 ignora os ultimos 21 dias; vol muda, entao compara so o momentum
    from strategies.cross_section import momentum_12_1
    pd.testing.assert_series_equal(momentum_12_1(base), momentum_12_1(spiked))


def test_month_end_dates_are_last_trading_day():
    idx = pd.to_datetime(["2020-01-30", "2020-01-31", "2020-02-03", "2020-02-28", "2020-03-02"])
    assert list(month_end_dates(idx)) == list(pd.to_datetime(["2020-01-31", "2020-02-28"]))


def test_portfolio_return_and_cost():
    idx = pd.to_datetime(["2020-01-31", "2020-02-14", "2020-02-28", "2020-03-31"])
    closes = pd.DataFrame({"A": [100, 105, 110, 121], "B": [100, 100, 90, 90]}, index=idx, dtype=float)
    pick_a = lambda hist: ["A"]
    r = run_monthly_portfolio(closes, pick_a, cost_bps=10)
    # fev: compra total (custo 0,1%) e A rende 10%; mar: sem troca, A rende 10%
    assert r.loc["2020-02-28"] == pytest.approx(1.10 * (1 - 0.001) - 1)
    assert r.loc["2020-03-31"] == pytest.approx(0.10)


def test_turnover_cost_on_switch():
    idx = pd.to_datetime(["2020-01-31", "2020-02-28", "2020-03-31"])
    closes = pd.DataFrame({"A": [100, 100, 100], "B": [100, 100, 100]}, index=idx, dtype=float)
    pick = lambda hist: ["A"] if len(hist) == 1 else ["B"]
    r = run_monthly_portfolio(closes, pick, cost_bps=10)
    # troca total A -> B: vende 100% e compra 100% = 2 x 0,1%
    assert r.loc["2020-03-31"] == pytest.approx(-0.002)


def test_selection_sees_only_past():
    idx = pd.bdate_range("2020-01-01", "2020-04-30")
    closes = pd.DataFrame({"A": 100.0, "B": 100.0}, index=idx)
    seen = []
    run_monthly_portfolio(closes, lambda hist: seen.append(hist.index[-1]) or ["A"], cost_bps=0)
    assert seen == list(month_end_dates(idx))[:-1]
