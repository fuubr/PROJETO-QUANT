import json

import numpy as np
import pandas as pd
import pytest

import data.rates as rates


class FakeResponse:
    def __init__(self, payload=None, text=None):
        self._payload, self.text, self.status_code = payload, text, 200

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(rates, "CACHE_DIR", tmp_path / "rates")


def test_annual_daily_roundtrip_and_known_value():
    assert rates.annual_to_daily(0.0) == 0.0
    assert rates.daily_to_annual(rates.annual_to_daily(0.10)) == pytest.approx(0.10)
    # the real BCB print seen in production: 0.05166 %/day is roughly 13.9% a.a.
    assert rates.daily_to_annual(0.0005166) == pytest.approx(0.139, abs=0.002)


def test_cdi_percent_is_converted_and_long_ranges_are_chunked(monkeypatch):
    calls = []

    def fake_get(url, params=None, **kw):
        calls.append(params)
        start = pd.to_datetime(params["dataInicial"], format="%d/%m/%Y")
        return FakeResponse([{"data": start.strftime("%d/%m/%Y"), "valor": "0.050000"}])

    monkeypatch.setattr(rates, "_http_get", fake_get)
    series = rates.fetch_cdi_daily("2015-01-01", "2026-09-30")
    assert len(calls) == 3  # 2015-19, 2020-24, 2025-26: each window under the BCB 10-year limit
    assert series.iloc[0] == pytest.approx(0.0005)  # "0.05" percent -> 0.0005 decimal


def test_cdi_rejects_a_unit_mistake(monkeypatch):
    monkeypatch.setattr(rates, "_http_get", lambda *a, **k: FakeResponse([{"data": "01/09/2026", "valor": "5.0"}]))
    with pytest.raises(ValueError, match="plausible"):
        rates.fetch_cdi_daily("2026-09-01", "2026-09-30")  # 5 %/day would be absurd -> not divided by 100 upstream


def test_tbill_parses_holidays_and_converts_annual_percent(monkeypatch):
    csv = "observation_date,DTB3\n2026-09-21,4.00\n2026-09-22,.\n2026-09-23,4.00\n"
    monkeypatch.setattr(rates, "_http_get", lambda *a, **k: FakeResponse(text=csv))
    series = rates.fetch_us_tbill_daily("2026-09-21", "2026-09-23")
    assert len(series) == 2  # the "." holiday row is dropped, not turned into zero
    assert series.iloc[0] == pytest.approx((1.04) ** (1 / 252) - 1)


def test_tbill_rejects_out_of_range_values(monkeypatch):
    csv = "observation_date,DTB3\n2026-09-21,400.0\n"
    monkeypatch.setattr(rates, "_http_get", lambda *a, **k: FakeResponse(text=csv))
    with pytest.raises(ValueError, match="plausible"):
        rates.fetch_us_tbill_daily("2026-09-21", "2026-09-21")


def test_risk_free_is_aligned_forward_filling_missing_days(monkeypatch):
    raw = pd.Series([0.0004, 0.0006], index=pd.to_datetime(["2026-09-01", "2026-09-03"]))
    monkeypatch.setattr(rates, "fetch_cdi_daily", lambda s, e: raw)
    index = pd.bdate_range("2026-09-01", "2026-09-04")
    series, label = rates.risk_free_daily("PETR4.SA", index)
    assert "BCB" in label
    assert series.loc["2026-09-02"] == pytest.approx(0.0004)  # a day with no print inherits the previous one
    assert series.loc["2026-09-04"] == pytest.approx(0.0006)
    assert series.notna().all()


def test_second_call_uses_the_cache(monkeypatch):
    counter = {"n": 0}

    def fetcher(start, end):
        counter["n"] += 1
        return pd.Series([0.0003], index=pd.to_datetime(["2026-09-01"]))

    monkeypatch.setattr(rates, "fetch_us_tbill_daily", fetcher)
    index = pd.bdate_range("2026-09-01", "2026-09-02")
    rates.risk_free_daily("SPY", index)
    rates.risk_free_daily("SPY", index)
    assert counter["n"] == 1


def test_fallback_is_loud_and_labelled_assumed(monkeypatch, capsys):
    def broken(*a, **k):
        raise RuntimeError("network down")

    monkeypatch.setattr(rates, "fetch_cdi_daily", broken)
    index = pd.bdate_range("2026-09-01", periods=5)
    series, label = rates.risk_free_daily("VALE3.SA", index)
    assert label.startswith("ASSUMIDA")  # never presented as real data
    assert "AVISO" in capsys.readouterr().out
    assert series.iloc[0] == pytest.approx(rates.annual_to_daily(0.10))
