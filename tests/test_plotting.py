import numpy as np
import pandas as pd

from backtest.plotting import plot_equity_curves


def test_plot_equity_curves_creates_file(tmp_path):
    index = pd.bdate_range("2020-01-01", periods=50)
    rng = np.random.default_rng(0)
    curves = {
        "estrategia_a": pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0005, 0.01, 50)), index=index),
        "estrategia_b": pd.Series(100_000 * np.cumprod(1 + rng.normal(0.0002, 0.015, 50)), index=index),
    }
    breaker_dates = {"estrategia_a": None, "estrategia_b": index[20]}
    output_path = tmp_path / "equity.png"

    plot_equity_curves(curves, breaker_dates, "Teste", output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
