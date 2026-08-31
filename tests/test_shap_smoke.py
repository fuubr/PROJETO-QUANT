import numpy as np
import pandas as pd

from run_stage3_model import _run_shap_analysis


def test_shap_analysis_runs_without_error(tmp_path, monkeypatch, capsys):
    """Smoke test for the SHAP code path, which has no other automated
    coverage -- confirmed during an external audit review to be a real gap:
    a broken shap.summary_plot(..., ax=ax) call (invalid keyword argument in
    the installed shap version) would only have failed when actually
    running the script, never during `pytest`. This test exercises the
    same function pytest would otherwise never touch.
    """

    from backtest import reporting
    monkeypatch.setattr(reporting, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr("run_stage3_model.RESULTS_DIR", tmp_path)

    rng = np.random.default_rng(0)
    n = 300
    index = pd.bdate_range("2020-01-01", periods=n)
    dataset = pd.DataFrame(
        {
            "momentum_5": rng.normal(size=n),
            "momentum_20": rng.normal(size=n),
            "momentum_60": rng.normal(size=n),
            "volatility_20": rng.uniform(0.005, 0.02, size=n),
            "sma_distance_50": rng.normal(size=n),
            "label": rng.integers(0, 2, size=n).astype(float),
        },
        index=index,
    )

    _run_shap_analysis(dataset, "SMOKE_TEST")

    output_path = tmp_path / "shap_summary_SMOKE_TEST.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0

    captured = capsys.readouterr()
    assert "Grafico SHAP salvo em" in captured.out
