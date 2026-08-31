import pandas as pd

from backtest.reporting import save_summary


def test_save_summary_writes_readable_csv(tmp_path, monkeypatch):
    monkeypatch.setattr("backtest.reporting.RESULTS_DIR", tmp_path)

    summary = pd.DataFrame(
        {"total_return": [0.1, 0.2]}, index=pd.Index(["asset_a", "asset_b"], name="asset")
    )

    path = save_summary(summary, stage_name="test_stage")

    assert path.exists()
    assert path.parent == tmp_path
    assert path.name.startswith("test_stage_")
    assert path.suffix == ".csv"

    reloaded = pd.read_csv(path, index_col="asset")
    pd.testing.assert_frame_equal(reloaded, summary)
