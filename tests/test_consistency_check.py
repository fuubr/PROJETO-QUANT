import numpy as np
import pandas as pd
import pytest


def _make_fake_close(n: int, seed: int) -> pd.Series:
    index = pd.bdate_range("2015-01-01", periods=n)
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0003, 0.012, n)
    return pd.Series(100 * np.exp(np.cumsum(returns)), index=index)


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    import paper_trading.model_store as model_store
    import paper_trading.state as state

    state_dir = tmp_path / "state"
    model_dir = tmp_path / "models"
    monkeypatch.setattr(state, "STATE_DIR", state_dir)
    monkeypatch.setattr(model_store, "MODEL_DIR", model_dir)
    return state_dir, model_dir


def test_batch_consistency_check_matches_daily_script_exactly(isolated_dirs):
    """The central assumption behind Stage 6's whole design: running the
    daily script day-by-day and running the batch engine over the same
    period with the same frozen model must agree almost exactly, since
    both call the same underlying step function. This is what makes
    "divergence" a meaningful signal instead of expected noise.
    """

    from config import (
        ASSET_LEVEL_CIRCUIT_BREAKER_PCT, CIRCUIT_BREAKER_DRAWDOWN_PCT,
        SLIPPAGE_BPS, STOP_LOSS_PCT, TRANSACTION_FEE_BPS, VAR_CONFIDENCE_LEVEL,
    )
    from paper_trading.consistency_check import check_consistency
    from paper_trading.model_store import load_frozen_model
    from paper_trading.state import load_log
    from run_stage6_paper_trading_daily import run_one_day_for_ticker

    close = _make_fake_close(1400, seed=9)

    # Simula 5 execucoes diarias, estendendo a serie um dia por vez.
    running_close = close.copy()
    for i in range(5):
        next_date = pd.bdate_range(running_close.index[-1], periods=2)[1]
        rng = np.random.default_rng(200 + i)
        new_price = running_close.iloc[-1] * (1 + rng.normal(0.0003, 0.012))
        running_close = pd.concat([running_close, pd.Series([new_price], index=[next_date])])
        run_one_day_for_ticker("CONSISTX", running_close)

    log = load_log("CONSISTX")
    trained, _ = load_frozen_model("CONSISTX")

    result = check_consistency(
        "CONSISTX", running_close, trained, log,
        fee_bps=TRANSACTION_FEE_BPS, slippage_bps=SLIPPAGE_BPS,
        circuit_breaker_pct=CIRCUIT_BREAKER_DRAWDOWN_PCT,
        asset_circuit_breaker_pct=ASSET_LEVEL_CIRCUIT_BREAKER_PCT,
        stop_loss_pct=STOP_LOSS_PCT, var_confidence_level=VAR_CONFIDENCE_LEVEL,
    )

    assert result.matches, (
        f"Divergencia inesperada: max_relative_diff={result.max_relative_equity_diff}"
    )
    assert result.days_compared == len(log)
