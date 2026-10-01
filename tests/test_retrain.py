import numpy as np
import pandas as pd
import pytest

from paper_trading.model_store import ModelVersion, list_versions, model_for_date, register_version, save_frozen_model
from paper_trading.predictions import versioned_probabilities
from paper_trading.retrain import evaluate_challenger, promote, train_on_history


def _close(n, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, n))), index=pd.bdate_range("2015-01-01", periods=n))


@pytest.fixture
def store(tmp_path, monkeypatch):
    import paper_trading.model_store as ms
    monkeypatch.setattr(ms, "MODEL_DIR", tmp_path / "models")
    return ms


def _freeze(close, ticker="T"):
    cut = close.index[1200]
    trained = train_on_history(close, cut)
    save_frozen_model(ticker, trained, str(cut.date()))
    return cut


def test_train_on_history_never_uses_prices_after_the_cutoff():
    """Truncating the future must not change anything the model learned:
    a model trained 'through T' from a long series equals one trained on a
    series that simply ends at T.
    """

    close = _close(1400)
    cut = close.index[1200]
    from_long = train_on_history(close, cut)
    from_short = train_on_history(close.loc[:cut], cut)
    X = np.random.default_rng(0).normal(size=(5, 5))
    cols = ["momentum_5", "momentum_20", "momentum_60", "volatility_20", "sma_distance_50"]
    df = pd.DataFrame(X, columns=cols)
    assert from_long.xgboost_calibrated.predict_proba(df) == pytest.approx(from_short.xgboost_calibrated.predict_proba(df))


def test_versions_start_with_the_frozen_base_and_never_backdate(store):
    close = _close(1400)
    cut = _freeze(close)
    versions = list_versions("T")
    assert [v.version for v in versions] == [1]
    with pytest.raises(ValueError, match="later"):
        register_version("T", versions[0].trained, cut, cut)  # same date as base -> refused


def test_model_for_date_picks_the_governing_version(store):
    close = _close(1400)
    cut = _freeze(close)
    v1 = list_versions("T")[0]
    register_version("T", v1.trained, close.index[1300], close.index[1301])
    versions = list_versions("T")
    assert model_for_date(versions, close.index[1250]).version == 1
    assert model_for_date(versions, close.index[1301]).version == 2
    assert model_for_date(versions, close.index[1399]).version == 2
    assert model_for_date(versions, close.index[0]) is None


def test_versioned_probabilities_use_each_versions_own_model(store):
    close = _close(1400)
    cut = _freeze(close)
    v1 = list_versions("T")[0]
    other = train_on_history(close, close.index[1000])  # a genuinely different model
    register_version("T", other, close.index[1000], close.index[1300])
    versions = list_versions("T")
    probs = versioned_probabilities(close, versions)
    d_old, d_new = close.index[1250], close.index[1350]
    from paper_trading.predictions import inference_features
    f = inference_features(close)
    assert probs[d_old] == pytest.approx(v1.trained.xgboost_calibrated.predict_proba(f.loc[[d_old]])[0, 1])
    assert probs[d_new] == pytest.approx(other.xgboost_calibrated.predict_proba(f.loc[[d_new]])[0, 1])


def test_challenger_is_insufficient_with_little_live_data(store):
    close = _close(1400)
    _freeze(close)
    result = evaluate_challenger(close, list_versions("T")[0])
    assert result.verdict == "INSUFICIENTE"  # ~199 matured days -> ~5 effective eval windows < 6... or fewer


def test_promotion_is_refused_without_evidence_and_allowed_with_force(store):
    close = _close(1400)
    _freeze(close)
    latest = list_versions("T")[-1]
    comparison = evaluate_challenger(close, latest)
    with pytest.raises(PermissionError):
        promote("T", close, latest, comparison, force=False)
    new = promote("T", close, latest, comparison, force=True)
    assert new.version == 2
    assert new.effective_from == close.index[-1] + pd.Timedelta(days=1)
    assert [v.version for v in list_versions("T")] == [1, 2]


def test_comparison_has_no_leakage_between_challenger_training_and_evaluation(store):
    close = _close(1400, seed=7)
    cut = close.index[600]  # plenty of matured live days -> a real verdict is computed
    save_frozen_model("T", train_on_history(close, cut), str(cut.date()))
    result = evaluate_challenger(close, list_versions("T")[0])
    assert result.verdict in {"CHALLENGER_MELHOR", "SEM_DIFERENCA", "INCUMBENTE_MELHOR"}
    assert result.n_effective >= 6
    # Training labels end at train_end; evaluation starts strictly after it.
    assert result.train_end < result.eval_start
    assert result.challenger is not None
