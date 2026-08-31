import pandas as pd

from models.dataset import build_dataset
from models.walk_forward import generate_walk_forward_splits


def test_purge_removes_labels_contaminated_by_planted_price_jump():
    """Concrete, value-level proof that purging prevents leakage -- not
    just index arithmetic (already covered in test_walk_forward.py), but an
    actual planted event.

    Construct a price series that is flat, then jumps sharply and
    permanently at a specific day. Any training row whose label window
    (horizon_days forward) reaches that jump day has a label that is
    almost entirely explained by information from at/after the jump --
    exactly the kind of boundary contamination purging exists to remove.

    Without purging, those contaminated rows remain in the training set.
    With purging (purge_size >= horizon_days), they are excluded.
    """

    horizon = 20
    n = 2000
    index = pd.bdate_range("2015-01-01", periods=n)
    close = pd.Series(100.0, index=index)

    jump_position = 1000
    close.iloc[jump_position:] = 200.0  # sudden, permanent price jump

    dataset = build_dataset(
        close,
        momentum_windows=(5, 20, 60),
        volatility_window=20,
        sma_distance_window=50,
        horizon_days=horizon,
    )

    jump_date = close.index[jump_position]
    test_start_pos = dataset.index.get_loc(jump_date)
    test_size = 50
    min_train_size = test_start_pos

    unpurged_splits = generate_walk_forward_splits(
        len(dataset), min_train_size=min_train_size, test_size=test_size, purge_size=0
    )
    train_idx_unpurged, _ = unpurged_splits[0]

    boundary_rows = dataset.iloc[train_idx_unpurged[-horizon:]]
    # These rows' labels look forward exactly into the jump -- almost all
    # should show label=1 (price went up), a stark, non-random pattern that
    # has nothing to do with genuine predictive skill and everything to do
    # with peeking at the test-period event.
    assert boundary_rows["label"].mean() > 0.9

    purged_splits = generate_walk_forward_splits(
        len(dataset), min_train_size=min_train_size, test_size=test_size, purge_size=horizon
    )
    train_idx_purged, test_idx_purged = purged_splits[0]

    contaminated_positions = set(train_idx_unpurged[-horizon:])
    remaining_positions = set(train_idx_purged)

    assert contaminated_positions.isdisjoint(remaining_positions)
    assert len(train_idx_purged) == len(train_idx_unpurged) - horizon
