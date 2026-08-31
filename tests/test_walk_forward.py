import numpy as np
import pytest

from models.walk_forward import generate_walk_forward_splits


def test_walk_forward_train_always_precedes_test():
    splits = generate_walk_forward_splits(n=1000, min_train_size=200, test_size=50, purge_size=20)

    assert len(splits) > 0
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min()


def test_walk_forward_purge_removes_boundary_rows():
    """The last purge_size positions immediately before each test block must
    NOT appear in that fold's training set -- those rows' labels look
    forward into the test period.
    """

    purge_size = 20
    splits = generate_walk_forward_splits(n=1000, min_train_size=200, test_size=50, purge_size=purge_size)

    for train_idx, test_idx in splits:
        gap = test_idx.min() - train_idx.max()
        assert gap > purge_size  # strictly more than purge_size positions apart


def test_walk_forward_is_expanding_not_rolling():
    """Later folds must include strictly more training data than earlier
    folds (expanding window) -- train set size should never shrink.
    """

    splits = generate_walk_forward_splits(n=1000, min_train_size=200, test_size=50, purge_size=20)

    train_sizes = [len(train_idx) for train_idx, _ in splits]
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) > 1  # actually growing, not constant


def test_walk_forward_test_blocks_do_not_overlap():
    splits = generate_walk_forward_splits(n=1000, min_train_size=200, test_size=50, purge_size=20)

    all_test_indices = np.concatenate([test_idx for _, test_idx in splits])
    assert len(all_test_indices) == len(set(all_test_indices))


def test_walk_forward_returns_empty_list_when_data_too_small():
    splits = generate_walk_forward_splits(n=50, min_train_size=200, test_size=50, purge_size=20)

    assert splits == []


def test_walk_forward_rejects_invalid_params():
    with pytest.raises(ValueError, match="min_train_size"):
        generate_walk_forward_splits(n=1000, min_train_size=0, test_size=50, purge_size=20)
    with pytest.raises(ValueError, match="test_size"):
        generate_walk_forward_splits(n=1000, min_train_size=200, test_size=0, purge_size=20)
    with pytest.raises(ValueError, match="purge_size"):
        generate_walk_forward_splits(n=1000, min_train_size=200, test_size=50, purge_size=-1)
