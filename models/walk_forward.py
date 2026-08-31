"""Purged walk-forward split generation.

Standard walk-forward (train on past, test on future block, slide forward)
is necessary but NOT sufficient to prevent leakage when the label itself
looks forward in time. A training row dated shortly before a test block's
start has a label that depends on price up to `horizon_days` later --
which can fall inside the test period. Without removing ("purging") those
boundary rows from training, the model would be trained on information
that overlaps with what it's later evaluated on, even though every row's
own date is technically "in the past".

This module produces POSITIONAL indices (not dates) into an already
sorted-by-date, already-deduplicated dataset. Callers are responsible for
sorting their dataset by date before use -- this module does not re-sort,
so a caller passing unsorted data would get silently wrong splits, which is
why models/dataset.py always returns date-sorted output.
"""

from __future__ import annotations

import numpy as np


def generate_walk_forward_splits(
    n: int,
    min_train_size: int,
    test_size: int,
    purge_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate (train_idx, test_idx) positional index pairs for an
    expanding-window, purged walk-forward split.

    Train always starts at position 0 (expanding window: each fold sees
    strictly more history than the last, never less). Each fold's train
    set excludes the last `purge_size` rows immediately before its test
    block, since those rows' labels look forward into the test period.

    Returns an empty list if there isn't enough data for even one fold --
    this is a valid, non-error outcome for small datasets (e.g. in tests),
    not a bug.
    """

    if min_train_size < 1:
        raise ValueError("min_train_size must be at least 1")
    if test_size < 1:
        raise ValueError("test_size must be at least 1")
    if purge_size < 0:
        raise ValueError("purge_size must be non-negative")

    splits = []
    test_start = min_train_size

    while test_start + test_size <= n:
        train_end = test_start - purge_size
        if train_end > 0:
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, test_start + test_size)
            splits.append((train_idx, test_idx))
        test_start += test_size

    return splits
