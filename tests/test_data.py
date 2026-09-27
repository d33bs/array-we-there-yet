"""Tests for benchmark data generation."""

import numpy as np

from array_we_there_yet.data import make_synthetic_dataset


def test_make_synthetic_dataset_is_deterministic() -> None:
    """Synthetic data uses the same values for the same seed."""
    first = make_synthetic_dataset(rows=5, dimensions=3, seed=7)
    second = make_synthetic_dataset(rows=5, dimensions=3, seed=7)

    assert first.metadata.equals(second.metadata)
    assert first.feature_names == ["feature_0000", "feature_0001", "feature_0002"]
    np.testing.assert_array_equal(first.matrix, second.matrix)
    assert first.matrix.dtype == np.float32
