"""Validation helpers for benchmark backends."""

from __future__ import annotations

import numpy as np
import pandas as pd

from array_we_there_yet.data import BenchmarkDataset


def assert_same_matrix(actual: np.ndarray, expected: np.ndarray) -> None:
    """Make sure that a backend reconstructs the reference matrix."""
    if actual.shape != expected.shape:
        msg = f"shape mismatch: {actual.shape} != {expected.shape}"
        raise AssertionError(msg)
    if actual.dtype != np.float32:
        msg = f"dtype mismatch: {actual.dtype} != float32"
        raise AssertionError(msg)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def assert_rows(actual: np.ndarray, expected: np.ndarray, rows: np.ndarray) -> None:
    """Make sure that random profile reads return the selected rows."""
    assert_same_matrix(actual, expected[rows, :])


def assert_features(
    actual: np.ndarray,
    expected: np.ndarray,
    features: np.ndarray,
) -> None:
    """Make sure that feature projection keeps order and values."""
    assert_same_matrix(actual, expected[:, features])


def assert_mixed_retrieval(
    actual: pd.DataFrame,
    *,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> None:
    """Make sure mixed reads return selected metadata and feature values."""
    expected_columns = [
        *dataset.metadata.columns.to_list(),
        *(dataset.feature_names[index] for index in features),
    ]
    if actual.columns.to_list() != expected_columns:
        msg = f"columns mismatch: {actual.columns.to_list()} != {expected_columns}"
        raise AssertionError(msg)

    actual_metadata = actual[dataset.metadata.columns].reset_index(drop=True)
    expected_rows = dataset.metadata.iloc[rows].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_metadata, expected_rows)

    names = [dataset.feature_names[index] for index in features]
    actual_matrix = actual[names].to_numpy(dtype=np.float32)
    assert_same_matrix(actual_matrix, dataset.matrix[np.ix_(rows, features)])
