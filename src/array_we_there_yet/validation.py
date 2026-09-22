"""Validation helpers for benchmark backends."""

from __future__ import annotations

import numpy as np


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
