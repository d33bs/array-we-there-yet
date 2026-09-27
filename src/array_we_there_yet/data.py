"""Synthetic data for storage-layout benchmarks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BenchmarkDataset:
    """A deterministic high-dimensional profile dataset."""

    name: str
    matrix: np.ndarray
    metadata: pd.DataFrame
    feature_names: list[str]

    @property
    def rows(self) -> int:
        """Return the number of profiles."""
        return int(self.matrix.shape[0])

    @property
    def dimensions(self) -> int:
        """Return the number of numeric features."""
        return int(self.matrix.shape[1])


def make_synthetic_dataset(
    *,
    rows: int,
    dimensions: int,
    seed: int = 42,
    name: str = "synthetic",
) -> BenchmarkDataset:
    """Create one deterministic matrix and matching metadata.

    Args:
        rows: Number of profile rows.
        dimensions: Number of numeric features per row.
        seed: Random seed for repeatable values.
        name: Dataset label used in result records.

    Returns:
        The generated dataset.
    """
    if rows < 1:
        msg = "rows must be at least 1"
        raise ValueError(msg)
    if dimensions < 1:
        msg = "dimensions must be at least 1"
        raise ValueError(msg)

    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(rows, dimensions)).astype(np.float32)
    sample_ids = [f"sample_{index:08d}" for index in range(rows)]
    metadata = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "plate_id": [f"plate_{index % 8:02d}" for index in range(rows)],
            "well_id": [
                f"{chr(65 + (index % 16))}{(index % 24) + 1:02d}"
                for index in range(rows)
            ],
        }
    )
    feature_names = [f"feature_{index:04d}" for index in range(dimensions)]

    return BenchmarkDataset(
        name=name,
        matrix=matrix,
        metadata=metadata,
        feature_names=feature_names,
    )


def wide_dataframe(dataset: BenchmarkDataset) -> pd.DataFrame:
    """Return the wide representation of a dataset."""
    features = pd.DataFrame(dataset.matrix, columns=pd.Index(dataset.feature_names))
    return pd.concat([dataset.metadata.reset_index(drop=True), features], axis=1)


def array_dataframe(dataset: BenchmarkDataset) -> pd.DataFrame:
    """Return the one-column feature-vector representation of a dataset."""
    result = dataset.metadata.copy()
    result["features"] = [row.tolist() for row in dataset.matrix]
    return result
