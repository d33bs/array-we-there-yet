"""Tests for native row selection and for column requests that scale badly."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from array_we_there_yet import benchmark
from array_we_there_yet.benchmark import LayoutRunner, layout_runners
from array_we_there_yet.data import BenchmarkDataset, make_synthetic_dataset

ROWS = np.array([37, 3, 19, 0, 25])  # unsorted on purpose
NATIVE_ROW_READERS = [
    ("parquet", "wide", "_read_parquet_wide_matrix"),
    ("parquet", "fixed_array", "_read_parquet_fixed_matrix"),
    ("duckdb", "wide", "_read_duckdb_wide_matrix"),
    ("duckdb", "duckdb_array", "_read_duckdb_array_matrix"),
    ("vortex", "wide", "_read_vortex_wide_matrix"),
    ("vortex", "fixed_array", "_read_vortex_fixed_matrix"),
    ("lance", "wide", "_read_lance_wide_matrix"),
    ("lance", "fixed_array", "_read_lance_fixed_matrix"),
]


def _dataset() -> BenchmarkDataset:
    return make_synthetic_dataset(rows=40, dimensions=6, seed=11)


def _runner(backend: str, layout: str) -> LayoutRunner:
    return next(
        runner
        for runner in layout_runners()
        if (runner.backend, runner.layout, runner.profile)
        == (backend, layout, "default")
    )


@pytest.mark.parametrize(("backend", "layout", "matrix_reader"), NATIVE_ROW_READERS)
def test_random_rows_select_rows_without_reading_the_whole_matrix(
    backend: str,
    layout: str,
    matrix_reader: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Row selection uses the format's own row access, not read-all-then-index."""
    dataset = _dataset()

    def forbidden(*_: Any, **__: Any) -> None:  # noqa: ANN401
        message = f"{matrix_reader} reads every row"
        raise AssertionError(message)

    monkeypatch.setattr(benchmark, matrix_reader, forbidden)
    runner = _runner(backend, layout)
    artifact = runner.write(dataset, tmp_path / "artifact")

    rows = runner.read_rows(artifact, dataset, ROWS)

    assert rows.dtype == np.float32
    np.testing.assert_allclose(rows, dataset.matrix[ROWS], rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(("backend", "layout", "_"), NATIVE_ROW_READERS)
def test_random_rows_keep_the_requested_order(
    backend: str, layout: str, _: str, tmp_path: Path
) -> None:
    """Rows come back in the order that was requested, not in file order."""
    dataset = _dataset()
    runner = _runner(backend, layout)
    artifact = runner.write(dataset, tmp_path / "artifact")
    requested = np.array([39, 0, 20, 5])

    rows = runner.read_rows(artifact, dataset, requested)

    np.testing.assert_allclose(rows[0], dataset.matrix[39], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rows[1], dataset.matrix[0], rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rows[3], dataset.matrix[5], rtol=1e-6, atol=1e-6)


class _RecordingDataset:
    """Wrap a Lance dataset and remember how many columns each call requested."""

    requested: list[int] = []  # noqa: RUF012

    def __init__(self, inner: Any) -> None:  # noqa: ANN401
        self._inner = inner

    def to_table(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        self.requested.append(len(kwargs.get("columns") or []))
        return self._inner.to_table(*args, **kwargs)

    def take(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        self.requested.append(len(kwargs.get("columns") or []))
        return self._inner.take(*args, **kwargs)


@pytest.mark.parametrize("read", ["matrix", "rows"])
def test_lance_wide_never_requests_every_column_by_name(
    read: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Naming thousands of columns costs about 1.35 ms each in Lance, so avoid it."""
    dataset = make_synthetic_dataset(rows=40, dimensions=64, seed=3)
    runner = _runner("lance", "wide")
    artifact = runner.write(dataset, tmp_path / "artifact")
    original = benchmark.lance.dataset
    _RecordingDataset.requested = []
    monkeypatch.setattr(
        benchmark.lance, "dataset", lambda *a, **k: _RecordingDataset(original(*a, **k))
    )

    if read == "matrix":
        matrix = runner.read_matrix(artifact, dataset)
        np.testing.assert_allclose(matrix, dataset.matrix, rtol=1e-6, atol=1e-6)
    else:
        rows = runner.read_rows(artifact, dataset, ROWS)
        np.testing.assert_allclose(rows, dataset.matrix[ROWS], rtol=1e-6, atol=1e-6)

    assert _RecordingDataset.requested
    assert max(_RecordingDataset.requested) < dataset.dimensions
