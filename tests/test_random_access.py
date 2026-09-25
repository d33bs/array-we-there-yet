"""Tests for native row selection and for column requests that scale badly."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from array_we_there_yet import benchmark
from array_we_there_yet.benchmark import LayoutRunner, layout_runners
from array_we_there_yet.data import BenchmarkDataset, make_synthetic_dataset
from array_we_there_yet.validation import assert_mixed_retrieval

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


FEATURES = np.array([4, 1, 5])
NATIVE_MIXED = [
    ("parquet", "wide"),
    ("parquet", "fixed_array"),
    ("duckdb", "wide"),
    ("duckdb", "duckdb_array"),
    ("vortex", "wide"),
    ("vortex", "fixed_array"),
    ("lance", "wide"),
    ("lance", "fixed_array"),
]


@pytest.mark.parametrize(("backend", "layout"), NATIVE_MIXED)
def test_mixed_retrieval_returns_the_requested_rows_and_features(
    backend: str, layout: str, tmp_path: Path
) -> None:
    """Metadata and features match the source for unsorted rows and features."""
    dataset = _dataset()
    runner = _runner(backend, layout)
    artifact = runner.write(dataset, tmp_path / "artifact")

    frame = benchmark._read_mixed(runner, artifact, dataset, ROWS, FEATURES)

    assert_mixed_retrieval(frame, dataset=dataset, rows=ROWS, features=FEATURES)


def test_mixed_retrieval_uses_native_row_selection_in_lance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lance mixed retrieval calls `take`, never a full `to_table` scan."""
    dataset = _dataset()
    calls: list[str] = []

    class Spy:
        def __init__(self, inner: Any) -> None:  # noqa: ANN401
            self._inner = inner

        def take(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
            calls.append("take")
            return self._inner.take(*args, **kwargs)

        def to_table(self, *_: Any, **__: Any) -> Any:  # noqa: ANN401
            message = "to_table scans every row"
            raise AssertionError(message)

    for layout in ["wide", "fixed_array"]:
        runner = _runner("lance", layout)
        artifact = runner.write(dataset, tmp_path / layout)
        original = benchmark.lance.dataset
        monkeypatch.setattr(
            benchmark.lance, "dataset", lambda *a, _o=original, **k: Spy(_o(*a, **k))
        )
        benchmark._read_mixed(runner, artifact, dataset, ROWS, FEATURES)
        monkeypatch.setattr(benchmark.lance, "dataset", original)

    assert calls == ["take", "take"]


def test_mixed_retrieval_uses_native_row_selection_in_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Parquet mixed retrieval takes rows from a dataset, not a full table read."""
    dataset = _dataset()

    def forbidden(*_: Any, **__: Any) -> None:  # noqa: ANN401
        message = "read_table reads every row"
        raise AssertionError(message)

    monkeypatch.setattr(benchmark.pq, "read_table", forbidden)
    for layout in ["wide", "fixed_array"]:
        runner = _runner("parquet", layout)
        artifact = runner.write(dataset, tmp_path / layout)

        frame = benchmark._read_mixed(runner, artifact, dataset, ROWS, FEATURES)

        assert len(frame) == len(ROWS)


def test_mixed_retrieval_uses_row_ids_in_duckdb(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DuckDB mixed retrieval filters by row id in SQL."""
    dataset = _dataset()
    queries: list[str] = []
    original = benchmark._duckdb_connect

    class Recorder:
        def __init__(self, connection: Any) -> None:  # noqa: ANN401
            self._connection = connection

        def __enter__(self) -> "Recorder":
            self._connection.__enter__()
            return self

        def __exit__(self, *exc: object) -> None:
            self._connection.__exit__(*exc)

        def execute(self, sql: str, *args: Any) -> Any:  # noqa: ANN401
            queries.append(sql)
            return self._connection.execute(sql, *args)

    for layout in ["wide", "duckdb_array"]:
        runner = _runner("duckdb", layout)
        artifact = runner.write(dataset, tmp_path / f"{layout}.duckdb")
        queries.clear()
        monkeypatch.setattr(
            benchmark, "_duckdb_connect", lambda *a, **k: Recorder(original(*a, **k))
        )
        benchmark._read_mixed(runner, artifact, dataset, ROWS, FEATURES)
        monkeypatch.setattr(benchmark, "_duckdb_connect", original)

        assert any("rowid IN" in query for query in queries)


def test_mixed_retrieval_does_not_read_every_row_in_vortex(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Vortex mixed retrieval scans by row index, not with a full read."""
    dataset = _dataset()

    def forbidden(*_: Any, **__: Any) -> None:  # noqa: ANN401
        message = "_read_vortex_all reads every row"
        raise AssertionError(message)

    monkeypatch.setattr(benchmark, "_read_vortex_all", forbidden)
    for layout in ["wide", "fixed_array"]:
        runner = _runner("vortex", layout)
        artifact = runner.write(dataset, tmp_path / layout)

        frame = benchmark._read_mixed(runner, artifact, dataset, ROWS, FEATURES)

        assert len(frame) == len(ROWS)
