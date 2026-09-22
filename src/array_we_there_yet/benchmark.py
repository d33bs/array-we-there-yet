"""Benchmark wide tables against fixed-size and packed array layouts."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pyarrow import ipc

from array_we_there_yet.data import (
    BenchmarkDataset,
    array_dataframe,
    make_synthetic_dataset,
    wide_dataframe,
)
from array_we_there_yet.validation import (
    assert_features,
    assert_rows,
    assert_same_matrix,
)

Layout = Literal[
    "wide",
    "fixed_array",
    "delimited_array",
    "json_array",
    "duckdb_array",
]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Benchmark parameters for a reproducible synthetic run."""

    rows: int = 2_000
    dimensions: tuple[int, ...] = (16, 64, 128)
    measured_repetitions: int = 3
    warmups: int = 1
    seed: int = 42
    random_row_count: int = 128
    feature_projection_count: int = 8
    threads: int = 1
    output_dir: Path = Path("results")
    artifact_dir: Path = Path("results/artifacts")
    figure_dir: Path = Path("figures")


@dataclass(frozen=True)
class BenchmarkResult:
    """One raw benchmark measurement."""

    backend: str
    layout: str
    dataset: str
    rows: int
    dimensions: int
    dtype: str
    operation: str
    operation_parameter: str
    iteration: int
    elapsed_seconds: float
    artifact_bytes: int
    peak_memory_bytes: int | None
    threads: int
    compression: str
    timestamp: str
    git_commit: str
    rows_per_second: float | None = None
    values_per_second: float | None = None


@dataclass(frozen=True)
class Artifact:
    """A written benchmark artifact and its metadata."""

    path: Path
    bytes: int


@dataclass(frozen=True)
class LayoutRunner:
    """Functions for one backend and layout."""

    backend: str
    layout: Layout
    compression: str
    write: Callable[[BenchmarkDataset, Path], Artifact]
    read_all: Callable[[Artifact, BenchmarkDataset], Any]
    read_matrix: Callable[[Artifact, BenchmarkDataset], np.ndarray]
    read_rows: Callable[[Artifact, BenchmarkDataset, np.ndarray], np.ndarray]
    read_features: Callable[[Artifact, BenchmarkDataset, np.ndarray], np.ndarray]
    compute_norm: Callable[[Artifact, BenchmarkDataset], np.ndarray]


def run_benchmarks(config: BenchmarkConfig) -> pd.DataFrame:
    """Run the configured benchmark and write raw outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    if config.artifact_dir.exists():
        shutil.rmtree(config.artifact_dir)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)

    records: list[BenchmarkResult] = []
    timestamp = pd.Timestamp.utcnow().isoformat()
    git_commit = _git_commit()

    for dimensions in config.dimensions:
        dataset = make_synthetic_dataset(
            rows=config.rows,
            dimensions=dimensions,
            seed=config.seed,
            name="synthetic",
        )
        row_count = min(config.random_row_count, dataset.rows)
        feature_count = min(config.feature_projection_count, dataset.dimensions)
        selected_rows = _selection(
            dataset.rows, row_count, seed=config.seed + dimensions
        )
        selected_features = _selection(
            dataset.dimensions,
            feature_count,
            seed=config.seed + dimensions + 1,
        )

        for runner in layout_runners():
            artifact = _measure_writes(
                config=config,
                runner=runner,
                dataset=dataset,
                timestamp=timestamp,
                git_commit=git_commit,
                records=records,
            )
            _measure_reads(
                config=config,
                runner=runner,
                dataset=dataset,
                artifact=artifact,
                selected_rows=selected_rows,
                selected_features=selected_features,
                timestamp=timestamp,
                git_commit=git_commit,
                records=records,
            )

    raw = pd.DataFrame(asdict(record) for record in records)
    raw_path = config.output_dir / "raw_results.parquet"
    raw.to_parquet(raw_path, index=False)
    raw.to_csv(config.output_dir / "raw_results.csv", index=False)
    write_environment(config.output_dir, config)
    return raw


def summarize_results(
    raw: pd.DataFrame, output_dir: Path = Path("results")
) -> pd.DataFrame:
    """Summarize raw timings with median and interquartile values."""
    group_columns = [
        "backend",
        "layout",
        "dataset",
        "rows",
        "dimensions",
        "dtype",
        "operation",
        "operation_parameter",
        "threads",
        "compression",
    ]
    summary = (
        raw.groupby(group_columns, dropna=False)
        .agg(
            median_seconds=("elapsed_seconds", "median"),
            q25_seconds=("elapsed_seconds", lambda values: values.quantile(0.25)),
            q75_seconds=("elapsed_seconds", lambda values: values.quantile(0.75)),
            artifact_bytes=("artifact_bytes", "max"),
            repetitions=("iteration", "count"),
        )
        .reset_index()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output_dir / "summary.parquet", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    return summary


def layout_runners() -> list[LayoutRunner]:
    """Return the backend and layout runners available in this environment."""
    return [
        LayoutRunner(
            backend="csv",
            layout="wide",
            compression="none",
            write=_write_csv_wide,
            read_all=_read_csv_wide_all,
            read_matrix=_read_csv_wide_matrix,
            read_rows=_read_csv_wide_rows,
            read_features=_read_csv_wide_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="csv",
            layout="delimited_array",
            compression="none",
            write=_write_csv_delimited_array,
            read_all=_read_csv_packed_all,
            read_matrix=_read_csv_delimited_matrix,
            read_rows=_read_csv_delimited_rows,
            read_features=_read_csv_delimited_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="csv",
            layout="json_array",
            compression="none",
            write=_write_csv_json_array,
            read_all=_read_csv_packed_all,
            read_matrix=_read_csv_json_matrix,
            read_rows=_read_csv_json_rows,
            read_features=_read_csv_json_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="arrow_ipc",
            layout="wide",
            compression="none",
            write=_write_arrow_wide,
            read_all=_read_arrow_all,
            read_matrix=_read_arrow_wide_matrix,
            read_rows=_read_arrow_wide_rows,
            read_features=_read_arrow_wide_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="arrow_ipc",
            layout="fixed_array",
            compression="none",
            write=_write_arrow_fixed_array,
            read_all=_read_arrow_all,
            read_matrix=_read_arrow_fixed_matrix,
            read_rows=_read_arrow_fixed_rows,
            read_features=_read_arrow_fixed_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="parquet",
            layout="wide",
            compression="snappy",
            write=_write_parquet_wide,
            read_all=_read_parquet_all,
            read_matrix=_read_parquet_wide_matrix,
            read_rows=_read_parquet_wide_rows,
            read_features=_read_parquet_wide_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="parquet",
            layout="fixed_array",
            compression="snappy",
            write=_write_parquet_fixed_array,
            read_all=_read_parquet_all,
            read_matrix=_read_parquet_fixed_matrix,
            read_rows=_read_parquet_fixed_rows,
            read_features=_read_parquet_fixed_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="duckdb",
            layout="wide",
            compression="duckdb_default",
            write=_write_duckdb_wide,
            read_all=_read_duckdb_wide_all,
            read_matrix=_read_duckdb_wide_matrix,
            read_rows=_read_duckdb_wide_rows,
            read_features=_read_duckdb_wide_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="duckdb",
            layout="duckdb_array",
            compression="duckdb_default",
            write=_write_duckdb_array,
            read_all=_read_duckdb_array_all,
            read_matrix=_read_duckdb_array_matrix,
            read_rows=_read_duckdb_array_rows,
            read_features=_read_duckdb_array_features,
            compute_norm=_compute_norm_from_matrix,
        ),
    ]


def write_environment(output_dir: Path, config: BenchmarkConfig) -> None:
    """Write environment metadata beside the benchmark outputs."""
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "artifact_dir": str(config.artifact_dir),
            "figure_dir": str(config.figure_dir),
        },
        "packages": {
            "duckdb": duckdb.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
        },
    }
    (output_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _measure_writes(
    *,
    config: BenchmarkConfig,
    runner: LayoutRunner,
    dataset: BenchmarkDataset,
    timestamp: str,
    git_commit: str,
    records: list[BenchmarkResult],
) -> Artifact:
    artifact = Artifact(Path(), 0)
    total_writes = config.warmups + config.measured_repetitions
    for iteration in range(total_writes):
        artifact_path = _artifact_path(config, runner, dataset, iteration)
        _prepare_artifact_path(artifact_path)
        started = time.perf_counter()
        artifact = runner.write(dataset, artifact_path)
        elapsed = time.perf_counter() - started
        _validate_artifact(runner, artifact, dataset)
        if iteration >= config.warmups:
            records.append(
                _result(
                    config=config,
                    runner=runner,
                    dataset=dataset,
                    operation="write",
                    operation_parameter="all",
                    iteration=iteration - config.warmups,
                    elapsed=elapsed,
                    artifact=artifact,
                    timestamp=timestamp,
                    git_commit=git_commit,
                    rows_per_second=dataset.rows / elapsed,
                    values_per_second=dataset.matrix.size / elapsed,
                )
            )
    return artifact


def _measure_reads(
    *,
    config: BenchmarkConfig,
    runner: LayoutRunner,
    dataset: BenchmarkDataset,
    artifact: Artifact,
    selected_rows: np.ndarray,
    selected_features: np.ndarray,
    timestamp: str,
    git_commit: str,
    records: list[BenchmarkResult],
) -> None:
    operations: list[tuple[str, str, Callable[[], Any], Callable[[Any], None]]] = [
        (
            "full_read",
            "all",
            lambda: runner.read_all(artifact, dataset),
            lambda _: None,
        ),
        (
            "matrix_materialization",
            "all",
            lambda: runner.read_matrix(artifact, dataset),
            lambda matrix: assert_same_matrix(matrix, dataset.matrix),
        ),
        (
            "random_rows",
            str(len(selected_rows)),
            lambda: runner.read_rows(artifact, dataset, selected_rows),
            lambda matrix: assert_rows(matrix, dataset.matrix, selected_rows),
        ),
        (
            "feature_projection",
            str(len(selected_features)),
            lambda: runner.read_features(artifact, dataset, selected_features),
            lambda matrix: assert_features(matrix, dataset.matrix, selected_features),
        ),
        (
            "vector_norm",
            "l2",
            lambda: runner.compute_norm(artifact, dataset),
            lambda vector: np.testing.assert_allclose(
                vector,
                np.linalg.norm(dataset.matrix, axis=1),
                rtol=1e-5,
                atol=1e-5,
            ),
        ),
    ]
    for operation, parameter, call, validate in operations:
        for _ in range(config.warmups):
            validate(call())
        for iteration in range(config.measured_repetitions):
            started = time.perf_counter()
            value = call()
            elapsed = time.perf_counter() - started
            validate(value)
            records.append(
                _result(
                    config=config,
                    runner=runner,
                    dataset=dataset,
                    operation=operation,
                    operation_parameter=parameter,
                    iteration=iteration,
                    elapsed=elapsed,
                    artifact=artifact,
                    timestamp=timestamp,
                    git_commit=git_commit,
                )
            )


def _result(
    *,
    config: BenchmarkConfig,
    runner: LayoutRunner,
    dataset: BenchmarkDataset,
    operation: str,
    operation_parameter: str,
    iteration: int,
    elapsed: float,
    artifact: Artifact,
    timestamp: str,
    git_commit: str,
    rows_per_second: float | None = None,
    values_per_second: float | None = None,
) -> BenchmarkResult:
    return BenchmarkResult(
        backend=runner.backend,
        layout=runner.layout,
        dataset=dataset.name,
        rows=dataset.rows,
        dimensions=dataset.dimensions,
        dtype=str(dataset.matrix.dtype),
        operation=operation,
        operation_parameter=operation_parameter,
        iteration=iteration,
        elapsed_seconds=elapsed,
        artifact_bytes=artifact.bytes,
        peak_memory_bytes=None,
        threads=config.threads,
        compression=runner.compression,
        timestamp=timestamp,
        git_commit=git_commit,
        rows_per_second=rows_per_second,
        values_per_second=values_per_second,
    )


def _selection(size: int, count: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(size, size=count, replace=False)).astype(np.int64)


def _artifact_path(
    config: BenchmarkConfig,
    runner: LayoutRunner,
    dataset: BenchmarkDataset,
    iteration: int,
) -> Path:
    name = (
        f"{dataset.name}_rows-{dataset.rows}_dims-{dataset.dimensions}_"
        f"{runner.backend}_{runner.layout}_iter-{iteration}"
    )
    suffix = {
        "csv": ".dir",
        "arrow_ipc": ".arrow",
        "parquet": ".parquet",
        "duckdb": ".duckdb",
    }[runner.backend]
    return config.artifact_dir / f"{name}{suffix}"


def _prepare_artifact_path(path: Path) -> None:
    if path.exists() and path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)


def _artifact_size(path: Path) -> int:
    if path.is_dir():
        return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    return path.stat().st_size


def _write_feature_names(directory: Path, feature_names: list[str]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "feature_names.json").write_text(
        json.dumps(feature_names, indent=2),
        encoding="utf-8",
    )


def _csv_path(artifact: Artifact) -> Path:
    return artifact.path / "data.csv"


def _write_csv_wide(dataset: BenchmarkDataset, path: Path) -> Artifact:
    path.mkdir(parents=True, exist_ok=True)
    wide_dataframe(dataset).to_csv(path / "data.csv", index=False)
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_csv_delimited_array(dataset: BenchmarkDataset, path: Path) -> Artifact:
    path.mkdir(parents=True, exist_ok=True)
    result = dataset.metadata.copy()
    result["features"] = [";".join(np.char.mod("%.9g", row)) for row in dataset.matrix]
    result.to_csv(path / "data.csv", index=False)
    _write_feature_names(path, dataset.feature_names)
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_csv_json_array(dataset: BenchmarkDataset, path: Path) -> Artifact:
    path.mkdir(parents=True, exist_ok=True)
    result = dataset.metadata.copy()
    result["features"] = [
        json.dumps([float(value) for value in row], separators=(",", ":"))
        for row in dataset.matrix
    ]
    result.to_csv(path / "data.csv", index=False)
    _write_feature_names(path, dataset.feature_names)
    return Artifact(path=path, bytes=_artifact_size(path))


def _read_csv_wide_all(artifact: Artifact, _: BenchmarkDataset) -> pd.DataFrame:
    return pd.read_csv(_csv_path(artifact))


def _read_csv_packed_all(artifact: Artifact, _: BenchmarkDataset) -> pd.DataFrame:
    return pd.read_csv(_csv_path(artifact))


def _read_csv_wide_matrix(artifact: Artifact, dataset: BenchmarkDataset) -> np.ndarray:
    table = pd.read_csv(_csv_path(artifact), usecols=pd.Index(dataset.feature_names))
    return table.to_numpy(dtype=np.float32)


def _read_csv_delimited_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    table = pd.read_csv(_csv_path(artifact), usecols=pd.Index(["features"]))
    values = [
        np.fromstring(item, sep=";", dtype=np.float32) for item in table["features"]
    ]
    matrix = np.vstack(values).astype(np.float32, copy=False)
    if matrix.shape[1] != dataset.dimensions:
        msg = f"expected {dataset.dimensions} delimited features"
        raise AssertionError(msg)
    return matrix


def _read_csv_json_matrix(artifact: Artifact, _: BenchmarkDataset) -> np.ndarray:
    table = pd.read_csv(_csv_path(artifact), usecols=pd.Index(["features"]))
    return np.array([json.loads(item) for item in table["features"]], dtype=np.float32)


def _read_csv_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_csv_wide_matrix(artifact, dataset)[rows, :]


def _read_csv_delimited_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_csv_delimited_matrix(artifact, dataset)[rows, :]


def _read_csv_json_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_csv_json_matrix(artifact, dataset)[rows, :]


def _read_csv_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    table = pd.read_csv(_csv_path(artifact), usecols=pd.Index(names))
    return table.to_numpy(dtype=np.float32)


def _read_csv_delimited_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_csv_delimited_matrix(artifact, dataset)[:, features]


def _read_csv_json_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_csv_json_matrix(artifact, dataset)[:, features]


def _arrow_wide_table(dataset: BenchmarkDataset) -> pa.Table:
    return pa.Table.from_pandas(wide_dataframe(dataset), preserve_index=False)


def _arrow_fixed_table(dataset: BenchmarkDataset) -> pa.Table:
    values = pa.array(dataset.matrix.reshape(-1), type=pa.float32())
    features = pa.FixedSizeListArray.from_arrays(values, dataset.dimensions)
    data = {
        "sample_id": pa.array(
            dataset.metadata["sample_id"].to_list(), type=pa.string()
        ),
        "plate_id": pa.array(dataset.metadata["plate_id"].to_list(), type=pa.string()),
        "well_id": pa.array(dataset.metadata["well_id"].to_list(), type=pa.string()),
        "features": features,
    }
    return pa.Table.from_pydict(data)


def _write_arrow_wide(dataset: BenchmarkDataset, path: Path) -> Artifact:
    with ipc.new_file(path, _arrow_wide_table(dataset).schema) as writer:
        writer.write_table(_arrow_wide_table(dataset))
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_arrow_fixed_array(dataset: BenchmarkDataset, path: Path) -> Artifact:
    table = _arrow_fixed_table(dataset)
    with ipc.new_file(path, table.schema) as writer:
        writer.write_table(table)
    _write_sidecar_for_file(path, dataset.feature_names)
    return Artifact(path=path, bytes=_artifact_size(path) + _sidecar_size(path))


def _read_arrow_all(artifact: Artifact, _: BenchmarkDataset) -> pa.Table:
    with ipc.open_file(artifact.path) as reader:
        return reader.read_all()


def _read_arrow_wide_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    table = _read_arrow_all(artifact, dataset).select(dataset.feature_names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_arrow_fixed_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    table = _read_arrow_all(artifact, dataset).select(["features"])
    return _fixed_array_to_matrix(table["features"], dataset.dimensions)


def _read_arrow_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_arrow_wide_matrix(artifact, dataset)[rows, :]


def _read_arrow_fixed_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_arrow_fixed_matrix(artifact, dataset)[rows, :]


def _read_arrow_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    table = _read_arrow_all(artifact, dataset).select(names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_arrow_fixed_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_arrow_fixed_matrix(artifact, dataset)[:, features]


def _write_parquet_wide(dataset: BenchmarkDataset, path: Path) -> Artifact:
    pq.write_table(_arrow_wide_table(dataset), path, compression="snappy")
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_parquet_fixed_array(dataset: BenchmarkDataset, path: Path) -> Artifact:
    pq.write_table(_arrow_fixed_table(dataset), path, compression="snappy")
    _write_sidecar_for_file(path, dataset.feature_names)
    return Artifact(path=path, bytes=_artifact_size(path) + _sidecar_size(path))


def _read_parquet_all(artifact: Artifact, _: BenchmarkDataset) -> pa.Table:
    return pq.read_table(artifact.path)


def _read_parquet_wide_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    table = pq.read_table(artifact.path, columns=dataset.feature_names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_parquet_fixed_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    table = pq.read_table(artifact.path, columns=["features"])
    return _fixed_array_to_matrix(table["features"], dataset.dimensions)


def _read_parquet_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_parquet_wide_matrix(artifact, dataset)[rows, :]


def _read_parquet_fixed_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_parquet_fixed_matrix(artifact, dataset)[rows, :]


def _read_parquet_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    table = pq.read_table(artifact.path, columns=names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_parquet_fixed_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_parquet_fixed_matrix(artifact, dataset)[:, features]


def _fixed_array_to_matrix(column: pa.ChunkedArray, dimensions: int) -> np.ndarray:
    array = column.combine_chunks()
    values = array.values.to_numpy(zero_copy_only=False)
    return values.reshape(len(array), dimensions).astype(np.float32, copy=False)


def _write_sidecar_for_file(path: Path, feature_names: list[str]) -> None:
    sidecar = _sidecar_path(path)
    sidecar.write_text(json.dumps(feature_names, indent=2), encoding="utf-8")


def _sidecar_path(path: Path) -> Path:
    return path.with_suffix(f"{path.suffix}.feature_names.json")


def _sidecar_size(path: Path) -> int:
    sidecar = _sidecar_path(path)
    if sidecar.exists():
        return sidecar.stat().st_size
    return 0


def _write_duckdb_wide(dataset: BenchmarkDataset, path: Path) -> Artifact:
    frame = wide_dataframe(dataset)
    with duckdb.connect(str(path)) as connection:
        connection.execute("SET threads = 1")
        connection.register("profiles", frame)
        connection.execute("CREATE TABLE wide AS SELECT * FROM profiles")
        connection.execute("CHECKPOINT")
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_duckdb_array(dataset: BenchmarkDataset, path: Path) -> Artifact:
    frame = array_dataframe(dataset)
    with duckdb.connect(str(path)) as connection:
        connection.execute("SET threads = 1")
        connection.register("profiles", frame)
        connection.execute("CREATE TABLE array_profiles AS SELECT * FROM profiles")
        connection.execute("CHECKPOINT")
    _write_sidecar_for_file(path, dataset.feature_names)
    return Artifact(path=path, bytes=_artifact_size(path) + _sidecar_size(path))


def _read_duckdb_wide_all(artifact: Artifact, _: BenchmarkDataset) -> pd.DataFrame:
    with duckdb.connect(str(artifact.path), read_only=True) as connection:
        return connection.execute("SELECT * FROM wide").fetchdf()


def _read_duckdb_array_all(artifact: Artifact, _: BenchmarkDataset) -> pd.DataFrame:
    with duckdb.connect(str(artifact.path), read_only=True) as connection:
        return connection.execute("SELECT * FROM array_profiles").fetchdf()


def _read_duckdb_wide_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    columns = ", ".join(f'"{name}"' for name in dataset.feature_names)
    with duckdb.connect(str(artifact.path), read_only=True) as connection:
        frame = connection.execute(f"SELECT {columns} FROM wide").fetchdf()
    return frame.to_numpy(dtype=np.float32)


def _read_duckdb_array_matrix(artifact: Artifact, _: BenchmarkDataset) -> np.ndarray:
    with duckdb.connect(str(artifact.path), read_only=True) as connection:
        frame = connection.execute("SELECT features FROM array_profiles").fetchdf()
    return np.vstack(frame["features"].to_numpy()).astype(np.float32, copy=False)


def _read_duckdb_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_duckdb_wide_matrix(artifact, dataset)[rows, :]


def _read_duckdb_array_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    return _read_duckdb_array_matrix(artifact, dataset)[rows, :]


def _read_duckdb_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    columns = ", ".join(f'"{name}"' for name in names)
    with duckdb.connect(str(artifact.path), read_only=True) as connection:
        frame = connection.execute(f"SELECT {columns} FROM wide").fetchdf()
    return frame.to_numpy(dtype=np.float32)


def _read_duckdb_array_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    expressions = ", ".join(
        f"list_extract(features, {index + 1}) AS feature_{position}"
        for position, index in enumerate(features)
    )
    with duckdb.connect(str(artifact.path), read_only=True) as connection:
        frame = connection.execute(
            f"SELECT {expressions} FROM array_profiles"
        ).fetchdf()
    return frame.to_numpy(dtype=np.float32)


def _compute_norm_from_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    runner = next(
        item
        for item in layout_runners()
        if item.backend in artifact.path.name and item.layout in artifact.path.name
    )
    matrix = runner.read_matrix(artifact, dataset)
    return np.linalg.norm(matrix, axis=1)


def _validate_artifact(
    runner: LayoutRunner,
    artifact: Artifact,
    dataset: BenchmarkDataset,
) -> None:
    matrix = runner.read_matrix(artifact, dataset)
    assert_same_matrix(matrix, dataset.matrix)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def set_thread_environment(threads: int) -> None:
    """Set common thread-count variables before backend imports do work."""
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(threads)


def read_raw_results(path: Path = Path("results/raw_results.parquet")) -> pd.DataFrame:
    """Read raw benchmark results."""
    return pd.read_parquet(path)
