"""Benchmark wide tables against fixed-size and packed array layouts."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import json
import math
import os
import platform
import shutil
import subprocess
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import duckdb
import lance
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import tiledb
import vortex as vx
import zarr
from numcodecs import Blosc, blosc

from array_we_there_yet.data import (
    BenchmarkDataset,
    array_dataframe,
    make_synthetic_dataset,
    wide_dataframe,
)
from array_we_there_yet.encodings import describe_encoding
from array_we_there_yet.validation import (
    assert_features,
    assert_mixed_retrieval,
    assert_rows,
    assert_same_matrix,
)

Layout = Literal[
    "wide",
    "fixed_array",
    "delimited_array",
    "json_array",
    "duckdb_array",
    "zarr_matrix",
    "tiledb_dense",
    "npy",
]
METADATA_COLUMNS = ("sample_id", "plate_id", "well_id")


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
    csv_max_rows: int | None = None


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
    cpu_seconds: float
    artifact_bytes: int
    peak_memory_bytes: int | None
    threads: int
    compression: str
    profile: str
    timestamp: str
    git_commit: str
    rows_per_second: float | None = None
    values_per_second: float | None = None
    calls: int = 1


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
    profile: str = "default"


BYTES_PER_TEXT_VALUE = 12
DISK_HEADROOM = 1.5


def required_disk_bytes(config: BenchmarkConfig) -> int:
    """Estimate the disk space that one layout needs at the widest feature count.

    The largest artifacts are CSV text at about 12 bytes per value. The benchmark
    keeps every write of one layout until it finishes measuring it.
    """
    writes = config.warmups + config.measured_repetitions
    values = config.rows * max(config.dimensions)
    return int(values * BYTES_PER_TEXT_VALUE * writes * DISK_HEADROOM)


def _check_disk_space(config: BenchmarkConfig) -> None:
    """Stop early when the disk cannot hold the artifacts of one layout."""
    required = required_disk_bytes(config)
    free = shutil.disk_usage(config.artifact_dir).free
    if free < required:
        message = (
            f"The benchmark needs about {required / 1e9:.1f} GB of free disk space "
            f"but only {free / 1e9:.1f} GB is free in {config.artifact_dir}. "
            "Free some space or use fewer rows or feature counts."
        )
        raise RuntimeError(message)


def _remove_artifacts(
    config: BenchmarkConfig,
    runner: LayoutRunner,
    dataset: BenchmarkDataset,
) -> None:
    """Delete every artifact that one layout wrote for one feature count."""
    for iteration in range(config.warmups + config.measured_repetitions):
        path = _artifact_path(config, runner, dataset, iteration)
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
        _sidecar_path(path).unlink(missing_ok=True)


def _write_npy(dataset: BenchmarkDataset, path: Path) -> Artifact:
    np.save(path, dataset.matrix)
    return Artifact(path=path, bytes=_artifact_size(path))


def _read_npy(artifact: Artifact, _: BenchmarkDataset) -> np.ndarray:
    return np.load(artifact.path)


def _read_npy_rows(
    artifact: Artifact, _: BenchmarkDataset, rows: np.ndarray
) -> np.ndarray:
    return np.asarray(np.load(artifact.path, mmap_mode="r")[rows])


def _read_npy_features(
    artifact: Artifact, _: BenchmarkDataset, features: np.ndarray
) -> np.ndarray:
    return np.array(np.load(artifact.path, mmap_mode="r")[:, features])


def _npy_norm(artifact: Artifact, dataset: BenchmarkDataset) -> np.ndarray:
    return np.linalg.norm(_read_npy(artifact, dataset), axis=1)


def numpy_floor_runner() -> LayoutRunner:
    """Return the plain NumPy file used as a speed-of-light baseline.

    It is not part of `layout_runners`, so it does not appear in the layout
    comparisons. It shows how far each format is from a plain memory copy.
    """
    return LayoutRunner(
        backend="numpy",
        layout="npy",
        compression="none",
        write=_write_npy,
        read_all=_read_npy,
        read_matrix=_read_npy,
        read_rows=_read_npy_rows,
        read_features=_read_npy_features,
        compute_norm=_npy_norm,
    )


def measure_numpy_floor(
    *,
    config: BenchmarkConfig,
    dataset: BenchmarkDataset,
    selected_rows: np.ndarray,
    selected_features: np.ndarray,
    timestamp: str,
    git_commit: str,
) -> list[BenchmarkResult]:
    """Time a plain `.npy` file for one feature count and remove it."""
    runner = numpy_floor_runner()
    records: list[BenchmarkResult] = []
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
        selected_rows=np.asarray(selected_rows),
        selected_features=np.asarray(selected_features),
        timestamp=timestamp,
        git_commit=git_commit,
        records=records,
        include_mixed=False,
    )
    _remove_artifacts(config, runner, dataset)
    return records


def _write_raw_results(
    config: BenchmarkConfig,
    records: list[BenchmarkResult],
    encodings: list[dict[str, object]],
    floor: list[BenchmarkResult] | None = None,
) -> pd.DataFrame:
    """Write the results so far, so that a crash keeps finished feature counts."""
    raw = pd.DataFrame(asdict(record) for record in records)
    raw.to_parquet(config.output_dir / "raw_results.parquet", index=False)
    pd.DataFrame(encodings).to_parquet(
        config.output_dir / "encodings.parquet", index=False
    )
    if floor:
        pd.DataFrame(asdict(record) for record in floor).to_parquet(
            config.output_dir / "floor_results.parquet", index=False
        )
    return raw


def run_benchmarks(config: BenchmarkConfig) -> pd.DataFrame:
    """Run the configured benchmark and write raw outputs."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    thread_limits = apply_thread_limits(config.threads)
    if config.artifact_dir.exists():
        shutil.rmtree(config.artifact_dir)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    _check_disk_space(config)

    records: list[BenchmarkResult] = []
    floor: list[BenchmarkResult] = []
    encodings: list[dict[str, object]] = []
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
            if _skipped(config, runner):
                continue
            artifact = _measure_writes(
                config=config,
                runner=runner,
                dataset=dataset,
                timestamp=timestamp,
                git_commit=git_commit,
                records=records,
            )
            encodings.append(
                {
                    "backend": runner.backend,
                    "layout": runner.layout,
                    "profile": runner.profile,
                    "dimensions": dataset.dimensions,
                    "encoding": describe_encoding(
                        runner.backend, runner.layout, artifact.path
                    ),
                }
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
            _remove_artifacts(config, runner, dataset)

        floor.extend(
            measure_numpy_floor(
                config=config,
                dataset=dataset,
                selected_rows=selected_rows,
                selected_features=selected_features,
                timestamp=timestamp,
                git_commit=git_commit,
            )
        )
        _write_raw_results(config, records, encodings, floor)

    raw = _write_raw_results(config, records, encodings, floor)
    write_environment(config.output_dir, config, thread_limits)
    return raw


def combine_runs(
    input_dirs: list[Path],
    output_dir: Path,
    *,
    allow_dirty: bool = False,
) -> pd.DataFrame:
    """Pool the raw results of several runs of the same code.

    Every raw row is kept and gets a `run` number, so the summary of the pooled
    results shows the variation between runs and not only within one run.
    """
    if len(input_dirs) < 2:  # noqa: PLR2004
        message = "Pooling needs at least two runs."
        raise ValueError(message)
    raws = [pd.read_parquet(path / "raw_results.parquet") for path in input_dirs]
    commits = {str(commit) for raw in raws for commit in raw["git_commit"].unique()}
    if len(commits) > 1:
        message = f"The runs come from different code versions: {sorted(commits)}."
        raise ValueError(message)
    if any(commit.endswith("-dirty") for commit in commits) and not allow_dirty:
        message = (
            f"The runs used uncommitted code ({sorted(commits)}). Commit the code "
            "and run again, or pass allow_dirty."
        )
        raise ValueError(message)

    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [raw.assign(run=number) for number, raw in enumerate(raws, start=1)],
        ignore_index=True,
    )
    combined.to_parquet(output_dir / "raw_results.parquet", index=False)
    floors = [
        pd.read_parquet(path / "floor_results.parquet").assign(run=number)
        for number, path in enumerate(input_dirs, start=1)
        if (path / "floor_results.parquet").exists()
    ]
    if floors:
        pd.concat(floors, ignore_index=True).to_parquet(
            output_dir / "floor_results.parquet", index=False
        )
    shutil.copy(input_dirs[0] / "encodings.parquet", output_dir / "encodings.parquet")
    environment = json.loads(
        (input_dirs[0] / "environment.json").read_text(encoding="utf-8")
    )
    environment["runs"] = len(input_dirs)
    environment["git_commit"] = commits.pop()
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )
    return combined


def summarize_results(
    raw: pd.DataFrame,
    output_dir: Path = Path("results"),
    file_name: str = "summary.parquet",
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
        "profile",
    ]
    raw = raw.assign(parallelism=raw["cpu_seconds"] / raw["elapsed_seconds"])
    summary = (
        raw.groupby(group_columns, dropna=False)
        .agg(
            median_seconds=("elapsed_seconds", "median"),
            q25_seconds=("elapsed_seconds", lambda values: values.quantile(0.25)),
            q75_seconds=("elapsed_seconds", lambda values: values.quantile(0.75)),
            median_cpu_seconds=("cpu_seconds", "median"),
            median_parallelism=("parallelism", "median"),
            artifact_bytes=("artifact_bytes", "max"),
            repetitions=("iteration", "count"),
        )
        .reset_index()
    )
    summary["noisy"] = (summary["q75_seconds"] - summary["q25_seconds"]) > (
        NOISY_RANGE * summary["median_seconds"]
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_parquet(output_dir / file_name, index=False)
    return summary


def layout_runners() -> list[LayoutRunner]:
    """Return the default runners and the compact runners for formats with a knob."""
    defaults = _default_runners()
    return [*defaults, *_compact_runners(defaults)]


def _compact_runners(defaults: list[LayoutRunner]) -> list[LayoutRunner]:
    """Return runners that write the same layouts with smaller-file settings.

    DuckDB, Lance, and Vortex are left out. Their writers expose no setting that
    the benchmark could verify to change the stored size.
    """
    writers = {
        ("csv", "wide"): (functools.partial(_write_csv_wide, compress=True), "gzip"),
        ("parquet", "wide"): (
            functools.partial(_write_parquet_wide, profile="compact"),
            "zstd",
        ),
        ("parquet", "fixed_array"): (
            functools.partial(_write_parquet_fixed_array, profile="compact"),
            "zstd",
        ),
        ("zarr", "wide"): (
            functools.partial(_write_zarr_wide, compact=True),
            "blosc_zstd",
        ),
        ("zarr", "zarr_matrix"): (
            functools.partial(_write_zarr_matrix, compact=True),
            "blosc_zstd",
        ),
        ("tiledb", "wide"): (
            functools.partial(_write_tiledb_wide, compact=True),
            "shuffle_zstd",
        ),
        ("tiledb", "tiledb_dense"): (
            functools.partial(_write_tiledb_dense, compact=True),
            "shuffle_zstd",
        ),
    }
    return [
        dataclasses.replace(
            runner,
            write=writers[(runner.backend, runner.layout)][0],
            compression=writers[(runner.backend, runner.layout)][1],
            profile="compact",
        )
        for runner in defaults
        if (runner.backend, runner.layout) in writers
    ]


def _default_runners() -> list[LayoutRunner]:
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
        LayoutRunner(
            backend="zarr",
            layout="wide",
            compression="zarr_default",
            write=_write_zarr_wide,
            read_all=_read_zarr_wide_all,
            read_matrix=_read_zarr_wide_matrix,
            read_rows=_read_zarr_wide_rows,
            read_features=_read_zarr_wide_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="zarr",
            layout="zarr_matrix",
            compression="zarr_default",
            write=_write_zarr_matrix,
            read_all=_read_zarr_matrix_all,
            read_matrix=_read_zarr_matrix,
            read_rows=_read_zarr_matrix_rows,
            read_features=_read_zarr_matrix_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="tiledb",
            layout="wide",
            compression="tiledb_default",
            write=_write_tiledb_wide,
            read_all=_read_tiledb_wide_all,
            read_matrix=_read_tiledb_wide_matrix,
            read_rows=_read_tiledb_wide_rows,
            read_features=_read_tiledb_wide_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="tiledb",
            layout="tiledb_dense",
            compression="tiledb_default",
            write=_write_tiledb_dense,
            read_all=_read_tiledb_dense_all,
            read_matrix=_read_tiledb_dense_matrix,
            read_rows=_read_tiledb_dense_rows,
            read_features=_read_tiledb_dense_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="vortex",
            layout="wide",
            compression="none",
            write=_write_vortex_wide,
            read_all=_read_vortex_all,
            read_matrix=_read_vortex_wide_matrix,
            read_rows=_read_vortex_wide_rows,
            read_features=_read_vortex_wide_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="vortex",
            layout="fixed_array",
            compression="none",
            write=_write_vortex_fixed_array,
            read_all=_read_vortex_all,
            read_matrix=_read_vortex_fixed_matrix,
            read_rows=_read_vortex_fixed_rows,
            read_features=_read_vortex_fixed_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="lance",
            layout="wide",
            compression="lance_default",
            write=_write_lance_wide,
            read_all=_read_lance_all,
            read_matrix=_read_lance_wide_matrix,
            read_rows=_read_lance_wide_rows,
            read_features=_read_lance_wide_features,
            compute_norm=_compute_norm_from_matrix,
        ),
        LayoutRunner(
            backend="lance",
            layout="fixed_array",
            compression="lance_default",
            write=_write_lance_fixed_array,
            read_all=_read_lance_all,
            read_matrix=_read_lance_fixed_matrix,
            read_rows=_read_lance_fixed_rows,
            read_features=_read_lance_fixed_features,
            compute_norm=_compute_norm_from_matrix,
        ),
    ]


def write_environment(
    output_dir: Path,
    config: BenchmarkConfig,
    thread_limits: dict[str, int | str] | None = None,
) -> None:
    """Write environment metadata beside the benchmark outputs."""
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": _git_commit(),
        "processor": platform.processor(),
        "hardware": hardware_info(),
        "thread_limits": thread_limits or {},
        "config": {
            **asdict(config),
            "output_dir": str(config.output_dir),
            "artifact_dir": str(config.artifact_dir),
        },
        "packages": {
            "duckdb": duckdb.__version__,
            "lance": lance.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "pyarrow": pa.__version__,
            "tiledb": tiledb.version(),
            "vortex": vx.__version__,
            "zarr": zarr.__version__,
        },
    }
    (output_dir / "environment.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _skipped(config: BenchmarkConfig, runner: LayoutRunner) -> bool:
    """Return whether a layout is left out because the dataset has too many rows."""
    return (
        runner.backend == "csv"
        and config.csv_max_rows is not None
        and config.rows > config.csv_max_rows
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
        artifact, elapsed, cpu = _timed(
            lambda path=artifact_path: runner.write(dataset, path)
        )
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
                    cpu=cpu,
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
    include_mixed: bool = True,
    only: frozenset[str] | None = None,
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
            "mixed_retrieval",
            f"{len(selected_rows)}_rows_{len(selected_features)}_features",
            lambda: _read_mixed(
                runner,
                artifact,
                dataset,
                selected_rows,
                selected_features,
            ),
            lambda frame: assert_mixed_retrieval(
                frame,
                dataset=dataset,
                rows=selected_rows,
                features=selected_features,
            ),
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
    if not include_mixed:
        operations = [item for item in operations if item[0] != "mixed_retrieval"]
    if only is not None:
        operations = [item for item in operations if item[0] in only]
    for operation, parameter, call, validate in operations:
        # At least one untimed call gives the size of a single call.
        warmup_seconds = 0.0
        for _ in range(max(config.warmups, 1)):
            value, warmup_seconds, _cpu = _timed(call)
            validate(value)
        calls = _calls_per_sample(warmup_seconds)
        for iteration in range(config.measured_repetitions):
            value, elapsed, cpu = _timed_repeated(call, calls)
            validate(value)
            records.append(
                _result(
                    config=config,
                    runner=runner,
                    dataset=dataset,
                    operation=operation,
                    operation_parameter=parameter,
                    iteration=iteration,
                    elapsed=elapsed / calls,
                    cpu=cpu / calls,
                    artifact=artifact,
                    timestamp=timestamp,
                    git_commit=git_commit,
                    calls=calls,
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
    cpu: float,
    artifact: Artifact,
    timestamp: str,
    git_commit: str,
    rows_per_second: float | None = None,
    values_per_second: float | None = None,
    calls: int = 1,
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
        cpu_seconds=cpu,
        artifact_bytes=artifact.bytes,
        peak_memory_bytes=None,
        threads=config.threads,
        compression=runner.compression,
        profile=runner.profile,
        timestamp=timestamp,
        git_commit=git_commit,
        rows_per_second=rows_per_second,
        values_per_second=values_per_second,
        calls=calls,
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
        f"{runner.backend}_{runner.layout}{_profile_suffix(runner)}_iter-{iteration}"
    )
    suffix = {
        "csv": ".dir",
        "parquet": ".parquet",
        "duckdb": ".duckdb",
        "vortex": ".vortex",
        "lance": ".lance",
        "zarr": ".zarr",
        "tiledb": ".tiledb",
        "numpy": ".npy",
    }[runner.backend]
    return config.artifact_dir / f"{name}{suffix}"


def _profile_suffix(runner: LayoutRunner) -> str:
    return "" if runner.profile == "default" else f"_{runner.profile}"


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
    compressed = artifact.path / "data.csv.gz"
    return compressed if compressed.exists() else artifact.path / "data.csv"


def _write_csv_wide(
    dataset: BenchmarkDataset,
    path: Path,
    *,
    compress: bool = False,
) -> Artifact:
    path.mkdir(parents=True, exist_ok=True)
    file_name = "data.csv.gz" if compress else "data.csv"
    wide_dataframe(dataset).to_csv(path / file_name, index=False)
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
        "[" + ",".join(np.char.mod("%.9g", row)) + "]" for row in dataset.matrix
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


def _read_csv_wide_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    names = _mixed_columns(dataset, features)
    table = pd.read_csv(_csv_path(artifact), usecols=pd.Index(names))
    return table.loc[:, names].iloc[rows].reset_index(drop=True)


def _read_csv_delimited_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_csv_delimited_matrix(artifact, dataset)[:, features]


def _read_csv_delimited_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    table = pd.read_csv(
        _csv_path(artifact),
        usecols=pd.Index([*METADATA_COLUMNS, "features"]),
    ).iloc[rows]
    values = [
        np.fromstring(item, sep=";", dtype=np.float32)[features]
        for item in table["features"]
    ]
    return _mixed_frame(
        metadata=table[list(METADATA_COLUMNS)],
        matrix=np.vstack(values).astype(np.float32, copy=False),
        dataset=dataset,
        features=features,
    )


def _read_csv_json_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_csv_json_matrix(artifact, dataset)[:, features]


def _read_csv_json_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    table = pd.read_csv(
        _csv_path(artifact),
        usecols=pd.Index([*METADATA_COLUMNS, "features"]),
    ).iloc[rows]
    matrix = np.asarray(
        [[json.loads(item)[index] for index in features] for item in table["features"]],
        dtype=np.float32,
    )
    return _mixed_frame(
        metadata=table[list(METADATA_COLUMNS)],
        matrix=matrix,
        dataset=dataset,
        features=features,
    )


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


def _write_vortex_wide(dataset: BenchmarkDataset, path: Path) -> Artifact:
    vx.io.write(_arrow_wide_table(dataset), str(path))
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_vortex_fixed_array(dataset: BenchmarkDataset, path: Path) -> Artifact:
    vx.io.write(_arrow_fixed_table(dataset), str(path))
    _write_sidecar_for_file(path, dataset.feature_names)
    return Artifact(path=path, bytes=_artifact_size(path) + _sidecar_size(path))


def _vortex_scan(artifact: Artifact, rows: np.ndarray) -> pa.Table:
    """Read the requested rows in file order. Vortex needs sorted row indices."""
    indices = vx.array(pa.array(np.sort(rows), type=pa.uint64()))
    table = vx.open(str(artifact.path)).scan(indices=indices).read_all()
    return table.to_arrow_table()


def _vortex_take(
    artifact: Artifact,
    rows: np.ndarray,
    columns: list[str],
) -> pa.Table:
    """Read the requested rows of some columns, in the requested order."""
    selected = _vortex_scan(artifact, rows).select(columns)
    return selected.take(pa.array(_ranks(rows)))


def _read_vortex_all(artifact: Artifact, _: BenchmarkDataset) -> pa.Table:
    return vx.open(str(artifact.path)).to_arrow().read_all()


def _read_vortex_wide_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    table = _read_vortex_all(artifact, dataset).select(dataset.feature_names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_vortex_fixed_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    table = _read_vortex_all(artifact, dataset).select(["features"])
    return _fixed_array_to_matrix(table["features"], dataset.dimensions)


def _read_vortex_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    table = _vortex_take(artifact, rows, dataset.feature_names)
    return _table_to_matrix(table)


def _read_vortex_fixed_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    table = _vortex_take(artifact, rows, ["features"])
    return _fixed_array_to_matrix(table["features"], dataset.dimensions)


def _read_vortex_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    table = _read_vortex_all(artifact, dataset).select(names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_vortex_wide_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    table = _vortex_scan(artifact, rows).select(_mixed_columns(dataset, features))
    return table.to_pandas().iloc[_ranks(rows)].reset_index(drop=True)


def _read_vortex_fixed_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_vortex_fixed_matrix(artifact, dataset)[:, features]


def _read_vortex_fixed_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    table = _vortex_scan(artifact, rows)
    order = _ranks(rows)
    metadata = table.select(METADATA_COLUMNS).to_pandas().iloc[order]
    matrix = _fixed_array_to_matrix(table["features"], dataset.dimensions)[order, :][
        :, features
    ]
    return _mixed_frame(
        metadata=metadata,
        matrix=matrix,
        dataset=dataset,
        features=features,
    )


def _write_lance_wide(dataset: BenchmarkDataset, path: Path) -> Artifact:
    lance.write_dataset(_arrow_wide_table(dataset), path)
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_lance_fixed_array(dataset: BenchmarkDataset, path: Path) -> Artifact:
    lance.write_dataset(_arrow_fixed_table(dataset), path)
    _write_sidecar_for_file(path, dataset.feature_names)
    return Artifact(path=path, bytes=_artifact_size(path) + _sidecar_size(path))


def _read_lance_all(artifact: Artifact, _: BenchmarkDataset) -> pa.Table:
    return lance.dataset(artifact.path).to_table()


def _read_lance_wide_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    # Naming every column makes Lance plan each one: about 1.35 ms per column.
    # Reading all columns and then selecting returns the same values much faster.
    table = lance.dataset(artifact.path).to_table().select(dataset.feature_names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_lance_fixed_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    table = lance.dataset(artifact.path).to_table(columns=["features"])
    return _fixed_array_to_matrix(table["features"], dataset.dimensions)


def _read_lance_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    table = lance.dataset(artifact.path).take(rows).select(dataset.feature_names)
    return _table_to_matrix(table)


def _read_lance_fixed_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    table = lance.dataset(artifact.path).take(rows, columns=["features"])
    return _fixed_array_to_matrix(table["features"], dataset.dimensions)


def _read_lance_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    table = lance.dataset(artifact.path).to_table(columns=names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_lance_wide_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    table = lance.dataset(artifact.path).take(
        rows, columns=_mixed_columns(dataset, features)
    )
    return table.to_pandas().reset_index(drop=True)


def _read_lance_fixed_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_lance_fixed_matrix(artifact, dataset)[:, features]


def _read_lance_fixed_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    table = lance.dataset(artifact.path).take(
        rows, columns=[*METADATA_COLUMNS, "features"]
    )
    metadata = table.select(METADATA_COLUMNS).to_pandas()
    matrix = _fixed_array_to_matrix(table["features"], dataset.dimensions)[:, features]
    return _mixed_frame(
        metadata=metadata,
        matrix=matrix,
        dataset=dataset,
        features=features,
    )


def _parquet_options(profile: str, float_columns: list[str]) -> dict[str, Any]:
    """Return `write_table` options for the default and compact profiles."""
    if profile == "compact":
        return {
            "compression": "zstd",
            "use_dictionary": False,
            "use_byte_stream_split": float_columns,
        }
    return {"compression": "snappy"}


def _write_parquet_wide(
    dataset: BenchmarkDataset,
    path: Path,
    *,
    profile: str = "default",
) -> Artifact:
    options = _parquet_options(profile, dataset.feature_names)
    pq.write_table(_arrow_wide_table(dataset), path, **options)
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_parquet_fixed_array(
    dataset: BenchmarkDataset,
    path: Path,
    *,
    profile: str = "default",
) -> Artifact:
    options = _parquet_options(profile, ["features.list.element"])
    pq.write_table(_arrow_fixed_table(dataset), path, **options)
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
    table = pads.dataset(artifact.path).take(pa.array(rows))
    return _table_to_matrix(table.select(dataset.feature_names))


def _read_parquet_fixed_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    table = pads.dataset(artifact.path).take(pa.array(rows), columns=["features"])
    return _fixed_array_to_matrix(table["features"], dataset.dimensions)


def _read_parquet_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    table = pq.read_table(artifact.path, columns=names)
    return table.to_pandas().to_numpy(dtype=np.float32)


def _read_parquet_wide_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    table = pads.dataset(artifact.path).take(
        pa.array(rows), columns=_mixed_columns(dataset, features)
    )
    return table.to_pandas().reset_index(drop=True)


def _read_parquet_fixed_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    return _read_parquet_fixed_matrix(artifact, dataset)[:, features]


def _read_parquet_fixed_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    table = pads.dataset(artifact.path).take(
        pa.array(rows), columns=[*METADATA_COLUMNS, "features"]
    )
    metadata = table.select(METADATA_COLUMNS).to_pandas()
    matrix = _fixed_array_to_matrix(table["features"], dataset.dimensions)[:, features]
    return _mixed_frame(
        metadata=metadata,
        matrix=matrix,
        dataset=dataset,
        features=features,
    )


def _table_to_matrix(table: pa.Table) -> np.ndarray:
    """Copy each column of an Arrow table into one float32 matrix."""
    matrix = np.empty((table.num_rows, table.num_columns), dtype=np.float32)
    for position, column in enumerate(table.columns):
        matrix[:, position] = column.to_numpy()
    return matrix


def _ranks(rows: np.ndarray) -> np.ndarray:
    """Return the position of each requested row in sorted order.

    Vortex reads sorted row indices. Taking these ranks from the sorted result
    restores the requested order.
    """
    return np.argsort(np.argsort(rows))


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
    with _duckdb_connect(path, read_only=False) as connection:
        connection.register("profiles", frame)
        connection.execute("CREATE TABLE wide AS SELECT * FROM profiles")
        connection.execute("CHECKPOINT")
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_duckdb_array(dataset: BenchmarkDataset, path: Path) -> Artifact:
    frame = array_dataframe(dataset)
    with _duckdb_connect(path, read_only=False) as connection:
        connection.register("profiles", frame)
        connection.execute(
            "CREATE TABLE array_profiles AS SELECT "
            f"{', '.join(METADATA_COLUMNS)}, "
            f"CAST(features AS FLOAT[{dataset.dimensions}]) AS features "
            "FROM profiles"
        )
        connection.execute("CHECKPOINT")
    _write_sidecar_for_file(path, dataset.feature_names)
    return Artifact(path=path, bytes=_artifact_size(path) + _sidecar_size(path))


def _duckdb_take(
    artifact: Artifact,
    table: str,
    rows: np.ndarray,
    columns: str = "*",
) -> pd.DataFrame:
    """Read the requested rows by row id, in the requested order."""
    wanted = ", ".join(str(int(row)) for row in rows)
    with _duckdb_connect(artifact.path, read_only=True) as connection:
        frame = connection.execute(
            f"SELECT rowid AS profile_row, {columns} FROM {table} "
            f"WHERE rowid IN ({wanted})"
        ).fetchdf()
    position = {int(row): index for index, row in enumerate(rows)}
    order = frame["profile_row"].map(position).to_numpy().argsort()
    return frame.iloc[order].reset_index(drop=True)


def _read_duckdb_wide_all(artifact: Artifact, _: BenchmarkDataset) -> pd.DataFrame:
    with _duckdb_connect(artifact.path, read_only=True) as connection:
        return connection.execute("SELECT * FROM wide").fetchdf()


def _read_duckdb_array_all(artifact: Artifact, _: BenchmarkDataset) -> pd.DataFrame:
    with _duckdb_connect(artifact.path, read_only=True) as connection:
        return connection.execute("SELECT * FROM array_profiles").fetchdf()


def _read_duckdb_wide_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    columns = ", ".join(f'"{name}"' for name in dataset.feature_names)
    with _duckdb_connect(artifact.path, read_only=True) as connection:
        frame = connection.execute(f"SELECT {columns} FROM wide").fetchdf()
    return frame.to_numpy(dtype=np.float32)


def _read_duckdb_array_matrix(artifact: Artifact, _: BenchmarkDataset) -> np.ndarray:
    with _duckdb_connect(artifact.path, read_only=True) as connection:
        frame = connection.execute("SELECT features FROM array_profiles").fetchdf()
    return np.vstack(frame["features"].to_numpy()).astype(np.float32, copy=False)


def _read_duckdb_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    frame = _duckdb_take(artifact, "wide", rows)
    return frame[dataset.feature_names].to_numpy(dtype=np.float32)


def _read_duckdb_array_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    frame = _duckdb_take(artifact, "array_profiles", rows)
    return np.vstack(frame["features"].to_numpy()).astype(np.float32, copy=False)


def _read_duckdb_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    columns = ", ".join(f'"{name}"' for name in names)
    with _duckdb_connect(artifact.path, read_only=True) as connection:
        frame = connection.execute(f"SELECT {columns} FROM wide").fetchdf()
    return frame.to_numpy(dtype=np.float32)


def _read_duckdb_wide_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    columns = ", ".join(f'"{name}"' for name in _mixed_columns(dataset, features))
    frame = _duckdb_take(artifact, "wide", rows, columns)
    return frame.drop(columns="profile_row")


def _read_duckdb_array_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    expressions = ", ".join(
        f"list_extract(features, {index + 1}) AS feature_{position}"
        for position, index in enumerate(features)
    )
    with _duckdb_connect(artifact.path, read_only=True) as connection:
        frame = connection.execute(
            f"SELECT {expressions} FROM array_profiles"
        ).fetchdf()
    return frame.to_numpy(dtype=np.float32)


def _read_duckdb_array_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    columns = ", ".join([*METADATA_COLUMNS, "features"])
    selected = _duckdb_take(artifact, "array_profiles", rows, columns)
    matrix = np.vstack(selected["features"].to_numpy()).astype(np.float32, copy=False)[
        :, features
    ]
    return _mixed_frame(
        metadata=selected[list(METADATA_COLUMNS)],
        matrix=matrix,
        dataset=dataset,
        features=features,
    )


def _zarr_compressor(*, compact: bool) -> dict[str, Any]:
    """Return `create_dataset` options: the library default or Blosc with zstd."""
    if not compact:
        return {}
    return {"compressor": Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)}


def _write_zarr_wide(
    dataset: BenchmarkDataset,
    path: Path,
    *,
    compact: bool = False,
) -> Artifact:
    root = zarr.open_group(str(path), mode="w")
    _write_zarr_metadata(root, dataset, layout="wide")
    feature_group = root.create_group("features")
    chunks = (min(dataset.rows, 1024),)
    for index, name in enumerate(dataset.feature_names):
        feature_group.create_dataset(
            name,
            data=dataset.matrix[:, index],
            chunks=chunks,
            dtype="f4",
            **_zarr_compressor(compact=compact),
        )
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_zarr_matrix(
    dataset: BenchmarkDataset,
    path: Path,
    *,
    compact: bool = False,
) -> Artifact:
    root = zarr.open_group(str(path), mode="w")
    _write_zarr_metadata(root, dataset, layout="zarr_matrix")
    root.create_dataset(
        "features",
        data=dataset.matrix,
        chunks=(min(dataset.rows, 1024), min(dataset.dimensions, 1024)),
        dtype="f4",
        **_zarr_compressor(compact=compact),
    )
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_zarr_metadata(
    root: zarr.hierarchy.Group,
    dataset: BenchmarkDataset,
    *,
    layout: str,
) -> None:
    root.attrs["layout"] = layout
    root.attrs["feature_names"] = dataset.feature_names
    root.create_dataset("feature_names", data=_string_values(dataset.feature_names))
    for column in ["sample_id", "plate_id", "well_id"]:
        root.create_dataset(column, data=_string_values(dataset.metadata[column]))


def _string_values(values: Iterable[object]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, *(len(value) for value in strings))
    return np.asarray(strings, dtype=f"U{width}")


def _read_zarr_wide_all(artifact: Artifact, dataset: BenchmarkDataset) -> pd.DataFrame:
    root = zarr.open_group(str(artifact.path), mode="r")
    data = _zarr_metadata_frame(root).to_dict(orient="list")
    feature_group = root["features"]
    for name in dataset.feature_names:
        data[name] = feature_group[name][:]
    return pd.DataFrame(data)


def _read_zarr_matrix_all(artifact: Artifact, _: BenchmarkDataset) -> dict[str, Any]:
    root = zarr.open_group(str(artifact.path), mode="r")
    return {
        "metadata": _zarr_metadata_frame(root),
        "feature_names": root["feature_names"][:],
        "features": root["features"][:],
    }


def _zarr_metadata_frame(root: zarr.hierarchy.Group) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": root["sample_id"][:],
            "plate_id": root["plate_id"][:],
            "well_id": root["well_id"][:],
        }
    )


def _read_zarr_wide_matrix(artifact: Artifact, dataset: BenchmarkDataset) -> np.ndarray:
    root = zarr.open_group(str(artifact.path), mode="r")
    feature_group = root["features"]
    columns = [feature_group[name][:] for name in dataset.feature_names]
    return np.column_stack(columns).astype(np.float32, copy=False)


def _read_zarr_matrix(artifact: Artifact, _: BenchmarkDataset) -> np.ndarray:
    root = zarr.open_group(str(artifact.path), mode="r")
    return np.asarray(root["features"][:], dtype=np.float32)


def _read_zarr_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    root = zarr.open_group(str(artifact.path), mode="r")
    feature_group = root["features"]
    columns = [feature_group[name].oindex[rows] for name in dataset.feature_names]
    return np.column_stack(columns).astype(np.float32, copy=False)


def _read_zarr_matrix_rows(
    artifact: Artifact,
    _dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    root = zarr.open_group(str(artifact.path), mode="r")
    return np.asarray(root["features"].oindex[rows, :], dtype=np.float32)


def _read_zarr_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    root = zarr.open_group(str(artifact.path), mode="r")
    feature_group = root["features"]
    names = [dataset.feature_names[index] for index in features]
    columns = [feature_group[name][:] for name in names]
    return np.column_stack(columns).astype(np.float32, copy=False)


def _read_zarr_wide_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    root = zarr.open_group(str(artifact.path), mode="r")
    feature_group = root["features"]
    names = _feature_names(dataset, features)
    metadata = _zarr_metadata_frame(root).iloc[rows]
    columns = [feature_group[name].oindex[rows] for name in names]
    return _mixed_frame(
        metadata=metadata,
        matrix=np.column_stack(columns).astype(np.float32, copy=False),
        dataset=dataset,
        features=features,
    )


def _read_zarr_matrix_features(
    artifact: Artifact,
    _dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    root = zarr.open_group(str(artifact.path), mode="r")
    return np.asarray(root["features"][:, features], dtype=np.float32)


def _read_zarr_matrix_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    root = zarr.open_group(str(artifact.path), mode="r")
    metadata = _zarr_metadata_frame(root).iloc[rows]
    matrix = np.asarray(root["features"].oindex[rows, features], dtype=np.float32)
    return _mixed_frame(
        metadata=metadata,
        matrix=matrix,
        dataset=dataset,
        features=features,
    )


def _tiledb_filters(*, compact: bool) -> tiledb.FilterList:
    """Return attribute filters: none by default, shuffle and zstd when compact."""
    if not compact:
        return tiledb.FilterList([])
    return tiledb.FilterList([tiledb.ByteShuffleFilter(), tiledb.ZstdFilter(level=5)])


def _write_tiledb_wide(
    dataset: BenchmarkDataset,
    path: Path,
    *,
    compact: bool = False,
) -> Artifact:
    domain = tiledb.Domain(_tiledb_profile_dim(dataset.rows))
    attributes = [
        tiledb.Attr(
            name=name,
            dtype=np.dtype("float32"),
            filters=_tiledb_filters(compact=compact),
        )
        for name in dataset.feature_names
    ]
    schema = tiledb.ArraySchema(domain=domain, sparse=False, attrs=attributes)
    tiledb.Array.create(str(path), schema)
    with tiledb.open(str(path), "w") as array:
        array[:] = {
            name: dataset.matrix[:, index]
            for index, name in enumerate(dataset.feature_names)
        }
        _write_tiledb_metadata(array, dataset, layout="wide")
    return Artifact(path=path, bytes=_artifact_size(path))


def _write_tiledb_dense(
    dataset: BenchmarkDataset,
    path: Path,
    *,
    compact: bool = False,
) -> Artifact:
    domain = tiledb.Domain(
        _tiledb_profile_dim(dataset.rows),
        tiledb.Dim(
            name="feature",
            domain=(0, dataset.dimensions - 1),
            tile=min(dataset.dimensions, 1024),
            dtype=np.dtype("int32"),
        ),
    )
    schema = tiledb.ArraySchema(
        domain=domain,
        sparse=False,
        attrs=[
            tiledb.Attr(
                name="value",
                dtype=np.dtype("float32"),
                filters=_tiledb_filters(compact=compact),
            )
        ],
    )
    tiledb.Array.create(str(path), schema)
    with tiledb.open(str(path), "w") as array:
        array[:] = dataset.matrix
        _write_tiledb_metadata(array, dataset, layout="tiledb_dense")
    return Artifact(path=path, bytes=_artifact_size(path))


def _tiledb_profile_dim(rows: int) -> tiledb.Dim:
    return tiledb.Dim(
        name="profile",
        domain=(0, rows - 1),
        tile=min(rows, 1024),
        dtype=np.dtype("int32"),
    )


def _write_tiledb_metadata(
    array: tiledb.DenseArray,
    dataset: BenchmarkDataset,
    *,
    layout: str,
) -> None:
    array.meta["layout"] = layout
    array.meta["feature_names"] = json.dumps(dataset.feature_names)
    for column in ["sample_id", "plate_id", "well_id"]:
        array.meta[column] = json.dumps(dataset.metadata[column].to_list())


def _read_tiledb_wide_all(
    artifact: Artifact,
    dataset: BenchmarkDataset,
) -> pd.DataFrame:
    with tiledb.open(str(artifact.path), "r") as array:
        result = array[:]
        data = _tiledb_metadata_frame(array).to_dict(orient="list")
    for name in dataset.feature_names:
        data[name] = result[name]
    return pd.DataFrame(data)


def _read_tiledb_dense_all(artifact: Artifact, _: BenchmarkDataset) -> dict[str, Any]:
    with tiledb.open(str(artifact.path), "r") as array:
        return {
            "metadata": _tiledb_metadata_frame(array),
            "feature_names": json.loads(str(array.meta["feature_names"])),
            "features": array[:]["value"],
        }


def _tiledb_metadata_frame(array: tiledb.DenseArray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            column: json.loads(str(array.meta[column]))
            for column in ["sample_id", "plate_id", "well_id"]
        }
    )


def _read_tiledb_wide_matrix(
    artifact: Artifact, dataset: BenchmarkDataset
) -> np.ndarray:
    with tiledb.open(str(artifact.path), "r") as array:
        result = array[:]
    columns = [result[name] for name in dataset.feature_names]
    return np.column_stack(columns).astype(np.float32, copy=False)


def _read_tiledb_dense_matrix(
    artifact: Artifact, _dataset: BenchmarkDataset
) -> np.ndarray:
    with tiledb.open(str(artifact.path), "r") as array:
        return np.asarray(array[:]["value"], dtype=np.float32)


def _read_tiledb_wide_rows(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    with tiledb.open(str(artifact.path), "r") as array:
        result = array.multi_index[rows]
    columns = [result[name] for name in dataset.feature_names]
    return np.column_stack(columns).astype(np.float32, copy=False)


def _read_tiledb_dense_rows(
    artifact: Artifact,
    _dataset: BenchmarkDataset,
    rows: np.ndarray,
) -> np.ndarray:
    with tiledb.open(str(artifact.path), "r") as array:
        return np.asarray(array.multi_index[rows, :]["value"], dtype=np.float32)


def _read_tiledb_wide_features(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    names = [dataset.feature_names[index] for index in features]
    with tiledb.open(str(artifact.path), "r") as array:
        result = array.query(attrs=names)[:]
    columns = [result[name] for name in names]
    return np.column_stack(columns).astype(np.float32, copy=False)


def _read_tiledb_wide_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    names = _feature_names(dataset, features)
    with tiledb.open(str(artifact.path), "r") as array:
        metadata = _tiledb_metadata_frame(array).iloc[rows]
        result = array.query(attrs=names).multi_index[rows]
    columns = [result[name] for name in names]
    return _mixed_frame(
        metadata=metadata,
        matrix=np.column_stack(columns).astype(np.float32, copy=False),
        dataset=dataset,
        features=features,
    )


def _read_tiledb_dense_features(
    artifact: Artifact,
    _dataset: BenchmarkDataset,
    features: np.ndarray,
) -> np.ndarray:
    with tiledb.open(str(artifact.path), "r") as array:
        matrix = array[:]["value"]
    return np.asarray(matrix[:, features], dtype=np.float32)


def _read_tiledb_dense_mixed(
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    with tiledb.open(str(artifact.path), "r") as array:
        metadata = _tiledb_metadata_frame(array).iloc[rows]
        matrix = np.asarray(
            array.multi_index[rows, :]["value"][:, features],
            dtype=np.float32,
        )
    return _mixed_frame(
        metadata=metadata,
        matrix=matrix,
        dataset=dataset,
        features=features,
    )


def _read_mixed(
    runner: LayoutRunner,
    artifact: Artifact,
    dataset: BenchmarkDataset,
    rows: np.ndarray,
    features: np.ndarray,
) -> pd.DataFrame:
    readers: dict[tuple[str, str], Callable[..., pd.DataFrame]] = {
        ("csv", "wide"): _read_csv_wide_mixed,
        ("csv", "delimited_array"): _read_csv_delimited_mixed,
        ("csv", "json_array"): _read_csv_json_mixed,
        ("parquet", "wide"): _read_parquet_wide_mixed,
        ("parquet", "fixed_array"): _read_parquet_fixed_mixed,
        ("duckdb", "wide"): _read_duckdb_wide_mixed,
        ("duckdb", "duckdb_array"): _read_duckdb_array_mixed,
        ("zarr", "wide"): _read_zarr_wide_mixed,
        ("zarr", "zarr_matrix"): _read_zarr_matrix_mixed,
        ("tiledb", "wide"): _read_tiledb_wide_mixed,
        ("tiledb", "tiledb_dense"): _read_tiledb_dense_mixed,
        ("vortex", "wide"): _read_vortex_wide_mixed,
        ("vortex", "fixed_array"): _read_vortex_fixed_mixed,
        ("lance", "wide"): _read_lance_wide_mixed,
        ("lance", "fixed_array"): _read_lance_fixed_mixed,
    }
    return readers[(runner.backend, runner.layout)](artifact, dataset, rows, features)


def _mixed_columns(dataset: BenchmarkDataset, features: np.ndarray) -> list[str]:
    return [*METADATA_COLUMNS, *_feature_names(dataset, features)]


def _feature_names(dataset: BenchmarkDataset, features: np.ndarray) -> list[str]:
    return [dataset.feature_names[index] for index in features]


def _mixed_frame(
    *,
    metadata: pd.DataFrame,
    matrix: np.ndarray,
    dataset: BenchmarkDataset,
    features: np.ndarray,
) -> pd.DataFrame:
    frame = metadata.reset_index(drop=True).copy()
    for position, name in enumerate(_feature_names(dataset, features)):
        frame[name] = matrix[:, position]
    return frame


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


GENERATED_PATHS = ("results", "site", "README.md")


def _git_commit() -> str:
    """Return the short commit hash, with `-dirty` when tracked code has changed.

    Files that the benchmark itself writes do not count as changes.
    """
    try:
        commit = _git("rev-parse", "--short", "HEAD")
        changes = _git(
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            ".",
            *(f":(exclude){path}" for path in GENERATED_PATHS),
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return f"{commit}-dirty" if changes else commit


def _git(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()


_thread_limit = 1


def apply_thread_limits(threads: int) -> dict[str, int | str]:
    """Limit thread pools that the libraries expose and report what was set.

    Lance and Vortex run their own native runtimes. This function cannot cap
    them, so the measured CPU time in the results shows their real parallelism.
    """
    global _thread_limit  # noqa: PLW0603
    _thread_limit = threads
    pa.set_cpu_count(threads)
    pa.set_io_thread_count(threads)
    blosc.set_nthreads(threads)
    blosc.use_threads = threads > 1
    with contextlib.suppress(tiledb.TileDBError):
        # The default context can be set once per process.
        tiledb.default_ctx(
            tiledb.Config(
                {
                    "sm.compute_concurrency_level": str(threads),
                    "sm.io_concurrency_level": str(threads),
                }
            )
        )
    tiledb_level = int(tiledb.default_ctx().config()["sm.compute_concurrency_level"])
    return {
        "arrow_cpu_threads": pa.cpu_count(),
        "arrow_io_threads": pa.io_thread_count(),
        "duckdb_threads": threads,
        "zarr_blosc_threads": threads,
        "tiledb_concurrency_level": tiledb_level,
        "lance": "library default",
        "vortex": "library default",
    }


def hardware_info() -> dict[str, Any]:
    """Return the CPU model, core count, and memory of this machine."""
    return {
        "cpu_model": _cpu_model(),
        "logical_cores": os.cpu_count() or 1,
        "memory_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
    }


def _cpu_model() -> str:
    """Return a readable CPU name, with the platform name as a fallback."""
    commands = [
        ["sysctl", "-n", "machdep.cpu.brand_string"],
        ["sh", "-c", "grep -m1 'model name' /proc/cpuinfo | cut -d: -f2"],
    ]
    for command in commands:
        try:
            name = subprocess.run(
                command, capture_output=True, check=True, text=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
        if name:
            return name
    return platform.processor() or platform.machine() or "unknown"


MIN_SAMPLE_SECONDS = 0.05
NOISY_RANGE = 0.5
MAX_CALLS_PER_SAMPLE = 1_000


def _calls_per_sample(call_seconds: float) -> int:
    """Return how many calls make one sample last at least 50 ms.

    Timings of a few milliseconds are dominated by timer and thread start-up
    noise. Repeating the call inside one sample and dividing removes most of it.
    """
    if call_seconds <= 0:
        return MAX_CALLS_PER_SAMPLE
    needed = math.ceil(MIN_SAMPLE_SECONDS / call_seconds - 1e-9)
    return min(max(needed, 1), MAX_CALLS_PER_SAMPLE)


def _timed_repeated(call: Callable[[], Any], calls: int) -> tuple[Any, float, float]:
    """Return the last result and the total wall and CPU seconds of all calls."""
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    value = None
    for _ in range(calls):
        value = call()
    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    return value, wall, cpu


def _timed(call: Callable[[], Any]) -> tuple[Any, float, float]:
    """Return the call result, wall seconds, and process CPU seconds."""
    wall_start = time.perf_counter()
    cpu_start = time.process_time()
    value = call()
    cpu = time.process_time() - cpu_start
    wall = time.perf_counter() - wall_start
    return value, wall, cpu


def _duckdb_connect(path: Path, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    """Open a DuckDB database with the configured thread limit."""
    return duckdb.connect(
        str(path),
        read_only=read_only,
        config={"threads": _thread_limit},
    )


def read_raw_results(path: Path = Path("results/raw_results.parquet")) -> pd.DataFrame:
    """Read raw benchmark results."""
    return pd.read_parquet(path)
