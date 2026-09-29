"""Tests for the benchmark runner."""

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from numcodecs import blosc
from zarr.core.config import config as zarr_config

from array_we_there_yet import benchmark
from array_we_there_yet.benchmark import (
    BenchmarkConfig,
    _write_csv_delimited_array,
    _write_csv_json_array,
    apply_thread_limits,
    hardware_info,
    layout_runners,
    run_benchmarks,
    summarize_results,
)
from array_we_there_yet.data import BenchmarkDataset, make_synthetic_dataset
from array_we_there_yet.report import ratio_table


def test_benchmark_runner_writes_results(tmp_path: Path) -> None:
    """A small benchmark run writes raw and summary files."""
    config = BenchmarkConfig(
        rows=12,
        dimensions=(4,),
        measured_repetitions=2,
        warmups=0,
        random_row_count=3,
        feature_projection_count=2,
        output_dir=tmp_path / "results",
        artifact_dir=tmp_path / "results" / "artifacts",
    )

    raw = run_benchmarks(config)
    summary = summarize_results(raw, config.output_dir)
    ratios = ratio_table(summary)

    assert not raw.empty
    assert not summary.empty
    assert not ratios.empty
    assert (config.output_dir / "raw_results.parquet").exists()
    assert (config.output_dir / "summary.parquet").exists()
    assert set(raw["operation"]) == {
        "feature_projection",
        "full_read",
        "matrix_materialization",
        "mixed_retrieval",
        "random_rows",
        "vector_norm",
        "write",
    }
    mixed = raw[raw["operation"] == "mixed_retrieval"]
    assert set(mixed["operation_parameter"]) == {"3_rows_2_features"}
    assert pd.api.types.is_float_dtype(raw["elapsed_seconds"])
    assert (raw["cpu_seconds"] >= 0).all()
    assert (raw["calls"] >= 1).all()
    assert (raw[raw["operation"] == "write"]["calls"] == 1).all()
    assert raw[raw["operation"] == "matrix_materialization"]["calls"].max() > 1
    assert set(raw["profile"]) == {"default", "compact"}
    assert set(summary["profile"]) == {"default", "compact"}
    assert not list(config.output_dir.glob("*.csv"))
    encodings = pd.read_parquet(config.output_dir / "encodings.parquet")
    assert set(encodings.columns) == {
        "backend",
        "layout",
        "profile",
        "dimensions",
        "encoding",
    }
    runner_keys = {
        (runner.backend, runner.layout, runner.profile) for runner in layout_runners()
    }
    assert set(
        zip(encodings["backend"], encodings["layout"], encodings["profile"])
    ) == (runner_keys)
    assert encodings["encoding"].notna().all()
    assert (summary["median_parallelism"] > 0).all()
    assert (summary["median_cpu_seconds"] >= 0).all()
    environment = json.loads((config.output_dir / "environment.json").read_text())
    assert environment["git_commit"] == set(raw["git_commit"]).pop()
    grouped = raw.groupby(
        [
            "backend",
            "layout",
            "profile",
            "dimensions",
            "operation",
            "operation_parameter",
        ]
    )
    assert set(grouped.size()) == {config.measured_repetitions}
    assert set(summary["repetitions"]) == {config.measured_repetitions}


def test_layout_runners_compare_array_native_formats_without_arrow_ipc() -> None:
    """The benchmark compares array-native formats instead of Arrow IPC."""
    backends = {runner.backend for runner in layout_runners()}

    assert "arrow_ipc" not in backends
    assert {"zarr", "tiledb", "vortex", "lance"}.issubset(backends)


def test_apply_thread_limits_caps_arrow_thread_pools() -> None:
    """Arrow reads and writes use the configured number of threads."""
    cpu_count, io_count = pa.cpu_count(), pa.io_thread_count()
    zarr_before = {
        "async.concurrency": zarr_config.get("async.concurrency"),
        "threading.max_workers": zarr_config.get("threading.max_workers"),
    }
    try:
        limits = apply_thread_limits(1)

        assert pa.cpu_count() == 1
        assert pa.io_thread_count() == 1
        assert limits["arrow_cpu_threads"] == 1
        assert limits["arrow_io_threads"] == 1
        assert limits["duckdb_threads"] == 1
        assert limits["zarr_blosc_threads"] == 1
        assert blosc.use_threads is False
        assert limits["lance"] == "library default"
        assert limits["vortex"] == "library default"
    finally:
        pa.set_cpu_count(cpu_count)
        pa.set_io_thread_count(io_count)
        blosc.use_threads = None
        zarr_config.set(zarr_before)


def test_apply_thread_limits_caps_zarr_thread_pools() -> None:
    """Zarr chunk loading and its sync executor honor the thread limit.

    Zarr 3 loads and decodes chunks through its own pool. The Blosc thread
    setting does not cap that pool, so the limit must go through the zarr
    config.
    """
    concurrency = zarr_config.get("async.concurrency")
    max_workers = zarr_config.get("threading.max_workers")
    threads = 2
    try:
        apply_thread_limits(threads)

        assert zarr_config.get("async.concurrency") == threads
        assert zarr_config.get("threading.max_workers") == threads
    finally:
        zarr_config.set(
            {"async.concurrency": concurrency, "threading.max_workers": max_workers}
        )


def test_hardware_info_describes_the_machine() -> None:
    """Environment metadata includes CPU model, core count, and memory."""
    info = hardware_info()

    assert info["cpu_model"]
    assert info["logical_cores"] >= 1
    assert info["memory_bytes"] > 0


def test_csv_packed_layouts_use_the_same_float_precision(tmp_path: Path) -> None:
    """JSON and delimited packed arrays store the same 9 significant digits."""
    dataset = make_synthetic_dataset(rows=4, dimensions=6, seed=7)

    delimited = _write_csv_delimited_array(dataset, tmp_path / "delimited")
    packed_json = _write_csv_json_array(dataset, tmp_path / "json")

    delimited_cell = pd.read_csv(delimited.path / "data.csv")["features"].iloc[0]
    json_cell = pd.read_csv(packed_json.path / "data.csv")["features"].iloc[0]
    json_values = json_cell.strip("[]").split(",")
    delimited_values = delimited_cell.split(";")
    assert json_values == delimited_values
    assert np.allclose(np.array(json_values, dtype=np.float32), dataset.matrix[0])


def test_duckdb_array_layout_stores_fixed_size_float32(tmp_path: Path) -> None:
    """The DuckDB array layout keeps float32 values in a fixed-size array."""
    dataset = make_synthetic_dataset(rows=6, dimensions=5, seed=3)
    runner = next(
        runner
        for runner in layout_runners()
        if runner.backend == "duckdb" and runner.layout == "duckdb_array"
    )

    artifact = runner.write(dataset, tmp_path / "profiles.duckdb")

    with duckdb.connect(str(artifact.path), read_only=True) as connection:
        column_types = dict(
            connection.execute(
                "SELECT column_name, data_type FROM duckdb_columns()"
            ).fetchall()
        )
    assert column_types["features"] == "FLOAT[5]"
    matrix = runner.read_matrix(artifact, dataset)
    assert matrix.dtype == np.float32
    np.testing.assert_allclose(matrix, dataset.matrix, rtol=1e-6, atol=1e-6)
    features = np.array([0, 3])
    np.testing.assert_allclose(
        runner.read_features(artifact, dataset, features),
        dataset.matrix[:, features],
        rtol=1e-6,
        atol=1e-6,
    )


def _small_config(
    tmp_path: Path, dimensions: tuple[int, ...] = (4,)
) -> BenchmarkConfig:
    """Return a tiny benchmark configuration for robustness tests."""
    return BenchmarkConfig(
        rows=6,
        dimensions=dimensions,
        measured_repetitions=1,
        warmups=0,
        random_row_count=2,
        feature_projection_count=2,
        output_dir=tmp_path / "results",
        artifact_dir=tmp_path / "results" / "artifacts",
    )


def test_artifacts_are_removed_after_each_layout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the layout being measured keeps files, and none remain at the end."""
    config = _small_config(tmp_path)
    layouts_on_disk: list[set[str]] = []
    original = benchmark._measure_reads

    def spy(**kwargs: Any) -> None:  # noqa: ANN401
        layouts_on_disk.append(
            {path.name.split("_iter-")[0] for path in config.artifact_dir.iterdir()}
        )
        original(**kwargs)

    monkeypatch.setattr(benchmark, "_measure_reads", spy)

    run_benchmarks(config)

    assert layouts_on_disk
    assert all(len(layouts) == 1 for layouts in layouts_on_disk)
    assert list(config.artifact_dir.iterdir()) == []


def test_partial_results_survive_a_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Results for finished feature counts are on disk when a later one fails."""
    failing_dimensions = 8
    config = _small_config(tmp_path, dimensions=(4, failing_dimensions))
    original = benchmark.make_synthetic_dataset

    def flaky(*, dimensions: int, **kwargs: Any) -> BenchmarkDataset:  # noqa: ANN401
        if dimensions == failing_dimensions:
            message = "No space left on device"
            raise OSError(message)
        return original(dimensions=dimensions, **kwargs)

    monkeypatch.setattr(benchmark, "make_synthetic_dataset", flaky)

    with pytest.raises(OSError, match="No space left"):
        run_benchmarks(config)

    partial = pd.read_parquet(config.output_dir / "raw_results.parquet")
    assert set(partial["dimensions"]) == {4}
    assert not partial.empty
    assert (config.output_dir / "encodings.parquet").exists()


def test_low_disk_space_stops_the_run_before_any_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run that cannot fit on disk fails early with a clear message."""
    config = _small_config(tmp_path)
    monkeypatch.setattr(
        benchmark.shutil,
        "disk_usage",
        lambda _: shutil._ntuple_diskusage(total=100, used=99, free=1),
    )

    with pytest.raises(RuntimeError, match="free disk space"):
        run_benchmarks(config)

    assert list(config.artifact_dir.iterdir()) == []


def test_required_disk_space_grows_with_the_largest_feature_count() -> None:
    """The estimate uses the widest dataset and every write kept for one layout."""
    small = BenchmarkConfig(rows=10, dimensions=(4,), warmups=1, measured_repetitions=3)
    large = BenchmarkConfig(
        rows=10, dimensions=(4, 400), warmups=1, measured_repetitions=3
    )

    assert benchmark.required_disk_bytes(large) == 100 * benchmark.required_disk_bytes(
        small
    )


def _git(directory: Path, *arguments: str) -> None:
    subprocess.run(["git", *arguments], cwd=directory, check=True, capture_output=True)


def _repository(tmp_path: Path) -> Path:
    """Create a git repository with one commit of source and one of results."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test")
    (tmp_path / "results").mkdir()
    (tmp_path / "source.py").write_text("value = 1\n", encoding="utf-8")
    (tmp_path / "results" / "summary.parquet").write_text("a", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-q", "-m", "first")
    return tmp_path


def test_git_commit_marks_uncommitted_source_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run from modified source code records a dirty commit."""
    monkeypatch.chdir(_repository(tmp_path))
    clean = benchmark._git_commit()

    (tmp_path / "source.py").write_text("value = 2\n", encoding="utf-8")

    assert benchmark._git_commit() == f"{clean}-dirty"
    assert "-dirty" not in clean


def test_git_commit_ignores_files_the_benchmark_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Changed results, the site, and README do not make the code dirty."""
    monkeypatch.chdir(_repository(tmp_path))
    clean = benchmark._git_commit()

    (tmp_path / "results" / "summary.parquet").write_text("b", encoding="utf-8")
    (tmp_path / "README.md").write_text("new", encoding="utf-8")
    (tmp_path / "untracked.txt").write_text("x", encoding="utf-8")

    assert benchmark._git_commit() == clean


def test_git_commit_is_unknown_outside_a_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A directory without git history records an unknown commit."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path.parent))

    assert benchmark._git_commit() == "unknown"


def test_calls_per_sample_repeats_fast_operations_up_to_a_limit() -> None:
    """A fast call is repeated until one sample lasts about 50 ms."""
    calls_for_one_millisecond = 50
    assert benchmark._calls_per_sample(0.001) == calls_for_one_millisecond
    assert benchmark._calls_per_sample(0.05) == 1
    assert benchmark._calls_per_sample(0.5) == 1
    assert benchmark._calls_per_sample(0.00001) == benchmark.MAX_CALLS_PER_SAMPLE
    assert benchmark._calls_per_sample(0.0) == benchmark.MAX_CALLS_PER_SAMPLE


def test_timed_repeated_calls_the_function_and_returns_the_last_value() -> None:
    """The total time covers every call, and the caller divides by the count."""
    repeats = 4
    counter = {"calls": 0}

    def call() -> int:
        counter["calls"] += 1
        return counter["calls"]

    value, wall, cpu = benchmark._timed_repeated(call, repeats)

    assert value == repeats
    assert counter["calls"] == repeats
    assert wall >= 0
    assert cpu >= 0


def test_csv_is_skipped_above_the_row_cap(tmp_path: Path) -> None:
    """A CSV row cap leaves CSV out of runs with more rows than the cap."""
    config = BenchmarkConfig(
        rows=12,
        dimensions=(4,),
        measured_repetitions=1,
        warmups=0,
        random_row_count=2,
        feature_projection_count=2,
        output_dir=tmp_path / "results",
        artifact_dir=tmp_path / "results" / "artifacts",
        csv_max_rows=11,
    )

    raw = run_benchmarks(config)

    assert "csv" not in set(raw["backend"])
    assert "parquet" in set(raw["backend"])


def test_measure_reads_can_time_only_some_operations(tmp_path: Path) -> None:
    """The `only` filter keeps just the named operations."""
    config = BenchmarkConfig(
        rows=12,
        dimensions=(4,),
        measured_repetitions=1,
        warmups=0,
        artifact_dir=tmp_path / "artifacts",
    )
    config.artifact_dir.mkdir()
    dataset = make_synthetic_dataset(rows=12, dimensions=4, seed=1)
    runner = next(
        runner
        for runner in layout_runners()
        if (runner.backend, runner.layout, runner.profile)
        == ("parquet", "wide", "default")
    )
    records: list[benchmark.BenchmarkResult] = []
    artifact = benchmark._measure_writes(
        config=config,
        runner=runner,
        dataset=dataset,
        timestamp="t",
        git_commit="c",
        records=records,
    )
    records.clear()

    benchmark._measure_reads(
        config=config,
        runner=runner,
        dataset=dataset,
        artifact=artifact,
        selected_rows=np.array([1, 3]),
        selected_features=np.array([0, 2]),
        timestamp="t",
        git_commit="c",
        records=records,
        only=frozenset({"matrix_materialization"}),
    )

    assert {record.operation for record in records} == {"matrix_materialization"}
