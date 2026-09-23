"""Tests for the benchmark runner."""

from pathlib import Path

import pandas as pd

from array_we_there_yet.benchmark import (
    BenchmarkConfig,
    layout_runners,
    run_benchmarks,
    summarize_results,
)
from array_we_there_yet.report import ratio_table


def test_benchmark_runner_writes_results(tmp_path: Path) -> None:
    """A small benchmark run writes raw and summary files."""
    config = BenchmarkConfig(
        rows=12,
        dimensions=(4,),
        measured_repetitions=1,
        warmups=0,
        random_row_count=3,
        feature_projection_count=2,
        output_dir=tmp_path / "results",
        artifact_dir=tmp_path / "results" / "artifacts",
        figure_dir=tmp_path / "figures",
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
        "random_rows",
        "vector_norm",
        "write",
    }
    assert pd.api.types.is_float_dtype(raw["elapsed_seconds"])


def test_layout_runners_compare_vortex_and_lance_without_arrow_ipc() -> None:
    """The benchmark compares Vortex and Lance instead of Arrow IPC."""
    backends = {runner.backend for runner in layout_runners()}

    assert "arrow_ipc" not in backends
    assert {"vortex", "lance"}.issubset(backends)
