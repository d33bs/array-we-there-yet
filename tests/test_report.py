"""Tests for report generation."""

from pathlib import Path

import pandas as pd

from array_we_there_yet.report import (
    _ordered_backends,
    render_results_section,
    write_figures,
)


def test_write_figures_prioritizes_absolute_comparisons(tmp_path: Path) -> None:
    """Report figures include side-by-side absolute comparisons first."""
    summary = pd.DataFrame(
        [
            {
                "backend": backend,
                "layout": layout,
                "dataset": "synthetic",
                "rows": 10,
                "dimensions": dimensions,
                "dtype": "float32",
                "operation": operation,
                "operation_parameter": "all",
                "threads": 1,
                "compression": "none",
                "median_seconds": seconds,
                "q25_seconds": seconds,
                "q75_seconds": seconds,
                "artifact_bytes": artifact_bytes,
                "repetitions": 1,
            }
            for backend in ["csv", "parquet"]
            for dimensions in [4, 8]
            for operation in ["write", "matrix_materialization"]
            for layout, seconds, artifact_bytes in [
                ("wide", 2.0, 2_000),
                ("fixed_array", 1.0, 1_000),
            ]
        ]
    )

    paths = write_figures(summary, tmp_path)
    names = [path.name for path in paths]
    section = render_results_section(summary=summary, figure_paths=paths)

    assert names[0] == "write_absolute_comparison.png"
    assert "matrix_materialization_absolute_comparison.png" in names
    assert "Read this first" in section
    assert "The largest time gain" in section
    assert "Primary side-by-side figures" in section
    assert "Secondary ratio figures" in section


def test_ordered_backends_puts_csv_first_and_arrow_ipc_last() -> None:
    """Plot panels use the requested benchmark-reading order."""
    assert _ordered_backends(["arrow_ipc", "duckdb", "csv", "parquet"]) == [
        "csv",
        "parquet",
        "duckdb",
        "arrow_ipc",
    ]
