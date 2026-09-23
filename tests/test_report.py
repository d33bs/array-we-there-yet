"""Tests for report generation."""

from pathlib import Path

import pandas as pd

from array_we_there_yet.report import (
    _direction_title,
    _facet_grid_shape,
    _ordered_backends,
    _ratio_axis_label,
    _time_summary_ratios,
    render_results_section,
    write_figures,
    write_parquet_performance_tables,
    write_ratio_tables,
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

    assert names[0] == "combined_facet_overview.png"
    assert "write_absolute_comparison.png" in names
    assert "matrix_materialization_absolute_comparison.png" in names
    assert "parquet_performance_tracking.png" in names
    assert "Summary of findings" in section
    assert "The largest time gain" in section
    assert "Primary facet plots" in section
    assert "![Benchmark Ratios]" in section
    assert "Figure 1. Time panels show time ratios." in section
    assert "Matrix materialization means reading the data into one" in section
    assert "Feature projection means reading only selected features." in section
    assert "Vector norm means the length of each feature vector." in section
    assert "Median Time shows the middle time ratio." in section
    assert "Worst Time shows the largest time ratio." in section
    assert "Ratios let different operations fit in one compact figure." in section
    assert "The y-axis uses a log scale to show small and large changes." in section
    assert "Figure 2. Parquet tracking shows absolute time" in section
    assert "Parquet performance tracking" in section
    assert "![Write]" not in section
    assert "| backend" not in section
    assert "[CSV ratio table](results/ratio_summary.csv)" in section
    assert "[Parquet ratio table](results/ratio_summary.parquet)" in section
    assert "[CSV Parquet tracking table](results/parquet_performance.csv)" in section
    assert "[Parquet tracking table](results/parquet_performance.parquet)" in section
    assert "Detailed per-operation plot files remain in `figures/`." in section

    ratio_table_paths = write_ratio_tables(summary, tmp_path)
    assert {path.name for path in ratio_table_paths} == {
        "ratio_summary.csv",
        "ratio_summary.parquet",
    }
    assert all(path.exists() for path in ratio_table_paths)

    parquet_table_paths = write_parquet_performance_tables(summary, tmp_path)
    assert {path.name for path in parquet_table_paths} == {
        "parquet_performance.csv",
        "parquet_performance.parquet",
    }
    assert all(path.exists() for path in parquet_table_paths)


def test_ordered_backends_puts_csv_first_and_fast_formats_later() -> None:
    """Plot panels use the requested benchmark-reading order."""
    assert _ordered_backends(["lance", "vortex", "duckdb", "csv", "parquet"]) == [
        "csv",
        "parquet",
        "duckdb",
        "vortex",
        "lance",
    ]


def test_combined_facet_grid_stays_compact() -> None:
    """Combined overview uses no more than three facets per row."""
    assert _facet_grid_shape(9) == (3, 3)
    assert _direction_title("Write", better="lower") == "Write\n(lower is better)"
    assert _ratio_axis_label("time_ratio") == "Time ratio (array / wide)"
    assert _ratio_axis_label("artifact_size_ratio") == "Size ratio (array / wide)"


def test_time_summary_ratios_describe_typical_and_worst_time() -> None:
    """Summary facets show median and worst time ratios by layout."""
    expected_median = 0.5
    expected_worst = 2.0
    ratios = pd.DataFrame(
        [
            {
                "backend": "parquet",
                "layout": "fixed_array",
                "dimensions": 8,
                "operation": operation,
                "time_ratio": ratio,
            }
            for operation, ratio in [
                ("write", 2.0),
                ("full_read", 0.5),
                ("matrix_materialization", 0.25),
            ]
        ]
    )

    summary = _time_summary_ratios(ratios)

    assert summary.loc[0, "median_time_ratio"] == expected_median
    assert summary.loc[0, "worst_time_ratio"] == expected_worst
