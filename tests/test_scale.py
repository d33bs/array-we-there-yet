"""Tests for the row-count sweep and the scaling check."""

from pathlib import Path

import pandas as pd
import pytest

from array_we_there_yet.benchmark import BenchmarkConfig, run_benchmarks
from array_we_there_yet.cli import ArrayWeThereYetCLI
from array_we_there_yet.main import report
from array_we_there_yet.report import EXAMPLE_LAYOUTS
from array_we_there_yet.scale import (
    parquet_read_bytes,
    run_row_sweep,
    run_scaling_check,
    scaling_rows,
    scaling_table,
)


def _csv_wide_summary(bytes_per_row: int = 93_750, rows: int = 2_000) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "backend": "csv",
                "layout": "wide",
                "profile": "default",
                "rows": rows,
                "dimensions": 4,
                "operation": "write",
                "operation_parameter": "all",
                "median_seconds": 1.0,
                "artifact_bytes": bytes_per_row * rows,
            }
        ]
    )


def test_scaling_rows_makes_a_csv_wide_file_of_the_requested_size() -> None:
    """A 1.5 GB file at 93,750 bytes per row has 16,000 rows."""
    assert scaling_rows(_csv_wide_summary()) == pytest.approx(16_000)
    assert scaling_rows(_csv_wide_summary(), dataset_gb=0.75) == pytest.approx(8_000)


def test_scaling_table_reports_the_median_of_each_layout() -> None:
    """The table holds size, write time, and read time for each layout."""
    rows = []
    for layout, write, matrix, size in [
        ("wide", 4.0, 2.0, 900),
        ("fixed_array", 1.0, 0.5, 300),
    ]:
        for operation, seconds in [
            ("write", write),
            ("matrix_materialization", matrix),
        ]:
            for repetition in range(2):
                rows.append(
                    {
                        "backend": "parquet",
                        "layout": layout,
                        "profile": "default",
                        "rows": 50,
                        "dimensions": 4,
                        "operation": operation,
                        "elapsed_seconds": seconds + repetition * 0.2,
                        "artifact_bytes": size,
                    }
                )

    table = scaling_table(pd.DataFrame(rows))

    fixed = table[table["layout"] == "fixed_array"].iloc[0]
    assert fixed["size_bytes"] == pytest.approx(300)
    assert fixed["write_seconds"] == pytest.approx(1.1)
    assert fixed["matrix_seconds"] == pytest.approx(0.6)
    assert set(table["layout"]) == {"wide", "fixed_array"}


def test_row_sweep_runs_each_row_count_and_leaves_csv_out_of_large_ones(
    tmp_path: Path,
) -> None:
    """Large row counts skip CSV, and every row count gets results."""
    summary = run_row_sweep(
        row_counts=(12, 24),
        dimensions=4,
        output_dir=tmp_path,
        measured_repetitions=1,
        warmups=0,
        csv_max_rows=12,
    )

    assert set(summary["rows"]) == {12, 24}
    csv_rows = summary[summary["backend"] == "csv"]["rows"]
    assert set(csv_rows) == {12}
    assert (tmp_path / "row_sweep_raw.parquet").exists()
    assert (tmp_path / "row_sweep_summary.parquet").exists()


def test_scaling_check_measures_every_layout_of_the_example(tmp_path: Path) -> None:
    """A tiny file goes through the same layouts as the real-world example."""
    summary = _csv_wide_summary(bytes_per_row=100, rows=10)

    table = run_scaling_check(summary=summary, output_dir=tmp_path, dataset_gb=1e-6)

    measured = set(
        table[["backend", "layout", "profile"]].itertuples(index=False, name=None)
    )
    assert measured == set(EXAMPLE_LAYOUTS)
    assert (table["rows"] == len(range(10))).all()
    assert (table["size_bytes"] > 0).all()
    parquet = table[table["backend"] == "parquet"]
    assert (parquet["footer_bytes"] > 0).all()
    assert (parquet["bytes_8_features"] > 0).all()
    assert table[table["backend"] != "parquet"]["footer_bytes"].isna().all()
    assert (tmp_path / "scaling_check.parquet").exists()
    assert list((tmp_path / "scaling" / "artifacts").iterdir()) == []


def test_parquet_read_bytes_for_every_column_is_the_file_size(tmp_path: Path) -> None:
    """Footer, column chunks, and framing add up to the whole file."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({f"feature_{i}": [float(i)] * 200 for i in range(20)})
    path = tmp_path / "wide.parquet"
    pq.write_table(table, path)

    assert parquet_read_bytes(path) == path.stat().st_size


def test_parquet_read_bytes_for_some_columns_is_smaller(tmp_path: Path) -> None:
    """Reading two of twenty columns downloads the footer and two column chunks."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table(
        {f"feature_{i}": [float(i) * 1.5 + j for j in range(2000)] for i in range(20)}
    )
    path = tmp_path / "wide.parquet"
    pq.write_table(table, path, use_dictionary=False, compression="none")

    two = parquet_read_bytes(path, ["feature_0", "feature_1"])

    assert two < path.stat().st_size / 5
    assert two > pq.ParquetFile(path).metadata.serialized_size


def test_the_command_line_offers_sweep_and_scaling_commands() -> None:
    """The two long checks are commands, like run and combine."""
    commands = {name for name in dir(ArrayWeThereYetCLI) if not name.startswith("_")}

    assert {"run", "combine", "report", "sweep", "scaling"} <= commands


def test_report_adds_the_row_scaling_figure_when_a_sweep_exists(
    tmp_path: Path,
) -> None:
    """A saved sweep summary appears as a figure next to the other results."""
    config = BenchmarkConfig(
        rows=6,
        dimensions=(4,),
        measured_repetitions=1,
        warmups=0,
        random_row_count=2,
        feature_projection_count=2,
        output_dir=tmp_path / "results",
        artifact_dir=tmp_path / "results" / "artifacts",
        figure_dir=tmp_path / "figures",
    )
    run_benchmarks(config)
    rows = []
    for count, seconds in [(1_000, 0.01), (100_000, 1.0)]:
        for operation in [
            "matrix_materialization",
            "random_rows",
            "feature_projection",
        ]:
            rows.append(
                {
                    "backend": "parquet",
                    "layout": "wide",
                    "profile": "default",
                    "rows": count,
                    "dimensions": 4,
                    "operation": operation,
                    "median_seconds": seconds,
                }
            )
    pd.DataFrame(rows).to_parquet(
        tmp_path / "results" / "row_sweep_summary.parquet", index=False
    )

    report(
        output_dir=str(tmp_path / "results"),
        figure_dir=str(tmp_path / "figures"),
        update_readme_file=False,
    )

    assert (tmp_path / "figures" / "row_scaling.png").exists()
