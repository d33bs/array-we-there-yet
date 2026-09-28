"""Row-count sweep and a scaling check against a real 1.5 GB file.

The main benchmark uses 2,000 rows, about 65 MB at the largest feature count. These
two functions check what happens with more rows.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from array_we_there_yet.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    _check_disk_space,
    _git_commit,
    _measure_reads,
    _measure_writes,
    _remove_artifacts,
    apply_thread_limits,
    layout_runners,
    run_benchmarks,
    summarize_results,
)
from array_we_there_yet.data import make_synthetic_dataset
from array_we_there_yet.report import (
    EXAMPLE_DATASET_GB,
    EXAMPLE_LAYOUTS,
    default_profile,
)

ROW_SWEEP_COUNTS = (2_000, 20_000, 200_000)
SWEEP_DIMENSIONS = 1_024
CSV_MAX_ROWS = 20_000
SCALING_REPETITIONS = 2
STREAM_FEATURES = 8


def run_row_sweep(  # noqa: PLR0913
    *,
    row_counts: tuple[int, ...] = ROW_SWEEP_COUNTS,
    dimensions: int = SWEEP_DIMENSIONS,
    output_dir: Path = Path("results"),
    measured_repetitions: int = 3,
    warmups: int = 1,
    csv_max_rows: int = CSV_MAX_ROWS,
) -> pd.DataFrame:
    """Run the benchmark at several row counts and one feature count.

    CSV is left out above `csv_max_rows` rows. Its files are 2.7 times the raw
    size and it is the slowest format.
    """
    frames = []
    for rows in row_counts:
        directory = output_dir / f"sweep_rows_{rows}"
        config = BenchmarkConfig(
            rows=rows,
            dimensions=(dimensions,),
            measured_repetitions=measured_repetitions,
            warmups=warmups,
            output_dir=directory,
            artifact_dir=directory / "artifacts",
            csv_max_rows=csv_max_rows,
        )
        frames.append(run_benchmarks(config))
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(output_dir / "row_sweep_raw.parquet", index=False)
    return summarize_results(raw, output_dir, "row_sweep_summary.parquet")


def scaling_rows(summary: pd.DataFrame, dataset_gb: float = EXAMPLE_DATASET_GB) -> int:
    """Return the row count that makes a CSV wide file as large as `dataset_gb`."""
    default = default_profile(summary)
    top = default[default["dimensions"] == default["dimensions"].max()]
    csv = top[
        (top["backend"] == "csv")
        & (top["layout"] == "wide")
        & (top["operation"] == "write")
    ].iloc[0]
    bytes_per_row = float(csv["artifact_bytes"]) / float(csv["rows"])
    return round(dataset_gb * 1e9 / bytes_per_row)


def run_scaling_check(
    *,
    summary: pd.DataFrame,
    output_dir: Path = Path("results"),
    dataset_gb: float = EXAMPLE_DATASET_GB,
) -> pd.DataFrame:
    """Write and read one real 1.5 GB CSV wide file and the layouts of the example.

    The result holds the measured size, write time, and matrix materialization
    time of each layout. It shows how close the linear scaling in the
    real-world example comes to a real file.
    """
    default = default_profile(summary)
    dimensions = int(default["dimensions"].max())
    rows = scaling_rows(summary, dataset_gb)
    directory = output_dir / "scaling"
    config = BenchmarkConfig(
        rows=rows,
        dimensions=(dimensions,),
        measured_repetitions=SCALING_REPETITIONS,
        warmups=0,
        output_dir=directory,
        artifact_dir=directory / "artifacts",
    )
    directory.mkdir(parents=True, exist_ok=True)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    apply_thread_limits(config.threads)
    _check_disk_space(config)
    dataset = make_synthetic_dataset(rows=rows, dimensions=dimensions, seed=config.seed)
    picks = np.arange(min(8, rows))
    features = np.arange(min(8, dimensions))
    timestamp = pd.Timestamp.now("UTC").isoformat()
    commit = _git_commit()

    records: list[BenchmarkResult] = []
    streaming: dict[tuple[str, str, str], dict[str, float]] = {}
    runners = {
        (runner.backend, runner.layout, runner.profile): runner
        for runner in layout_runners()
    }
    for key in EXAMPLE_LAYOUTS:
        runner = runners.get(key)
        if runner is None:
            continue
        artifact = _measure_writes(
            config=config,
            runner=runner,
            dataset=dataset,
            timestamp=timestamp,
            git_commit=commit,
            records=records,
        )
        if runner.backend == "parquet":
            streaming[key] = _parquet_streaming(artifact.path, runner.layout, dataset)
        _measure_reads(
            config=config,
            runner=runner,
            dataset=dataset,
            artifact=artifact,
            selected_rows=picks,
            selected_features=features,
            timestamp=timestamp,
            git_commit=commit,
            records=records,
            only=frozenset({"matrix_materialization"}),
        )
        _remove_artifacts(config, runner, dataset)

    raw = pd.DataFrame([record.__dict__ for record in records])
    raw.to_parquet(directory / "scaling_raw.parquet", index=False)
    table = scaling_table(raw, streaming)
    table.to_parquet(output_dir / "scaling_check.parquet", index=False)
    return table


PARQUET_FRAMING_BYTES = 12  # "PAR1" at both ends and a 4-byte footer length.


def parquet_read_bytes(path: Path, columns: list[str] | None = None) -> int:
    """Return the bytes a range-request client downloads to read some columns.

    A client reads the footer, then only the column chunks it needs. Reading every
    column gives the size of the file.
    """
    metadata = pq.ParquetFile(path).metadata
    wanted = None if columns is None else set(columns)
    total = 0
    for group in range(metadata.num_row_groups):
        row_group = metadata.row_group(group)
        for index in range(row_group.num_columns):
            column = row_group.column(index)
            top_level = column.path_in_schema.split(".")[0]
            if wanted is None or top_level in wanted:
                total += column.total_compressed_size
    return total + metadata.serialized_size + PARQUET_FRAMING_BYTES


def _parquet_streaming(path: Path, layout: str, dataset) -> dict[str, float]:  # noqa: ANN001
    """Measure the footer and the bytes needed to read 8 features of a Parquet file."""
    wanted = (
        dataset.feature_names[:STREAM_FEATURES] if layout == "wide" else ["features"]
    )
    return {
        "footer_bytes": float(pq.ParquetFile(path).metadata.serialized_size),
        "bytes_8_features": float(parquet_read_bytes(path, wanted)),
    }


def scaling_table(
    raw: pd.DataFrame,
    streaming: dict[tuple[str, str, str], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Return the measured size, write time, and read time of each layout."""
    rows = []
    keys = raw[["backend", "layout", "profile"]].drop_duplicates()
    for backend, layout, profile in keys.itertuples(index=False, name=None):
        group = raw[
            (raw["backend"] == backend)
            & (raw["layout"] == layout)
            & (raw["profile"] == profile)
        ]
        write = group[group["operation"] == "write"]
        matrix = group[group["operation"] == "matrix_materialization"]
        rows.append(
            {
                "backend": backend,
                "layout": layout,
                "profile": profile,
                "rows": int(group["rows"].iloc[0]),
                "dimensions": int(group["dimensions"].iloc[0]),
                "size_bytes": float(group["artifact_bytes"].max()),
                "write_seconds": float(write["elapsed_seconds"].median()),
                "matrix_seconds": float(matrix["elapsed_seconds"].median()),
                "footer_bytes": (streaming or {})
                .get((backend, layout, profile), {})
                .get("footer_bytes", float("nan")),
                "bytes_8_features": (streaming or {})
                .get((backend, layout, profile), {})
                .get("bytes_8_features", float("nan")),
            }
        )
    return pd.DataFrame(rows)
