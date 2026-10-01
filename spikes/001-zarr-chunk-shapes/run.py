"""One-off Zarr chunk-shape sweep using the project's measured read paths.

Run: uv run --frozen python spikes/001-zarr-chunk-shapes/run.py
Small check: add --smoke. Artifacts are temporary; CSV/JSON stay beside this file.
"""

# One-off measurement helpers carry shared experiment context.
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import random
import shutil
import statistics
import tempfile
import time
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd
import zarr

from array_we_there_yet.benchmark import (
    Artifact,
    _artifact_size,
    _calls_per_sample,
    _read_zarr_matrix,
    _read_zarr_matrix_features,
    _read_zarr_matrix_rows,
    _selection,
    _timed_repeated,
    _write_zarr_metadata,
    _zarr_compressor,
    apply_thread_limits,
)
from array_we_there_yet.data import BenchmarkDataset, make_synthetic_dataset

SHAPES = {
    "baseline": (1024, 1024),
    "short_rows": (128, 1024),
    "narrow_features": (1024, 128),
    "balanced": (256, 256),
}
CASES = ((20_000, 1024), (2_000, 8192))
READS = ("matrix", "rows", "features")
COLUMNS = (
    "rows",
    "features",
    "shape",
    "row_chunk",
    "feature_chunk",
    "operation",
    "repetition",
    "wall_seconds",
    "cpu_seconds",
    "calls",
    "artifact_bytes",
    "touched_chunks",
    "selected_chunk_bytes",
    "decoded_bytes",
)


def chunk_cost(
    path: Path,
    rows: int,
    features: int,
    chunks: tuple[int, int],
    operation: str,
    selected_rows: np.ndarray,
    selected_features: np.ndarray,
) -> tuple[int, int, int]:
    """Count selected physical chunks, their stored bytes, and decoded values."""
    row_chunk, feature_chunk = chunks
    row_blocks = range(math.ceil(rows / row_chunk))
    feature_blocks = range(math.ceil(features / feature_chunk))
    if operation == "rows":
        row_blocks = sorted({int(index // row_chunk) for index in selected_rows})
    elif operation == "features":
        feature_blocks = sorted(
            {int(index // feature_chunk) for index in selected_features}
        )
    touched = [(r, c) for r in row_blocks for c in feature_blocks]
    chunk_bytes = 0
    decoded_bytes = 0
    for row, column in touched:
        chunk_file = path / "features" / "c" / str(row) / str(column)
        chunk_bytes += chunk_file.stat().st_size
        decoded_bytes += (
            min(row_chunk, rows - row * row_chunk)
            * min(feature_chunk, features - column * feature_chunk)
            * np.dtype("float32").itemsize
        )
    return len(touched), chunk_bytes, decoded_bytes


def record(
    writer: csv.DictWriter,
    file: TextIO,
    *,
    dataset: BenchmarkDataset,
    name: str,
    chunks: tuple[int, int],
    operation: str,
    repetition: int,
    wall: float,
    cpu: float,
    calls: int,
    artifact: Artifact,
    selected_rows: np.ndarray,
    selected_features: np.ndarray,
) -> None:
    if operation == "write":
        touched, chunk_bytes, decoded_bytes = (0, 0, 0)
    else:
        touched, chunk_bytes, decoded_bytes = chunk_cost(
            artifact.path,
            dataset.rows,
            dataset.dimensions,
            chunks,
            operation,
            selected_rows,
            selected_features,
        )
    writer.writerow(
        {
            "rows": dataset.rows,
            "features": dataset.dimensions,
            "shape": name,
            "row_chunk": chunks[0],
            "feature_chunk": chunks[1],
            "operation": operation,
            "repetition": repetition,
            "wall_seconds": wall / calls,
            "cpu_seconds": cpu / calls,
            "calls": calls,
            "artifact_bytes": artifact.bytes,
            "touched_chunks": touched,
            "selected_chunk_bytes": chunk_bytes,
            "decoded_bytes": decoded_bytes,
        }
    )
    file.flush()


def run_case(
    rows: int,
    features: int,
    scratch: Path,
    writer: csv.DictWriter,
    file: TextIO,
    *,
    smoke: bool,
) -> None:
    dataset = make_synthetic_dataset(rows=rows, dimensions=features, seed=42)
    selected_rows = _selection(rows, 128, seed=42 + features)
    selected_features = _selection(features, 8, seed=42 + features + 1)
    calls = {
        "matrix": lambda artifact: _read_zarr_matrix(artifact, dataset),
        "rows": lambda artifact: _read_zarr_matrix_rows(
            artifact, dataset, selected_rows
        ),
        "features": lambda artifact: _read_zarr_matrix_features(
            artifact, dataset, selected_features
        ),
    }
    expected = {
        "matrix": dataset.matrix,
        "rows": dataset.matrix[selected_rows, :],
        "features": dataset.matrix[:, selected_features],
    }
    rng = random.Random(42 + rows + features)
    artifacts: dict[str, Artifact] = {}
    print(
        f"CASE {rows}x{features}: {dataset.matrix.nbytes / 1e6:.1f} MB raw",
        flush=True,
    )

    # Shuffle shape order on every write round; exclude the first as warmup.
    for repetition in range(1 if smoke else 4):
        names = list(SHAPES)
        rng.shuffle(names)
        for name in names:
            chunks = SHAPES[name]
            path = scratch / f"{rows}x{features}-{name}-{repetition}"
            wall_start = time.perf_counter()
            cpu_start = time.process_time()
            root = zarr.open_group(str(path), mode="w")
            _write_zarr_metadata(root, dataset, layout="zarr_matrix")
            root.create_array(
                "features",
                data=dataset.matrix,
                chunks=chunks,
                **_zarr_compressor(compact=False),
            )
            cpu = time.process_time() - cpu_start
            wall = time.perf_counter() - wall_start
            artifact = Artifact(path=path, bytes=_artifact_size(path))
            if name in artifacts:
                shutil.rmtree(artifacts[name].path)
            artifacts[name] = artifact
            if repetition:
                record(
                    writer,
                    file,
                    dataset=dataset,
                    name=name,
                    chunks=chunks,
                    operation="write",
                    repetition=repetition,
                    wall=wall,
                    cpu=cpu,
                    calls=1,
                    artifact=artifact,
                    selected_rows=selected_rows,
                    selected_features=selected_features,
                )
            print(
                f"  write {name} rep={repetition}: {wall:.3f}s, "
                f"{artifact.bytes / 1e6:.1f} MB",
                flush=True,
            )

    calibration: dict[tuple[str, str], int] = {}
    jobs = [(name, op) for name in SHAPES for op in READS]
    rng.shuffle(jobs)
    for name, operation in jobs:
        result, wall, _ = _timed_repeated(lambda: calls[operation](artifacts[name]), 1)
        np.testing.assert_array_equal(result, expected[operation])
        calibration[name, operation] = _calls_per_sample(wall)
    for repetition in range(1, 3 if smoke else 8):
        rng.shuffle(jobs)
        for name, operation in jobs:
            result, wall, cpu = _timed_repeated(
                lambda: calls[operation](artifacts[name]), calibration[name, operation]
            )
            np.testing.assert_array_equal(result, expected[operation])
            record(
                writer,
                file,
                dataset=dataset,
                name=name,
                chunks=SHAPES[name],
                operation=operation,
                repetition=repetition,
                wall=wall,
                cpu=cpu,
                calls=calibration[name, operation],
                artifact=artifacts[name],
                selected_rows=selected_rows,
                selected_features=selected_features,
            )
        print(f"  read round {repetition} complete", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="small validation run")
    parser.add_argument("--label", choices=("run2",), help="separate second trial")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    output.mkdir(parents=True, exist_ok=True)
    label = f"{args.label or 'smoke'}_" if args.label or args.smoke else ""
    csv_path = output / f"{label}measurements.csv"
    env_path = output / f"{label}environment.json"
    limits = apply_thread_limits(1)
    environment = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "zarr": zarr.__version__,
        "numpy": np.__version__,
        "thread_limits": limits,
        "seed": 42,
        "row_selection": 128,
        "feature_selection": 8,
        "write_repetitions": 0 if args.smoke else 3,
        "read_repetitions": 2 if args.smoke else 7,
        "note": "Warm OS page cache; no S3. write measures metadata + feature array. "
        "read operations call production Zarr readers including open_group; "
        "validation is outside the timer. selected_chunk_bytes is stored "
        "chunk payload, not measured disk I/O.",
    }
    env_path.write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    with (
        tempfile.TemporaryDirectory(
            prefix="zarr-chunk-sweep-", dir=Path.home() / ".hermes/cache/scratch"
        ) as tmp,
        csv_path.open("w", newline="", encoding="utf-8") as file,
    ):
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        cases = ((256, 256),) if args.smoke else CASES
        for rows, features in cases:
            run_case(rows, features, Path(tmp), writer, file, smoke=args.smoke)
    print(f"Saved {csv_path} and {env_path}", flush=True)
    if not args.smoke:
        frame = pd.read_csv(csv_path)
        summary = frame.groupby(
            [
                "rows",
                "features",
                "shape",
                "row_chunk",
                "feature_chunk",
                "operation",
            ],
            as_index=False,
        ).agg(
            median_ms=("wall_seconds", lambda x: 1000 * statistics.median(x)),
            median_cpu_ms=("cpu_seconds", lambda x: 1000 * statistics.median(x)),
            artifact_mb=("artifact_bytes", lambda x: statistics.median(x) / 1e6),
            chunks=("touched_chunks", "first"),
            stored_mb=("selected_chunk_bytes", lambda x: x.iloc[0] / 1e6),
            decoded_mb=("decoded_bytes", lambda x: x.iloc[0] / 1e6),
        )
        result = output / f"{label}summary.csv"
        summary.to_csv(result, index=False)
        print(summary.round(3).to_string(index=False))
        print(f"Saved {result}", flush=True)


if __name__ == "__main__":
    main()
