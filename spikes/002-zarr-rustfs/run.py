"""Measure Zarr matrix reads over a metered RustFS S3-compatible endpoint.

Run: uv run --frozen python spikes/002-zarr-rustfs/run.py [--smoke | --label run2]
Only the benchmark-owned bench/002/ prefix is replaced on the local RustFS.
"""

# The disposable measurement harness passes shared experiment context.
# ruff: noqa: PLR0913, PLR0917

from __future__ import annotations

import argparse
import csv
import json
import platform
import random
import socket
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, TextIO

import numpy as np
import pandas as pd
import s3fs
import zarr
from proxy import Meter, Proxy
from zarr.core.sync import sync
from zarr.storage import FsspecStore

from array_we_there_yet.benchmark import (
    _selection,
    _write_zarr_metadata,
    _zarr_compressor,
    apply_thread_limits,
)
from array_we_there_yet.data import BenchmarkDataset, make_synthetic_dataset

ENDPOINT = "http://127.0.0.1:9000"
ACCESS = "b"  # Deliberately nonsecret; RustFS listens on localhost only.
SECRET = "b"
BUCKET = "bench"
SHAPES = {"baseline": (1024, 1024), "narrow_features": (1024, 128)}
CASES = ((2000, 8192),)
OPERATIONS = ("matrix", "rows", "features")
LATENCIES_MS = (0, 20)
COLUMNS = (
    "rows",
    "features",
    "shape",
    "row_chunk",
    "feature_chunk",
    "operation",
    "latency_ms",
    "repetition",
    "wall_seconds",
    "cpu_seconds",
    "artifact_bytes",
    "get_requests",
    "head_requests",
    "chunk_gets",
    "metadata_gets",
    "list_gets",
    "response_bytes",
    "chunk_bytes",
)


def filesystem(endpoint: str, *, asynchronous: bool) -> s3fs.S3FileSystem:
    """Keep async Zarr clients on Zarr's loop; use a separate setup client."""
    return s3fs.S3FileSystem(
        key=ACCESS,
        secret=SECRET,
        asynchronous=asynchronous,
        client_kwargs={"endpoint_url": endpoint},
        use_listings_cache=False,
        skip_instance_cache=True,
    )


def close_zarr_client(fs: s3fs.S3FileSystem) -> None:
    """Close the async aiobotocore session on the loop that opened it."""
    if fs._s3creator is not None:
        sync(fs._s3creator.__aexit__(None, None, None))


def open_group(
    fs: s3fs.S3FileSystem, path: str, *, mode: Literal["r", "w"]
) -> zarr.Group:
    store = FsspecStore(fs, read_only=mode == "r", path=path)
    group = zarr.open_group(store, mode=mode)
    if not isinstance(group, zarr.Group):
        message = f"{path} was not a Zarr group"
        raise TypeError(message)
    return group


def features_array(group: zarr.Group) -> zarr.Array:
    array = group["features"]
    if not isinstance(array, zarr.Array):
        message = "features was not a Zarr array"
        raise TypeError(message)
    return array


def write_artifact(
    setup_fs: s3fs.S3FileSystem,
    write_fs: s3fs.S3FileSystem,
    path: str,
    dataset: BenchmarkDataset,
    chunks: tuple[int, int],
) -> int:
    """Write the same metadata and default-compressed matrix as the local spike."""
    if setup_fs.exists(path):
        setup_fs.rm(path, recursive=True)
    group = open_group(write_fs, path, mode="w")
    _write_zarr_metadata(group, dataset, layout="zarr_matrix")
    group.create_array(
        "features",
        data=dataset.matrix,
        chunks=chunks,
        **_zarr_compressor(compact=False),
    )
    # Include all stored objects, metadata and chunks. Never timed.
    objects = setup_fs.find(path, detail=True)
    if not isinstance(objects, dict):
        message = "S3 object listing returned no size metadata"
        raise TypeError(message)
    return sum(int(info["size"]) for info in objects.values())


def read_operation(
    fs: s3fs.S3FileSystem,
    path: str,
    operation: str,
    rows: np.ndarray,
    selected_features: np.ndarray,
) -> np.ndarray:
    """Open per operation, like the main benchmark's Zarr readers."""
    array = features_array(open_group(fs, path, mode="r"))
    if operation == "matrix":
        return np.asarray(array[:], dtype=np.float32)
    if operation == "rows":
        return np.asarray(array.oindex[rows, :], dtype=np.float32)
    if operation == "features":
        return np.asarray(array[:, selected_features], dtype=np.float32)
    msg = f"Unknown operation: {operation}"
    raise ValueError(msg)


def write_measurement(
    writer: csv.DictWriter,
    file: TextIO,
    *,
    dataset: BenchmarkDataset,
    shape: str,
    operation: str,
    latency_ms: int,
    repetition: int,
    wall: float,
    cpu: float,
    artifact_bytes: int,
    counters: dict[str, int],
) -> None:
    writer.writerow(
        {
            "rows": dataset.rows,
            "features": dataset.dimensions,
            "shape": shape,
            "row_chunk": SHAPES[shape][0],
            "feature_chunk": SHAPES[shape][1],
            "operation": operation,
            "latency_ms": latency_ms,
            "repetition": repetition,
            "wall_seconds": wall,
            "cpu_seconds": cpu,
            "artifact_bytes": artifact_bytes,
            **counters,
        }
    )
    file.flush()


def run_case(
    rows: int,
    dimensions: int,
    setup_fs: s3fs.S3FileSystem,
    write_fs: s3fs.S3FileSystem,
    read_fs: s3fs.S3FileSystem,
    meter: Meter,
    writer: csv.DictWriter,
    file: TextIO,
    *,
    smoke: bool,
    run2: bool,
) -> None:
    dataset = make_synthetic_dataset(rows=rows, dimensions=dimensions, seed=42)
    selected_rows = _selection(rows, min(rows, 128), seed=42 + dimensions)
    selected_features = _selection(dimensions, min(dimensions, 8), seed=43 + dimensions)
    expected = {
        "matrix": dataset.matrix,
        "rows": dataset.matrix[selected_rows, :],
        "features": dataset.matrix[:, selected_features],
    }
    artifacts: dict[str, tuple[str, int]] = {}
    print(
        f"CASE {rows}x{dimensions}: {dataset.matrix.nbytes / 1e6:.1f} MB raw",
        flush=True,
    )
    for shape, chunks in SHAPES.items():
        path = f"{BUCKET}/002/{rows}x{dimensions}/{shape}"
        stored_bytes = write_artifact(setup_fs, write_fs, path, dataset, chunks)
        artifacts[shape] = path, stored_bytes
        print(f"  wrote {shape}: {stored_bytes / 1e6:.2f} MB", flush=True)

    for latency_ms in (0,) if smoke else LATENCIES_MS:
        jobs = [(name, op) for name in SHAPES for op in OPERATIONS]
        rng = random.Random(42 + rows + dimensions + latency_ms + int(run2))
        meter.reset(latency_ms=latency_ms)
        for name, operation in jobs:
            result = read_operation(
                read_fs, artifacts[name][0], operation, selected_rows, selected_features
            )
            np.testing.assert_array_equal(result, expected[operation])
        rounds = 2 if smoke else 7
        for repetition in range(rounds):
            rng.shuffle(jobs)
            for name, operation in jobs:
                path, stored_bytes = artifacts[name]
                meter.reset(latency_ms=latency_ms)
                wall_start = time.perf_counter()
                cpu_start = time.process_time()
                result = read_operation(
                    read_fs, path, operation, selected_rows, selected_features
                )
                cpu = time.process_time() - cpu_start
                wall = time.perf_counter() - wall_start
                counters = meter.snapshot()
                np.testing.assert_array_equal(result, expected[operation])
                assert counters["chunk_gets"] > 0, "No chunk GETs counted"
                assert counters["response_bytes"] >= counters["chunk_bytes"]
                write_measurement(
                    writer,
                    file,
                    dataset=dataset,
                    shape=name,
                    operation=operation,
                    latency_ms=latency_ms,
                    repetition=repetition,
                    wall=wall,
                    cpu=cpu,
                    artifact_bytes=stored_bytes,
                    counters=counters,
                )
            print(
                f"  {latency_ms} ms latency, read round {repetition + 1}/{rounds}",
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="tiny integration check")
    parser.add_argument("--label", choices=("run2",), help="second independent trial")
    args = parser.parse_args()
    output = Path(__file__).resolve().parent
    prefix = f"{args.label or 'smoke'}_" if args.label or args.smoke else ""
    socket.setdefaulttimeout(90)
    with socket.create_connection(("127.0.0.1", 9000), timeout=5):
        pass
    limits = apply_thread_limits(1)
    image = subprocess.run(
        ["docker", "inspect", "rustfs-s3-bench", "--format", "{{.Image}}"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    meter = Meter()
    with Proxy(meter) as proxy:
        environment = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], text=True
            ).strip(),
            "platform": platform.platform(),
            "python": platform.python_version(),
            "zarr": zarr.__version__,
            "s3fs": s3fs.__version__,
            "rustfs_image_id": image,
            "upstream": ENDPOINT,
            "read_endpoint": proxy.endpoint,
            "latencies_ms": [0] if args.smoke else list(LATENCIES_MS),
            "thread_limits": limits,
            "read_repetitions": 2 if args.smoke else 7,
            "note": "Warm localhost RustFS; proxy delays every HTTP request. "
            "Response bytes exclude headers; no WAN or AWS S3.",
        }
        (output / f"{prefix}environment.json").write_text(
            json.dumps(environment, indent=2) + "\n", encoding="utf-8"
        )
        setup_fs = filesystem(ENDPOINT, asynchronous=False)
        write_fs = filesystem(ENDPOINT, asynchronous=True)
        read_fs = filesystem(proxy.endpoint, asynchronous=True)
        try:
            if BUCKET not in [p.rstrip("/").split("/")[-1] for p in setup_fs.ls("/")]:
                setup_fs.mkdir(BUCKET)
            cases = ((128, 256),) if args.smoke else CASES
            measurements = output / f"{prefix}measurements.csv"
            with measurements.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=COLUMNS)
                writer.writeheader()
                for rows, dimensions in cases:
                    run_case(
                        rows,
                        dimensions,
                        setup_fs,
                        write_fs,
                        read_fs,
                        meter,
                        writer,
                        file,
                        smoke=args.smoke,
                        run2=bool(args.label),
                    )
        finally:
            close_zarr_client(read_fs)
            close_zarr_client(write_fs)
        frame = pd.read_csv(measurements)
        summary = frame.groupby(
            ["rows", "features", "shape", "operation", "latency_ms"],
            as_index=False,
        ).agg(
            median_ms=("wall_seconds", lambda s: 1000 * statistics.median(s)),
            get_requests=("get_requests", "median"),
            head_requests=("head_requests", "median"),
            chunk_gets=("chunk_gets", "median"),
            response_mb=("response_bytes", lambda s: statistics.median(s) / 1e6),
            artifact_mb=("artifact_bytes", lambda s: statistics.median(s) / 1e6),
        )
        summary.to_csv(output / f"{prefix}summary.csv", index=False)
        print(summary.round(2).to_string(index=False))
        print(f"Saved {measurements}", flush=True)


if __name__ == "__main__":
    main()
