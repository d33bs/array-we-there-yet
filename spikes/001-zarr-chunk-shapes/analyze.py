"""Validate and pool the two independent Zarr chunk-shape trials."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
KEYS = ["rows", "features", "shape", "row_chunk", "feature_chunk", "operation"]
EXPECTED_ROWS = 192
EXPECTED_GROUPS = 32
WRITE_REPS = 3
READ_REPS = 7
RUNS = 2
frames = [
    pd.read_csv(HERE / "measurements.csv").assign(run=1),
    pd.read_csv(HERE / "run2_measurements.csv").assign(run=2),
]
for frame in frames:
    assert len(frame) == EXPECTED_ROWS, f"Incomplete trial: {len(frame)} records"
    assert not frame.isna().any().any(), "Missing measurements"
    counts = frame.groupby(KEYS).size()
    assert len(counts) == EXPECTED_GROUPS, f"Missing shape/operation: {len(counts)}"
    assert all(
        counts.loc[counts.index.get_level_values("operation") == "write"] == WRITE_REPS
    )
    assert all(
        counts.loc[counts.index.get_level_values("operation") != "write"] == READ_REPS
    )

all_runs = pd.concat(frames, ignore_index=True)
pooled = all_runs.groupby(KEYS, as_index=False).agg(
    median_ms=("wall_seconds", lambda x: x.median() * 1000),
    median_cpu_ms=("cpu_seconds", lambda x: x.median() * 1000),
    artifact_mb=("artifact_bytes", lambda x: x.median() / 1e6),
    touched_chunks=("touched_chunks", "first"),
    selected_chunk_mb=("selected_chunk_bytes", lambda x: x.iloc[0] / 1e6),
    decoded_mb=("decoded_bytes", lambda x: x.iloc[0] / 1e6),
    samples=("wall_seconds", "size"),
)
assert all(pooled.loc[pooled.operation == "write", "samples"] == RUNS * WRITE_REPS)
assert all(pooled.loc[pooled.operation != "write", "samples"] == RUNS * READ_REPS)
pooled.to_csv(HERE / "pooled_summary.csv", index=False)
for (rows, features), group in pooled.groupby(["rows", "features"]):
    print(f"\n{rows:,} rows x {features:,} features (pooled medians, ms):")
    print(
        group.pivot(index="shape", columns="operation", values="median_ms")
        .round(2)
        .to_string()
    )
    baseline = group.set_index(["shape", "operation"])["median_ms"].loc["baseline"]
    ratios = group.assign(
        vs_baseline=lambda df: df.apply(
            lambda row: row.median_ms / baseline.loc[row.operation], axis=1
        )
    )
    print("Relative to baseline (<1 means faster):")
    print(
        ratios.pivot(index="shape", columns="operation", values="vs_baseline")
        .round(2)
        .to_string()
    )
print(f"\nSaved {HERE / 'pooled_summary.csv'}")
