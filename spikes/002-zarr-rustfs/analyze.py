"""Validate and pool the metered RustFS trials; keep local-disk context separate."""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TRIALS = (HERE / "measurements.csv", HERE / "run2_measurements.csv")
LOCAL = HERE.parent / "001-zarr-chunk-shapes" / "pooled_summary.csv"
GROUP = ["rows", "features", "shape", "operation", "latency_ms"]
SHAPES = ("baseline", "narrow_features")
OPERATIONS = ("matrix", "rows", "features")
LATENCIES = (0, 20)
REPS = 7
EXPECTED_GROUPS = len(SHAPES) * len(OPERATIONS) * len(LATENCIES)


def main() -> None:
    frames = [pd.read_csv(path) for path in TRIALS]
    for frame in frames:
        assert not frame.isna().any().any(), "Missing measurement fields"
        assert len(frame) == EXPECTED_GROUPS * REPS, "Incomplete trial"
        counts = frame.groupby(GROUP).size()
        assert len(counts) == EXPECTED_GROUPS, "Missing workload/shape cell"
        assert (counts == REPS).all(), "Unequal repetitions"
        assert (frame["get_requests"] >= frame["chunk_gets"]).all()
        assert (frame["response_bytes"] >= frame["chunk_bytes"]).all()
        assert (
            frame["get_requests"]
            == frame["chunk_gets"] + frame["metadata_gets"] + frame["list_gets"]
        ).all(), "GET breakdown does not add up"
    pooled = pd.concat(frames, ignore_index=True)
    summary = pooled.groupby(GROUP, as_index=False).agg(
        samples=("wall_seconds", "size"),
        median_ms=("wall_seconds", lambda x: 1000 * x.median()),
        q25_ms=("wall_seconds", lambda x: 1000 * x.quantile(0.25)),
        q75_ms=("wall_seconds", lambda x: 1000 * x.quantile(0.75)),
        get_requests=("get_requests", "median"),
        head_requests=("head_requests", "median"),
        chunk_gets=("chunk_gets", "median"),
        metadata_gets=("metadata_gets", "median"),
        list_gets=("list_gets", "median"),
        response_mb=("response_bytes", lambda x: x.median() / 1e6),
        chunk_mb=("chunk_bytes", lambda x: x.median() / 1e6),
        artifact_mb=("artifact_bytes", lambda x: x.median() / 1e6),
    )
    assert len(summary) == EXPECTED_GROUPS
    assert (summary["samples"] == REPS * len(TRIALS)).all()
    summary.to_csv(HERE / "pooled_summary.csv", index=False)
    for latency in LATENCIES:
        frame = summary[summary["latency_ms"] == latency]
        print(f"\nRustFS with {latency} ms added per HTTP request, pooled median:")
        table = frame.pivot(index="operation", columns="shape", values="median_ms")
        table["narrow / baseline"] = table["narrow_features"] / table["baseline"]
        print(table.round(2).to_string())
        columns = [
            "shape",
            "operation",
            "get_requests",
            "head_requests",
            "chunk_gets",
            "response_mb",
        ]
        print(frame[columns].round(2).to_string(index=False))
    if LOCAL.exists():
        local = pd.read_csv(LOCAL)
        local = local[
            (local["rows"] == pooled["rows"].iloc[0])
            & (local["features"] == pooled["features"].iloc[0])
            & local["shape"].isin(SHAPES)
            & local["operation"].isin(OPERATIONS)
        ]
        local[["shape", "operation", "median_ms"]].to_csv(
            HERE / "local_disk_context.csv", index=False
        )
        print(
            "\nSeparate local-disk measurements saved as context, not a paired trial."
        )
    print(f"Saved {HERE / 'pooled_summary.csv'}")


if __name__ == "__main__":
    main()
