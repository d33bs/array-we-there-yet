"""Tests for pooling several benchmark runs and rebuilding the report."""

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from array_we_there_yet.benchmark import (
    BenchmarkConfig,
    combine_runs,
    run_benchmarks,
)
from array_we_there_yet.main import _parse_inputs, combine, report


def _raw(commit: str, seconds: float) -> pd.DataFrame:
    rows = []
    for iteration in range(3):
        rows.append(
            {
                "backend": "parquet",
                "layout": "wide",
                "dataset": "synthetic",
                "rows": 10,
                "dimensions": 4,
                "dtype": "float32",
                "operation": "full_read",
                "operation_parameter": "all",
                "iteration": iteration,
                "elapsed_seconds": seconds + iteration * 0.001,
                "cpu_seconds": seconds,
                "artifact_bytes": 100,
                "peak_memory_bytes": None,
                "threads": 1,
                "compression": "snappy",
                "profile": "default",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "git_commit": commit,
                "calls": 1,
            }
        )
    return pd.DataFrame(rows)


def _run_directory(root: Path, name: str, commit: str, seconds: float) -> Path:
    directory = root / name
    directory.mkdir()
    _raw(commit, seconds).to_parquet(directory / "raw_results.parquet", index=False)
    _raw(commit, seconds).assign(backend="numpy", layout="npy").to_parquet(
        directory / "floor_results.parquet", index=False
    )
    pd.DataFrame(
        [
            {
                "backend": "parquet",
                "layout": "wide",
                "profile": "default",
                "dimensions": 4,
                "encoding": "snappy",
            }
        ]
    ).to_parquet(directory / "encodings.parquet", index=False)
    (directory / "environment.json").write_text(
        json.dumps({"git_commit": commit, "python": "3.11"}), encoding="utf-8"
    )
    return directory


def test_combine_runs_keeps_every_row_and_numbers_the_runs(tmp_path: Path) -> None:
    """Pooled results hold all repetitions, labeled by run."""
    first = _run_directory(tmp_path, "a", "abc1234", 1.0)
    second = _run_directory(tmp_path, "b", "abc1234", 1.1)
    output = tmp_path / "pooled"

    raw = combine_runs([first, second], output)

    pooled_rows = 6
    assert len(raw) == pooled_rows
    assert set(raw["run"]) == {1, 2}
    assert len(pd.read_parquet(output / "floor_results.parquet")) == pooled_rows
    environment = json.loads((output / "environment.json").read_text())
    assert environment["runs"] == len(set(raw["run"]))
    assert environment["git_commit"] == "abc1234"
    assert (output / "encodings.parquet").exists()


def test_combine_runs_refuses_runs_from_different_code(tmp_path: Path) -> None:
    """Runs from different commits must not be pooled."""
    first = _run_directory(tmp_path, "a", "abc1234", 1.0)
    second = _run_directory(tmp_path, "b", "def5678", 1.0)

    with pytest.raises(ValueError, match="different code versions"):
        combine_runs([first, second], tmp_path / "pooled")


def test_combine_runs_refuses_uncommitted_code_unless_allowed(tmp_path: Path) -> None:
    """A dirty commit hash does not identify the code, so it is refused."""
    first = _run_directory(tmp_path, "a", "abc1234-dirty", 1.0)
    second = _run_directory(tmp_path, "b", "abc1234-dirty", 1.0)

    with pytest.raises(ValueError, match="uncommitted"):
        combine_runs([first, second], tmp_path / "pooled")

    raw = combine_runs([first, second], tmp_path / "pooled", allow_dirty=True)
    assert len(raw) == len(_raw("x", 1.0)) * 2


def test_combine_runs_needs_at_least_two_runs(tmp_path: Path) -> None:
    """Pooling one run is a mistake."""
    only = _run_directory(tmp_path, "a", "abc1234", 1.0)

    with pytest.raises(ValueError, match="at least two"):
        combine_runs([only], tmp_path / "pooled")


def test_report_and_combine_rebuild_the_summary_from_raw_results(
    tmp_path: Path,
) -> None:
    """Pooling two copies of one run doubles the repetitions in the summary."""
    config = BenchmarkConfig(
        rows=6,
        dimensions=(4,),
        measured_repetitions=2,
        warmups=0,
        random_row_count=2,
        feature_projection_count=2,
        output_dir=tmp_path / "one",
        artifact_dir=tmp_path / "one" / "artifacts",
    )
    run_benchmarks(config)
    shutil.copytree(config.output_dir, tmp_path / "two")
    (tmp_path / "two" / "artifacts").rmdir()

    single = report(
        output_dir=str(config.output_dir),
        site_dir=str(tmp_path / "site_one"),
    )
    pooled = combine(
        inputs=f"{config.output_dir},{tmp_path / 'two'}",
        output_dir=str(tmp_path / "pooled"),
        site_dir=str(tmp_path / "site_pooled"),
        allow_dirty=True,
    )

    one = pd.read_parquet(single["summary"])
    both = pd.read_parquet(pooled["summary"])
    assert set(one["repetitions"]) == {2}
    assert set(both["repetitions"]) == {4}
    assert (tmp_path / "pooled" / "floor_summary.parquet").exists()


def test_combine_runs_takes_the_code_version_from_the_raw_results(
    tmp_path: Path,
) -> None:
    """The pooled environment names the commit that the raw results recorded."""
    first = _run_directory(tmp_path, "a", "abc1234", 1.0)
    second = _run_directory(tmp_path, "b", "abc1234", 1.1)
    (first / "environment.json").write_text(
        json.dumps({"git_commit": "abc1234-dirty", "python": "3.11"}),
        encoding="utf-8",
    )

    combine_runs([first, second], tmp_path / "pooled")

    environment = json.loads((tmp_path / "pooled" / "environment.json").read_text())
    assert environment["git_commit"] == "abc1234"


def test_parse_inputs_accepts_what_the_command_line_can_produce() -> None:
    """Fire turns `a,b` into a tuple, so both text and sequences must work."""
    expected = [Path("a"), Path("b")]

    assert _parse_inputs("a,b") == expected
    assert _parse_inputs("a, b") == expected
    assert _parse_inputs(("a", "b")) == expected
    assert _parse_inputs(["a", "b"]) == expected
    assert _parse_inputs("a") == [Path("a")]
