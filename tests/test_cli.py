"""Tests for the CLI module."""

import inspect
import subprocess
from pathlib import Path

from array_we_there_yet.cli import ArrayWeThereYetCLI
from array_we_there_yet.main import _parse_dimensions
from array_we_there_yet.main import run as run_benchmark


def test_default_feature_counts_include_three_larger_sizes() -> None:
    """The default benchmark includes the larger feature-count options."""
    expected = "256,512,1024,2048,4096,8192"
    assert inspect.signature(run_benchmark).parameters["dimensions"].default == expected
    assert (
        inspect.signature(ArrayWeThereYetCLI.run).parameters["dimensions"].default
        == expected
    )
    assert _parse_dimensions(expected) == (
        256,
        512,
        1024,
        2048,
        4096,
        8192,
    )


def test_run_cli_smoke(tmp_path: Path) -> None:
    """The CLI can run a small benchmark without updating README."""
    output = subprocess.run(
        [
            "uv",
            "run",
            "array-we-there-yet",
            "run",
            "--rows=8",
            "--dimensions=4",
            "--measured_repetitions=1",
            "--warmups=0",
            f"--output_dir={tmp_path / 'results'}",
            f"--artifact_dir={tmp_path / 'results' / 'artifacts'}",
            f"--site_dir={tmp_path / 'site'}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "raw_results" in output.stdout
    assert "parquet_performance" not in output.stdout
