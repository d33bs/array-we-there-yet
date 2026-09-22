"""Tests for the CLI module."""

import subprocess
from pathlib import Path


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
            f"--figure_dir={tmp_path / 'figures'}",
            "--update_readme_file=False",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "raw_results" in output.stdout
