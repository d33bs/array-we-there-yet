"""Command-line interface for array-we-there-yet."""

from __future__ import annotations

import fire

from array_we_there_yet.main import combine as combine_runs_and_report
from array_we_there_yet.main import report as report_results
from array_we_there_yet.main import run
from array_we_there_yet.main import scaling as run_scaling
from array_we_there_yet.main import sweep as run_sweep


class ArrayWeThereYetCLI:
    """CLI commands for benchmark runs."""

    def run(
        self,
        rows: int = 2_000,
        dimensions: str = "256,512,1024,2048,4096,8192",
        measured_repetitions: int = 3,
        warmups: int = 1,
        output_dir: str = "results",
        artifact_dir: str = "results/artifacts",
        site_dir: str = "site",
    ) -> dict[str, str]:
        """Run the benchmark suite."""
        return run(
            rows=rows,
            dimensions=dimensions,
            measured_repetitions=measured_repetitions,
            warmups=warmups,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            site_dir=site_dir,
        )

    def report(
        self,
        output_dir: str = "results",
        site_dir: str = "site",
    ) -> dict[str, str]:
        """Rebuild the tables and the report page from saved raw results."""
        return report_results(
            output_dir=output_dir,
            site_dir=site_dir,
        )

    def combine(
        self,
        inputs: str | tuple[str, ...],
        output_dir: str = "results",
        site_dir: str = "site",
        allow_dirty: bool = False,
    ) -> dict[str, str]:
        """Pool runs of the same code. Give the run directories as a comma list."""
        return combine_runs_and_report(
            inputs=inputs,
            output_dir=output_dir,
            site_dir=site_dir,
            allow_dirty=allow_dirty,
        )

    def sweep(
        self,
        row_counts: str | tuple[int, ...] = "2000,20000,200000",
        dimensions: int = 1024,
        output_dir: str = "results",
        site_dir: str = "site",
    ) -> dict[str, str]:
        """Run the benchmark at several row counts, then rebuild the report."""
        return run_sweep(
            row_counts=row_counts,
            dimensions=dimensions,
            output_dir=output_dir,
            site_dir=site_dir,
        )

    def scaling(
        self,
        output_dir: str = "results",
        site_dir: str = "site",
    ) -> dict[str, str]:
        """Write and read a real 1.5 GB CSV wide file, then rebuild the report."""
        return run_scaling(
            output_dir=output_dir,
            site_dir=site_dir,
        )


def trigger() -> None:
    """Start the Fire command-line interface."""
    fire.Fire(ArrayWeThereYetCLI)
