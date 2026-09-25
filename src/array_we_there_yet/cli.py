"""Command-line interface for array-we-there-yet."""

from __future__ import annotations

import fire

from array_we_there_yet.main import combine as combine_runs_and_report
from array_we_there_yet.main import report as report_results
from array_we_there_yet.main import run


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
        figure_dir: str = "figures",
        update_readme_file: bool = True,
    ) -> dict[str, str]:
        """Run the benchmark suite."""
        return run(
            rows=rows,
            dimensions=dimensions,
            measured_repetitions=measured_repetitions,
            warmups=warmups,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
            figure_dir=figure_dir,
            update_readme_file=update_readme_file,
        )

    def report(
        self,
        output_dir: str = "results",
        figure_dir: str = "figures",
        update_readme_file: bool = True,
    ) -> dict[str, str]:
        """Rebuild the tables, figures, and README from saved raw results."""
        return report_results(
            output_dir=output_dir,
            figure_dir=figure_dir,
            update_readme_file=update_readme_file,
        )

    def combine(
        self,
        inputs: str,
        output_dir: str = "results",
        figure_dir: str = "figures",
        update_readme_file: bool = True,
        allow_dirty: bool = False,
    ) -> dict[str, str]:
        """Pool runs of the same code. Give the run directories as a comma list."""
        return combine_runs_and_report(
            inputs=inputs,
            output_dir=output_dir,
            figure_dir=figure_dir,
            update_readme_file=update_readme_file,
            allow_dirty=allow_dirty,
        )


def trigger() -> None:
    """Start the Fire command-line interface."""
    fire.Fire(ArrayWeThereYetCLI)
