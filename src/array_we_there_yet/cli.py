"""Command-line interface for array-we-there-yet."""

from __future__ import annotations

import fire

from array_we_there_yet.main import run


class ArrayWeThereYetCLI:
    """CLI commands for benchmark runs."""

    def run(
        self,
        rows: int = 2_000,
        dimensions: str = "16,64,128",
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


def trigger() -> None:
    """Start the Fire command-line interface."""
    fire.Fire(ArrayWeThereYetCLI)
