"""Public entry points for array-we-there-yet."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from array_we_there_yet.benchmark import (
    BenchmarkConfig,
    combine_runs,
    run_benchmarks,
    summarize_results,
)
from array_we_there_yet.report import (
    update_readme,
    write_figures,
    write_profile_tables,
    write_ratio_tables,
)


def run(
    *,
    rows: int = 2_000,
    dimensions: str = "256,512,1024,2048,4096,8192",
    measured_repetitions: int = 3,
    warmups: int = 1,
    seed: int = 42,
    output_dir: str = "results",
    artifact_dir: str = "results/artifacts",
    figure_dir: str = "figures",
    update_readme_file: bool = True,
) -> dict[str, str]:
    """Run benchmarks, summarize results, generate figures, and update README."""
    config = BenchmarkConfig(
        rows=rows,
        dimensions=_parse_dimensions(dimensions),
        measured_repetitions=measured_repetitions,
        warmups=warmups,
        seed=seed,
        output_dir=Path(output_dir),
        artifact_dir=Path(artifact_dir),
        figure_dir=Path(figure_dir),
    )
    run_benchmarks(config)
    return report(
        output_dir=output_dir,
        figure_dir=figure_dir,
        update_readme_file=update_readme_file,
    )


def report(
    *,
    output_dir: str = "results",
    figure_dir: str = "figures",
    update_readme_file: bool = True,
) -> dict[str, str]:
    """Rebuild the summary, tables, figures, and README from saved raw results."""
    output = Path(output_dir)
    summary = summarize_results(pd.read_parquet(output / "raw_results.parquet"), output)
    floor_path = output / "floor_results.parquet"
    floor = (
        summarize_results(pd.read_parquet(floor_path), output, "floor_summary.parquet")
        if floor_path.exists()
        else None
    )
    ratio_tables = write_ratio_tables(summary, output)
    write_profile_tables(summary, output)
    figures = write_figures(summary, Path(figure_dir))
    if update_readme_file:
        update_readme(
            readme_path=Path("README.md"),
            summary=summary,
            figure_paths=figures,
            environment=json.loads(
                (output / "environment.json").read_text(encoding="utf-8")
            ),
            encodings=pd.read_parquet(output / "encodings.parquet"),
            floor=floor,
        )
    return {
        "raw_results": str(output / "raw_results.parquet"),
        "summary": str(output / "summary.parquet"),
        "ratio_summary": str(ratio_tables[0]),
        "figures": str(figure_dir),
    }


def combine(
    *,
    inputs: str,
    output_dir: str = "results",
    figure_dir: str = "figures",
    update_readme_file: bool = True,
    allow_dirty: bool = False,
) -> dict[str, str]:
    """Pool the results of several runs of the same code, then rebuild the report."""
    directories = [Path(item.strip()) for item in inputs.split(",") if item.strip()]
    combine_runs(directories, Path(output_dir), allow_dirty=allow_dirty)
    return report(
        output_dir=output_dir,
        figure_dir=figure_dir,
        update_readme_file=update_readme_file,
    )


def _parse_dimensions(dimensions: object) -> tuple[int, ...]:
    """Parse dimensions from CLI strings, integers, and Fire tuple values."""
    if isinstance(dimensions, int):
        return (dimensions,)
    if isinstance(dimensions, (list, tuple)):
        return tuple(int(value) for value in dimensions)
    dimensions_text = str(dimensions)
    for character in "()[]":
        dimensions_text = dimensions_text.replace(character, "")
    return tuple(int(value.strip()) for value in dimensions_text.split(",") if value)
