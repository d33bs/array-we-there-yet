"""Public entry points for array-we-there-yet."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from array_we_there_yet.benchmark import (
    BenchmarkConfig,
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
    parsed_dimensions = _parse_dimensions(dimensions)
    config = BenchmarkConfig(
        rows=rows,
        dimensions=parsed_dimensions,
        measured_repetitions=measured_repetitions,
        warmups=warmups,
        seed=seed,
        output_dir=Path(output_dir),
        artifact_dir=Path(artifact_dir),
        figure_dir=Path(figure_dir),
    )
    raw = run_benchmarks(config)
    summary = summarize_results(raw, config.output_dir)
    ratio_tables = write_ratio_tables(summary, config.output_dir)
    write_profile_tables(summary, config.output_dir)
    figures = write_figures(summary, config.figure_dir)
    if update_readme_file:
        environment_path = config.output_dir / "environment.json"
        update_readme(
            readme_path=Path("README.md"),
            summary=summary,
            figure_paths=figures,
            environment=json.loads(environment_path.read_text(encoding="utf-8")),
            encodings=pd.read_parquet(config.output_dir / "encodings.parquet"),
        )
    return {
        "raw_results": str(config.output_dir / "raw_results.parquet"),
        "summary": str(config.output_dir / "summary.parquet"),
        "ratio_summary": str(ratio_tables[0]),
        "figures": str(config.figure_dir),
    }


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
