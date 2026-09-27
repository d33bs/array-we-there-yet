"""Tests for the NumPy floor measurement and its report section."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from array_we_there_yet.benchmark import (
    BenchmarkConfig,
    measure_numpy_floor,
    run_benchmarks,
)
from array_we_there_yet.data import make_synthetic_dataset
from array_we_there_yet.report import floor_section, floor_table

FLOOR_OPERATIONS = {
    "write",
    "full_read",
    "matrix_materialization",
    "random_rows",
    "feature_projection",
    "vector_norm",
}


def _config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        rows=20,
        dimensions=(6,),
        measured_repetitions=2,
        warmups=0,
        random_row_count=4,
        feature_projection_count=2,
        output_dir=tmp_path / "results",
        artifact_dir=tmp_path / "results" / "artifacts",
    )


def test_numpy_floor_measures_every_operation_and_cleans_up(tmp_path: Path) -> None:
    """The floor times a plain `.npy` file and leaves no file behind."""
    config = _config(tmp_path)
    config.artifact_dir.mkdir(parents=True)
    dataset = make_synthetic_dataset(rows=20, dimensions=6, seed=1)

    records = measure_numpy_floor(
        config=config,
        dataset=dataset,
        selected_rows=np.array([3, 0, 9, 15]),
        selected_features=np.array([4, 1]),
        timestamp="t",
        git_commit="abc",
    )

    frame = pd.DataFrame(records)
    assert set(frame["operation"]) == FLOOR_OPERATIONS
    assert set(frame["backend"]) == {"numpy"}
    assert set(frame["layout"]) == {"npy"}
    assert (frame["calls"] >= 1).all()
    assert len(frame) == len(FLOOR_OPERATIONS) * config.measured_repetitions
    assert list(config.artifact_dir.iterdir()) == []


def test_a_run_saves_the_floor_next_to_the_results(tmp_path: Path) -> None:
    """The floor is written as its own Parquet file, not mixed into the layouts."""
    config = _config(tmp_path)

    raw = run_benchmarks(config)

    floor = pd.read_parquet(config.output_dir / "floor_results.parquet")
    assert set(floor["operation"]) == FLOOR_OPERATIONS
    assert "numpy" not in set(raw["backend"])


def _floor() -> pd.DataFrame:
    rows = [
        {
            "dimensions": 16,
            "operation": "matrix_materialization",
            "median_seconds": 0.01,
        },
        {"dimensions": 16, "operation": "random_rows", "median_seconds": 0.001},
        {"dimensions": 16, "operation": "feature_projection", "median_seconds": 0.002},
    ]
    return pd.DataFrame(rows)


def _summary() -> pd.DataFrame:
    rows = []
    for backend, layout, matrix, random_rows, projection in [
        ("parquet", "fixed_array", 0.05, 0.004, 0.004),
        ("parquet", "wide", 0.1, 0.005, 0.0001),
    ]:
        for operation, seconds in [
            ("matrix_materialization", matrix),
            ("random_rows", random_rows),
            ("feature_projection", projection),
        ]:
            rows.append(
                {
                    "backend": backend,
                    "layout": layout,
                    "profile": "default",
                    "dimensions": 16,
                    "operation": operation,
                    "median_seconds": seconds,
                }
            )
    return pd.DataFrame(rows)


def test_floor_table_divides_each_layout_by_the_numpy_floor() -> None:
    """Each value says how many times slower a layout is than the floor."""
    table = floor_table(_summary(), _floor())

    fixed = table[table["layout"] == "fixed_array"].iloc[0]
    assert fixed["matrix_materialization"] == pytest.approx(5.0)
    assert fixed["random_rows"] == pytest.approx(4.0)
    assert fixed["feature_projection"] == pytest.approx(2.0)


def test_floor_section_explains_the_floor_and_lists_layouts() -> None:
    """The section names the floor time and shows one row for each layout."""
    text = "\n".join(floor_section(_summary(), _floor()))

    assert text.startswith("### Distance from a plain NumPy file")
    assert "A NumPy `.npy` file loads into memory in 10 ms" in text
    assert (
        "For matrix materialization this is a floor, because no format can be "
        "faster than a memory copy."
    ) in text
    assert (
        "For random rows and feature projection it is a baseline, not a floor. A "
        "row-major file is a poor layout for reading a few columns, so a value "
        "below 1x is possible."
    ) in text
    assert "| Parquet `wide` | 10x | 5x | 0.05x |" in text
    assert "| Parquet `fixed_array` | 5x | 4x | 2x |" in text


def test_floor_section_is_absent_without_floor_data() -> None:
    """Older results have no floor and get no section."""
    assert floor_section(_summary(), None) == []
    assert floor_section(_summary(), pd.DataFrame()) == []
