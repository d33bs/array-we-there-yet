"""Tests for encoding inspection."""

from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from array_we_there_yet.benchmark import LayoutRunner, layout_runners
from array_we_there_yet.data import make_synthetic_dataset
from array_we_there_yet.encodings import describe_encoding

EXPECTED_TEXT = {
    ("csv", "wide"): "uncompressed decimal text",
    ("csv", "delimited_array"): "uncompressed decimal text",
    ("csv", "json_array"): "uncompressed decimal text",
    ("parquet", "wide"): "snappy; ",
    ("parquet", "fixed_array"): "snappy; ",
    ("duckdb", "wide"): "FLOAT; ",
    ("duckdb", "duckdb_array"): "FLOAT[6]; ",
    ("zarr", "wide"): "Blosc lz4 level 5, byte shuffle",
    ("zarr", "zarr_matrix"): "Blosc lz4 level 5, byte shuffle",
    ("tiledb", "wide"): "attributes: none; dimensions: Zstd",
    ("tiledb", "tiledb_dense"): "attributes: none; dimensions: Zstd",
    ("vortex", "wide"): "encodings chosen by Vortex",
    ("vortex", "fixed_array"): "encodings chosen by Vortex",
    ("lance", "wide"): "encodings chosen by Lance",
    ("lance", "fixed_array"): "encodings chosen by Lance",
}


@pytest.mark.parametrize(
    "runner",
    [runner for runner in layout_runners() if runner.profile == "default"],
    ids=lambda runner: f"{runner.backend}-{runner.layout}",
)
def test_describe_encoding_reports_what_each_format_wrote(
    runner: LayoutRunner, tmp_path: Path
) -> None:
    """Each artifact reports its compression and encodings in plain text."""
    dataset = make_synthetic_dataset(rows=8, dimensions=6, seed=1)
    artifact = runner.write(dataset, tmp_path / "artifact")

    text = describe_encoding(runner.backend, runner.layout, artifact.path)

    assert EXPECTED_TEXT[(runner.backend, runner.layout)] in text


def test_describe_encoding_names_unknown_backends() -> None:
    """Unknown backends get a clear message instead of an error."""
    assert describe_encoding("mystery", "wide", Path()) == "not inspected"


def test_describe_encoding_sees_gzip_csv(tmp_path: Path) -> None:
    """A gzip-compressed CSV artifact is reported as compressed."""
    directory = tmp_path / "artifact"
    directory.mkdir()
    pd.DataFrame({"a": [1]}).to_csv(directory / "data.csv.gz", compression="gzip")

    assert "gzip" in describe_encoding("csv", "wide", directory)


def test_parquet_encoding_reports_the_number_of_row_groups(tmp_path: Path) -> None:
    """Row groups decide whether row selection can skip any data."""
    table = pa.table({"feature_0000": [float(i) for i in range(300)]})
    one = tmp_path / "one.parquet"
    several = tmp_path / "several.parquet"
    pq.write_table(table, one)
    pq.write_table(table, several, row_group_size=100)

    assert "1 row group" in describe_encoding("parquet", "wide", one)
    assert "3 row groups" in describe_encoding("parquet", "wide", several)
