"""Describe the compression and encodings that each format wrote."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import duckdb
import lance
import pyarrow.parquet as pq
import tiledb

FEATURE_COLUMN = "feature_0000"
BLOSC_SHUFFLE_NAMES = {0: "no shuffle", 1: "byte shuffle", 2: "bit shuffle"}


def describe_encoding(backend: str, layout: str, path: Path) -> str:
    """Return a plain-text description of how an artifact stores its features."""
    describers = {
        "csv": _describe_csv,
        "parquet": _describe_parquet,
        "duckdb": _describe_duckdb,
        "zarr": _describe_zarr,
        "tiledb": _describe_tiledb,
        "lance": _describe_lance,
        "vortex": _describe_vortex,
    }
    describer = describers.get(backend)
    if describer is None:
        return "not inspected"
    return describer(layout, path)


def _describe_csv(_: str, path: Path) -> str:
    if (path / "data.csv.gz").exists():
        return "gzip-compressed decimal text"
    return "uncompressed decimal text"


def _describe_parquet(layout: str, path: Path) -> str:
    metadata = pq.ParquetFile(path).metadata
    wanted = FEATURE_COLUMN if layout == "wide" else "features.list.element"
    row_group = metadata.row_group(0)
    for index in range(metadata.num_columns):
        column = row_group.column(index)
        if column.path_in_schema == wanted:
            encodings = ", ".join(sorted(column.encodings))
            return f"{column.compression.lower()}; {encodings}"
    return "not inspected"


def _describe_duckdb(layout: str, path: Path) -> str:
    table = "wide" if layout == "wide" else "array_profiles"
    column = FEATURE_COLUMN if layout == "wide" else "features"
    with duckdb.connect(str(path), read_only=True) as connection:
        data_type = connection.execute(
            "SELECT data_type FROM duckdb_columns() "
            "WHERE table_name = ? AND column_name = ?",
            [table, column],
        ).fetchone()
        segments = connection.execute(f"PRAGMA storage_info('{table}')").fetchall()
    methods = Counter(
        row[8]
        for row in segments
        if row[8] and (row[1] == column or row[1].startswith("feature_"))
    )
    names = ", ".join(sorted(methods))
    return f"{data_type[0] if data_type else 'unknown'}; {names}"


def _describe_zarr(layout: str, path: Path) -> str:
    array_path = "features" if layout == "zarr_matrix" else f"features/{FEATURE_COLUMN}"
    metadata = json.loads((path / array_path / ".zarray").read_text(encoding="utf-8"))
    compressor = metadata.get("compressor")
    if not compressor:
        return "uncompressed"
    shuffle = BLOSC_SHUFFLE_NAMES.get(compressor.get("shuffle"), "unknown shuffle")
    return (
        f"Blosc {compressor['cname']} level {compressor['clevel']}, {shuffle}"
        if compressor.get("id") == "blosc"
        else str(compressor.get("id"))
    )


def _describe_tiledb(_: str, path: Path) -> str:
    with tiledb.open(str(path)) as array:
        schema = array.schema
        attribute = _filters_text(schema.attr(0).filters)
        dimensions = _filters_text(schema.domain.dim(0).filters)
    return f"attributes: {attribute}; dimensions: {dimensions}"


def _filters_text(filters: tiledb.FilterList) -> str:
    names = [type(item).__name__.removesuffix("Filter") for item in filters]
    return " + ".join(names) if names else "none"


def _describe_lance(_: str, path: Path) -> str:
    version = lance.dataset(str(path)).data_storage_version
    return f"encodings chosen by Lance (data storage version {version})"


def _describe_vortex(_: str, __: Path) -> str:
    return "encodings chosen by Vortex"
