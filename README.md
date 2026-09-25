# Array We There Yet?

A benchmark of storage formats for high-dimensional profile data.
In this data, each row holds many numeric features.

The benchmark compares seven storage backends. For each backend, it measures a
wide layout against an array-like layout:

- A wide layout has one column for each feature.
- An array-like layout stores all features of one row in a single field.

The results compare each array-like layout with the wide layout in its own
format. They also compare every layout with CSV wide, the most common way to
share this kind of data. CSV is a text format, so ratios against it exaggerate
the gain of any binary format.

<!-- array-we-there-yet-results:start -->

## Summary

> **Main finding.** Array-like layouts read whole feature matrices 4.4x to 1,300x faster than wide layouts in 6 of 7 backends. Reading only 8 features gives mixed results: array-like layouts are faster in Vortex (25x), Lance (12x), and DuckDB (2.1x), about the same in Parquet, and slower in Zarr (16x), CSV (5.8x), and TileDB (1.2x).

- **Whole-matrix reads.** Matrix materialization is faster with the array-like layout in Lance (1,300x), TileDB (240x), Zarr (94x), Vortex (33x), DuckDB (7.2x), and Parquet (4.4x). It is slower in CSV (1.4x).
- **Selecting a few features.** Reading 8 features is faster in Vortex (25x), Lance (12x), and DuckDB (2.1x). It is about the same in Parquet. It is slower in Zarr (16x), CSV (5.8x), and TileDB (1.2x).
- **Text packing.** CSV packed arrays are slower than CSV wide for write, matrix materialization, random rows, feature projection, mixed retrieval, and vector norm, and faster only for full read. They are 11% larger.
- **Encoding settings matter.** Parquet's array layout is 29% smaller than its wide layout by default and 15% smaller with the compact profile. Default dictionary encoding inflates the wide layout.
- **Limits.** These results come from a single machine, synthetic data, warm caches, and 3 repetitions per timing. See Limitations.

## Results

The benchmark used synthetic data with 2,000 rows and 256 to 8,192 features. Each timing is the median of 3 repeated runs.

### Key findings

Each cell compares the array-like layout with the wide layout of the same backend at 8,192 features. Negative percentages and "lower" mean faster or smaller.

| Backend | Layout            | Matrix materialization | Feature projection | Write        | Storage size |
| ------- | ----------------- | ---------------------- | ------------------ | ------------ | ------------ |
| CSV     | `delimited_array` | +34%                   | 5.5x higher        | +40%         | +11%         |
| CSV     | `json_array`      | +54%                   | 6.2x higher        | +42%         | +11%         |
| Parquet | `fixed_array`     | -77%                   | -7%                | -88%         | -29%         |
| DuckDB  | `duckdb_array`    | -86%                   | -53%               | -37%         | -57%         |
| Zarr    | `zarr_matrix`     | 94x lower              | 16x higher         | 190x lower   | -10%         |
| TileDB  | `tiledb_dense`    | 240x lower             | +21%               | 6,000x lower | -11%         |
| Vortex  | `fixed_array`     | 33x lower              | 25x lower          | 28x lower    | -14%         |
| Lance   | `fixed_array`     | 1,300x lower           | 12x lower          | 18x lower    | -8%          |

These layouts used more than one thread in at least one operation: Lance `wide` (median 1.1, up to 3.7), Vortex `fixed_array` (median 3.1, up to 7.8), Vortex `wide` (median 1.2, up to 3.2). Parallelism is CPU time divided by wall time, so 1.0 means one busy thread. The benchmark cannot limit the native thread pools of Lance and Vortex.

### How to read the figures

- Each panel shows one operation or one summary measure.
- Lower is better in every panel.
- Ratio panels divide one result by a reference result. A value of 1.0 means the same as the reference.
- Geometric Mean Time is the geometric mean of the time ratios across all operations. The geometric mean is the standard way to average ratios.
- Error bars show the q25-to-q75 range across repeated runs.
- The y-axis uses a log scale to show small and large changes.
- Dashed lines are wide layouts. Solid lines are array-like layouts.

### Every layout against CSV wide

![Time and storage ratios for every layout against CSV wide](figures/combined_facet_overview.png)

Figure 1. Every layout divided by CSV wide, which is the flat line at 1.0. Parquet wide is the only other wide layout shown here. Figure 2 shows all wide layouts.

CSV wide is the reference because it is the most common way to share this kind of data. It is a text format, so ratios against it exaggerate the gain of any binary format. Figure 3 is the fairer comparison: each array-like layout against a wide layout in the same format.

### Wide layouts

![Time and storage ratios for wide layouts against CSV wide](figures/wide_layouts_facet_overview.png)

Figure 2. Wide layouts only, divided by CSV wide, which is the flat line at 1.0.

### Array-like layouts against their own wide layout

![Time and storage ratios for array-like layouts against wide layouts](figures/backend_wide_facet_overview.png)

Figure 3. Each array-like layout divided by the wide layout of the same backend. Values below 1.0 favor the array-like layout.

### Encoding sensitivity

Encoding and compression choices change file size and speed. This table checks whether the array-versus-wide results hold under both write profiles. Each cell shows the array-like layout relative to the wide layout of the same backend, first with the default profile and then with the compact profile. The last column says whether every measure keeps its direction, better or worse than wide, in both profiles.

| Backend | Layout         | Storage size | Matrix materialization | Feature projection      | Write                       | Conclusion     |
| ------- | -------------- | ------------ | ---------------------- | ----------------------- | --------------------------- | -------------- |
| Parquet | `fixed_array`  | -29% → -15%  | -77% → -75%            | -7% → +35%              | -88% → -76%                 | Reverses       |
| Zarr    | `zarr_matrix`  | -10% → -10%  | 94x lower → 55x lower  | 16x higher → 28x higher | 190x lower → 41x lower      | Same direction |
| TileDB  | `tiledb_dense` | -11% → -15%  | 240x lower → 52x lower | +21% → 4.2x higher      | 6,000x lower → 1,400x lower | Same direction |

### Encodings observed

The table lists the compression and encodings that each format wrote at the largest feature count. Bytes per value is the artifact size divided by the number of stored values. A raw `float32` value takes 4 bytes. Most size differences between the binary formats come from default compression and encoding choices, not from the layout. Random values compress poorly, so the sizes describe this data set and not real profile data.

Parquet turns on dictionary encoding by default. Every value in this data is unique, so the dictionary adds overhead. Parquet wide takes 5.67 bytes per value by default and 3.97 with the compact profile. DuckDB wide takes 8.26 bytes per value, about 2.1x the raw size. The benchmark did not investigate why. It stores 8,192 columns of 2,000 values each.

| Backend | Layout            | Profile | Compression and encodings                            | Bytes per value |
| ------- | ----------------- | ------- | ---------------------------------------------------- | --------------- |
| CSV     | `wide`            | default | uncompressed decimal text                            | 10.92           |
| CSV     | `wide`            | compact | gzip-compressed decimal text                         | 4.68            |
| CSV     | `delimited_array` | default | uncompressed decimal text                            | 12.17           |
| CSV     | `json_array`      | default | uncompressed decimal text                            | 12.17           |
| Parquet | `wide`            | default | snappy; PLAIN, RLE, RLE_DICTIONARY                   | 5.67            |
| Parquet | `wide`            | compact | zstd; BYTE_STREAM_SPLIT, RLE                         | 3.97            |
| Parquet | `fixed_array`     | default | snappy; PLAIN, RLE, RLE_DICTIONARY                   | 4.05            |
| Parquet | `fixed_array`     | compact | zstd; BYTE_STREAM_SPLIT, RLE                         | 3.38            |
| DuckDB  | `wide`            | default | FLOAT; ALP, ALPRD, Constant                          | 8.26            |
| DuckDB  | `duckdb_array`    | default | FLOAT[8192]; ALPRD, Constant                         | 3.55            |
| Zarr    | `wide`            | default | Blosc lz4 level 5, byte shuffle                      | 4.20            |
| Zarr    | `wide`            | compact | Blosc zstd level 5, byte shuffle                     | 3.88            |
| Zarr    | `zarr_matrix`     | default | Blosc lz4 level 5, byte shuffle                      | 3.76            |
| Zarr    | `zarr_matrix`     | compact | Blosc zstd level 5, byte shuffle                     | 3.48            |
| TileDB  | `wide`            | default | attributes: none; dimensions: Zstd                   | 4.59            |
| TileDB  | `wide`            | compact | attributes: ByteShuffle + Zstd; dimensions: Zstd     | 4.23            |
| TileDB  | `tiledb_dense`    | default | attributes: none; dimensions: Zstd                   | 4.10            |
| TileDB  | `tiledb_dense`    | compact | attributes: ByteShuffle + Zstd; dimensions: Zstd     | 3.59            |
| Vortex  | `wide`            | default | encodings chosen by Vortex                           | 4.12            |
| Vortex  | `fixed_array`     | default | encodings chosen by Vortex                           | 3.54            |
| Lance   | `wide`            | default | encodings chosen by Lance (data storage version 2.2) | 4.35            |
| Lance   | `fixed_array`     | default | encodings chosen by Lance (data storage version 2.2) | 4.01            |

### Appendix: storage size against time

This appendix uses synthetic random data. It shows how the compact profile moves the size and the time of each format. It is not a ranking of formats for real data.

![Storage size against time for the default and compact profiles](figures/encoding_profiles.png)

Figure 4. Storage size against time at the largest feature count. Each arrow goes from the default profile (filled) to the compact profile (open). Points without an arrow have no compact profile. Squares are wide layouts and circles are array-like layouts. Lower and further left is better.

Changes from the default profile to the compact profile:

| Layout                | Storage size | Matrix materialization | Write       |
| --------------------- | ------------ | ---------------------- | ----------- |
| CSV `wide`            | -57%         | +25%                   | 4.3x higher |
| Parquet `wide`        | -30%         | +32%                   | -29%        |
| Parquet `fixed_array` | -17%         | +42%                   | +38%        |
| Zarr `wide`           | -8%          | +1%                    | +24%        |
| Zarr `zarr_matrix`    | -7%          | +71%                   | 5.6x higher |
| TileDB `wide`         | -8%          | +1%                    | -34%        |
| TileDB `tiledb_dense` | -12%         | 4.7x higher            | 2.8x higher |

The compact profile changes only the write settings in Write settings. DuckDB, Lance, and Vortex have no compact profile.

<!-- array-we-there-yet-results:end -->

## Layouts

The benchmark writes these layouts:

| Backend | Wide layout               | Array-like layout                |
| ------- | ------------------------- | -------------------------------- |
| CSV     | One column per feature    | Delimited string and JSON string |
| Parquet | One column per feature    | `FixedSizeList<float32>`         |
| DuckDB  | One column per feature    | `FLOAT[N]` fixed-size array      |
| Zarr    | One array per feature     | 2D chunked feature array         |
| TileDB  | One attribute per feature | Dense 2D array                   |
| Vortex  | One column per feature    | `FixedSizeList<float32>`         |
| Lance   | One column per feature    | `FixedSizeList<float32>`         |

For packed and fixed-array layouts, the size of `feature_names.json` is included
in the artifact size. This accounts for the feature names that a wide schema
stores in its columns.

Parquet has no fixed-size list type. PyArrow writes the `FixedSizeList` layout as
an ordinary variable-length list and restores the fixed size from the Arrow
schema stored in the file metadata. Readers that do not use that schema see a
variable-length list.

## Operations

Each run measures these operations:

| Operation              | Description                                                                                                                  |
| ---------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Write                  | Write the full dataset.                                                                                                      |
| Full read              | Read the full table.                                                                                                         |
| Matrix materialization | Read the data into one `N x D` NumPy array.                                                                                  |
| Random rows            | Read 128 random rows, using each format's own row-selection call. CSV has none, so it reads the whole file.                  |
| Feature projection     | Read a fixed set of features.                                                                                                |
| Mixed retrieval        | Read metadata and selected features together. The selected columns are read for all rows, and the rows are picked afterward. |
| Vector norm            | Read the matrix as in matrix materialization, then compute the L2 norm of each row in NumPy.                                 |

Each result is stored as an absolute time. The report also stores ratios to CSV
wide and ratios to the wide layout of the same backend. A ratio below `1.0`
means the layout is faster or smaller than its reference.

## Methodology

- **Data.** Each feature count uses a separate synthetic dataset with a fixed
  seed of 42. It has 2,000 rows, independent standard-normal `float32` features,
  and three string metadata columns: `sample_id`, `plate_id`, and `well_id`.
- **Protocol.** For each backend, layout, and feature count, the benchmark
  writes the dataset several times to new artifacts. It then times each read
  operation on the last artifact. One warmup run comes before the measured runs.
- **Timing.** Wall time comes from `time.perf_counter`. CPU time comes from
  `time.process_time`, which counts every thread in the process. Checking the
  returned values happens after the timer stops and is not included.
- **Cache state.** The benchmark does not drop the operating system page cache.
  All reads are warm-cache reads.
- **Selections.** Random rows and projected features are 128 rows and 8 features
  chosen with fixed seeds. The counts do not change with the feature count.
  Each format selects rows with its own call: `take` in Lance, `dataset.take` in
  Parquet, `rowid` in DuckDB, `scan(indices=...)` in Vortex, `oindex` in Zarr, and
  `multi_index` in TileDB. The dataset has 2,000 rows in one row group or chunk,
  so most formats still decode the whole block. For them random rows costs about
  as much as matrix materialization.
- **Column requests.** Wide layouts are read with a full scan followed by a column
  selection. The benchmark never names thousands of columns in one request.
  Lance takes about 1.35 ms for each column that a request names. Naming all
  8,192 columns took 11 s, against 0.57 s for a full scan that returns the same
  values.
- **Returned objects.** Matrix materialization returns a NumPy array for every
  backend. Full read returns the native object of each library: an Arrow table,
  a pandas DataFrame, or a dictionary of arrays. Full read times therefore
  include different conversion work.
- **Statistics.** Each timing is the median of the repeated runs. Error bars are
  the q25-to-q75 range. Ratios divide medians. Summary panels average ratios
  with the geometric mean.
- **Threads.** The benchmark asks for one thread. It enforces this for Arrow,
  DuckDB, TileDB, and the Blosc compressor that Zarr uses. Lance and Vortex use
  native thread pools that the benchmark cannot limit. The `median_parallelism` column in
  `results/summary.parquet` is the CPU time divided by the wall time. A value near
  1.0 means one thread was busy. A larger value means more than one. For layouts
  that used more than one thread, the results also compare CPU time, which counts
  every thread.
- **Correctness.** Every write and read is checked for shape, `float32` type, and
  values within `1e-6` of the source data.
- **Encodings.** Each format runs with its default settings. Formats with a
  verified compression setting also run a compact profile. Encoding and
  compression choices change file size and speed. The Encoding sensitivity table
  checks whether the array-versus-wide results hold under both profiles. The
  Encodings observed table lists what each format wrote.
- **Artifact size.** The size is the bytes on disk, including `feature_names.json`
  for layouts that store no column names.

<!-- array-we-there-yet-setup:start -->

## Environment

| Item      | Value                                                                                                   |
| --------- | ------------------------------------------------------------------------------------------------------- |
| CPU       | Apple M4 Pro (12 logical cores)                                                                         |
| Memory    | 48 GiB                                                                                                  |
| Platform  | macOS-26.7-arm64-arm-64bit                                                                              |
| Python    | 3.11.13                                                                                                 |
| `duckdb`  | 1.5.5                                                                                                   |
| `lance`   | 12.0.0                                                                                                  |
| `numpy`   | 2.2.6                                                                                                   |
| `pandas`  | 2.2.3                                                                                                   |
| `pyarrow` | 25.0.1                                                                                                  |
| `tiledb`  | 0.36.1                                                                                                  |
| `vortex`  | 0.86.1                                                                                                  |
| `zarr`    | 2.18.7                                                                                                  |
| Threads   | Arrow CPU 1, Arrow I/O 1, DuckDB 1, TileDB 1, Zarr Blosc 1; Lance and Vortex use their library defaults |

## Backends and access paths

| Backend | Package or binding              | Layouts measured                        | Impact on timing                                                              |
| ------- | ------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------- |
| CSV     | `pandas` CSV I/O                | `wide`, `delimited_array`, `json_array` | Text parsing and packed-array decoding can add Python and pandas cost.        |
| Parquet | `pyarrow.parquet`               | `wide`, `fixed_array`                   | Compiled Arrow readers can reduce Python overhead for column and array reads. |
| DuckDB  | `duckdb` Python package         | `wide`, `duckdb_array`                  | DuckDB runs queries in its native engine before results return to Python.     |
| Zarr    | `zarr` Python package           | `wide`, `zarr_matrix`                   | Chunked array reads can favor matrix-shaped access patterns.                  |
| TileDB  | `tiledb` Python package         | `wide`, `tiledb_dense`                  | TileDB native I/O is included through the Python binding.                     |
| Vortex  | `vortex-data` (`vortex` import) | `wide`, `fixed_array`                   | Vortex reads and Arrow conversion both count in these timings.                |
| Lance   | `lance` Python package          | `wide`, `fixed_array`                   | Lance scans and Arrow table conversion both count in these timings.           |

The timings include each package and binding, not only the storage layout. Different language layers can change the result.

<!-- array-we-there-yet-setup:end -->

## Write settings

The benchmark writes every layout with a **default** profile. It uses only the
settings listed here. All other settings are the defaults of the package version
in the Environment table.

Formats with a compression setting that the benchmark verified also run a
**compact** profile. It writes smaller files and can cost time. The figures at
the top of the results use only the default profile. The Encoding sensitivity
table and the appendix compare the two.

| Backend | Default profile                                                                                                                       | Compact profile                                                        |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| CSV     | No compression. Packed arrays use 9 significant digits per value.                                                                     | `gzip` on the wide layout only.                                        |
| Parquet | `snappy` compression through `pyarrow.parquet.write_table`. Dictionary encoding is on.                                                | `zstd`, dictionary encoding off, and byte-stream split on float data.  |
| DuckDB  | One `CREATE TABLE` followed by `CHECKPOINT`. Arrays are `FLOAT[N]`.                                                                   | None. DuckDB chooses its own compression.                              |
| Zarr    | Chunks of 1,024 rows. The matrix layout also uses at most 1,024 features. The default Blosc compressor: `lz4`, level 5, byte shuffle. | Blosc with `zstd`, level 5, byte shuffle.                              |
| TileDB  | Dense array. Tile extents of at most 1,024 rows and 1,024 features. No attribute filters.                                             | Byte shuffle followed by `zstd` at level 5 on the attributes.          |
| Vortex  | None. The benchmark calls `vortex.io.write` with defaults.                                                                            | None. The benchmark found no setting to change.                        |
| Lance   | None. The benchmark calls `lance.write_dataset` with defaults.                                                                        | None. The compression settings that the benchmark tried had no effect. |

## Limitations

- **Scale.** The largest dataset is about 65 MB and fits in memory. It does not
  show row-group, chunk, or fragment behavior at scale, or reads that exceed
  memory.
- **Repetitions.** Each timing uses 3 repetitions, so the q25-to-q75 error bars
  come from only 3 values. A second full run with the same settings agreed with
  the first within 2% for the median measurement, and within 12% for 90% of the
  measurements. About 1 in 20 measurements differed by more than 25%. They were
  mostly write timings, especially TileDB writes, and timings of a few
  milliseconds or less. The error bars understate this run-to-run variation.
- **Cache.** The results do not show cold-storage or object-storage reads.
- **Data.** Independent random values are close to the worst case for compression.
  The best possible lossless size for this data is about 3.3 bytes per value. The
  storage sizes say little about the dictionary, run-length, and delta encodings
  that real profile data would use. The data has no missing values and one
  numeric type.
- **Machine.** All results come from one machine, described in the Environment
  table.
- **Software stack.** Timings include the Python bindings and the conversion to
  the returned object. They do not measure the storage formats alone.
- **Tuning.** The benchmark runs two profiles. Neither profile is tuned for the
  data. Other row-group sizes, chunk sizes, and codecs can change the results.
  Vortex, Lance, and DuckDB have no compact profile.
- **Scope.** The benchmark does not measure filters, updates, concurrent access,
  schema changes, vector search, or tool support.

## Terminology

| Term                   | Meaning                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| Array-like layout      | A layout that stores all features of a row in one field.                                              |
| Artifact               | A file or directory that a benchmark run writes.                                                      |
| Backend                | A storage format and the library that reads it. DuckDB is a database engine with its own file format. |
| Feature                | One numeric measurement for a row.                                                                    |
| Matrix materialization | Reading data into an `N x D` NumPy array.                                                             |
| Metadata               | Descriptive columns, such as sample ID or plate ID.                                                   |
| Packed array           | An array stored as text inside one CSV field.                                                         |
| Ratio                  | One result divided by a reference result.                                                             |
| Wide layout            | A layout that stores each feature in its own column.                                                  |

## Usage

Install the dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Run the full benchmark and update the results section of this README:

```bash
uv run array-we-there-yet run
```

The defaults use 2,000 rows, 256 to 8,192 features, three measured repetitions,
and one warmup run. A full run takes about 4 hours on the machine in the
Environment table.

Run a small smoke test:

```bash
uv run array-we-there-yet run --rows=100 --dimensions=8 --measured_repetitions=1 --warmups=0
```

## Outputs

The benchmark writes these files:

| Path                                        | Contents                                                  |
| ------------------------------------------- | --------------------------------------------------------- |
| `results/raw_results.parquet`               | One row per measurement.                                  |
| `results/summary.parquet`                   | Median and quartile timings and artifact sizes.           |
| `results/csv_wide_ratio_summary.parquet`    | Ratios to CSV wide.                                       |
| `results/wide_layout_ratio_summary.parquet` | Wide layout ratios to CSV wide.                           |
| `results/ratio_summary.parquet`             | Array-like ratios to the wide layout of the same backend. |
| `results/profile_comparison.parquet`        | Compact profile ratios to the default profile.            |
| `results/encodings.parquet`                 | Compression and encodings that each format wrote.         |
| `results/environment.json`                  | Configuration and package versions.                       |
| `figures/*.png`                             | Figures, including one plot for each operation.           |

The benchmark deletes the storage artifacts of each layout after it measures
that layout. A full run needs about 1.2 GB of free disk space. The run saves
`raw_results.parquet` and `encodings.parquet` after each feature count, so a
failed run keeps the feature counts that finished.

All result tables are Parquet files. The benchmark does not write CSV copies.
Read them with `pandas.read_parquet`.

## Development

This repository comes from the
[`CU-DBMI/template-uv-python-research-software`](https://github.com/CU-DBMI/template-uv-python-research-software)
template. It uses `uv`, `pytest`, and pre-commit.

Run the tests:

```bash
uv run pytest
```

Run the Ruff checks:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

Run the full local pipeline:

```bash
uv run poe pipeline
```
