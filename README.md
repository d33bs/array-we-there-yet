# Array We There Yet?

This project compares two ways to store high-dimensional profile data.
High-dimensional data means each row has many numeric values.

The first layout is a wide table, with one column for each feature.
The second layout stores all features for one row in one array-like field.
An array-like field stores an ordered list of values in one table cell.

## At a Glance

The starter benchmark uses synthetic data with 2,000 rows and up to 128
features. Synthetic data is generated data with known values.

Early results show three useful patterns:

- Arrow IPC fixed arrays made matrix materialization much faster than Arrow IPC wide tables.
- Parquet fixed arrays made full reads and matrix materialization faster than Parquet wide tables.
- CSV packed arrays were slower than CSV wide tables for most operations.

The benchmark is not a final ranking. It shows where each layout helps or hurts.

## Terms

| Term                   | Meaning                                                   |
| ---------------------- | --------------------------------------------------------- |
| Array                  | An ordered collection of values.                          |
| Artifact               | A file or directory written by a benchmark run.           |
| Backend                | The storage system that writes and reads the data.        |
| Feature                | One numeric measurement for a profile row.                |
| Feature projection     | Reading only selected feature columns or array positions. |
| Fixed-size array       | An array where every row has the same number of values.   |
| L2 norm                | The usual vector length from Euclidean geometry.          |
| Matrix materialization | Reading data and creating an `N x D` NumPy array.         |
| Metadata               | Descriptive columns, such as sample ID or plate ID.       |
| `N x D`                | A matrix with `N` rows and `D` feature columns.           |
| Packed array           | An array stored as text inside one CSV field.             |
| Profile                | One row of feature values and metadata.                   |
| Ratio                  | The array-like result divided by the wide result.         |
| Schema                 | The named rules for columns and their value types.        |
| Semantics              | The meaning of the data.                                  |
| Syntax                 | The form of the data.                                     |
| Vector                 | A numeric array used for math.                            |
| Wide table             | A table with one column for each feature.                 |

For ratios, a value less than `1.0` favors the array-like layout.
A value more than `1.0` favors the wide layout.

## Concept Model

A table is data in rows and columns.
It is a data idea, not one file format.

One row stores one profile.
Metadata identifies or describes that profile.
Feature data measures that profile.

A wide layout spreads one vector across many feature columns.
An array-like layout stores one vector in one column.

This difference is about form and meaning.
Syntax is the data form, such as one column or many columns.
Semantics are the data meaning, such as one profile vector.

A fixed-size array carries a length rule with the data.
That rule tells other tools how many values belong to each row.

## Layouts

The benchmark currently writes these layouts:

| Backend   | Wide layout            | Array-like layout                |
| --------- | ---------------------- | -------------------------------- |
| CSV       | One column per feature | Delimited string and JSON string |
| Arrow IPC | One column per feature | `FixedSizeList<float32>`         |
| Parquet   | One column per feature | `FixedSizeList<float32>`         |
| DuckDB    | One column per feature | List column                      |

For packed and fixed-array layouts, `feature_names.json` is measured with the
artifact size. This accounts for feature names that the wide schema stores
directly.

## Operations

Each run measures these operations:

- Write the full dataset.
- Read the full table.
- Materialize an `N x D` NumPy matrix.
- Read a fixed set of random profiles.
- Project a fixed set of features.
- Compute one row-wise L2 norm vector.

The report stores absolute timings and array-to-wide ratios. For time and
storage ratios, values less than `1.0` favor the array-like layout.

The side-by-side figures show the absolute values first.
The ratio figures then show the relative change from each wide baseline.

## Usage

Install dependencies with `uv`:

```bash
uv sync
```

Run the starter benchmark and update this README:

```bash
uv run array-we-there-yet run
```

Run a small smoke benchmark:

```bash
uv run array-we-there-yet run --rows=100 --dimensions=8 --measured_repetitions=1 --warmups=0
```

## Outputs

The benchmark writes these files:

- `results/raw_results.parquet`
- `results/raw_results.csv`
- `results/summary.parquet`
- `results/summary.csv`
- `results/environment.json`
- `figures/*.png`

The benchmark also writes storage artifacts under `results/artifacts/`.

## Development

This repository was generated from
[`CU-DBMI/template-uv-python-research-software`](https://github.com/CU-DBMI/template-uv-python-research-software).
It keeps the template structure for `uv`, `pytest`, pre-commit, and local agent-guidance
structure.

Run tests:

```bash
uv run pytest
```

Run Ruff checks:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
```

Run the full local pipeline:

```bash
uv run poe pipeline
```

## References

- [Simple English skill](https://github.com/AminBlg/SimpleEnglish/tree/main/skills/simple-english) for the plain-language rules used in this README.
- [The Legend of Arrow: A Link to the Vector](https://gist.github.com/d33bs/8ae640bf97aacd97a4e3893e41795bcc) for the concept model behind this benchmark.
- [CU-DBMI UV Python research software template](https://github.com/CU-DBMI/template-uv-python-research-software) for the project scaffold.
- [Apache Arrow](https://arrow.apache.org/) for Arrow IPC and fixed-size list arrays.
- [Apache Parquet](https://parquet.apache.org/) for columnar storage.
- [DuckDB](https://duckdb.org/) for the DuckDB backend.
- [NumPy](https://numpy.org/) for matrix materialization.
- [uv](https://docs.astral.sh/uv/) for Python environment management.

<!-- array-we-there-yet-results:start -->

## Current Results

These starter results use synthetic data with 128 features.
A ratio less than 1.0 favors the array-like layout.
The primary figures show absolute values with wide and array-like layouts side by side.

Use the table below to find the largest changes.
Then open the side-by-side figures to see the absolute size of each difference.

Read this first:

- The largest time gain is arrow_ipc fixed_array for random rows at 0.122x wide.
- The largest time loss is csv json_array for feature projection at 8.9x wide.
- The smallest array-like artifact is arrow_ipc fixed_array at 0.956x wide.
- The largest array-like artifact is csv json_array at 1.79x wide.

| backend   | layout          | operation              | operation_parameter | time_ratio | artifact_size_ratio |
| :-------- | :-------------- | :--------------------- | :------------------ | ---------: | ------------------: |
| arrow_ipc | fixed_array     | feature_projection     | 8                   |      0.337 |               0.956 |
| arrow_ipc | fixed_array     | full_read              | all                 |      0.514 |               0.956 |
| arrow_ipc | fixed_array     | matrix_materialization | all                 |      0.134 |               0.956 |
| arrow_ipc | fixed_array     | random_rows            | 128                 |      0.122 |               0.956 |
| arrow_ipc | fixed_array     | vector_norm            | l2                  |      0.179 |               0.956 |
| arrow_ipc | fixed_array     | write                  | all                 |      0.552 |               0.956 |
| csv       | delimited_array | feature_projection     | 8                   |      4.245 |               1.113 |
| csv       | delimited_array | full_read              | all                 |      1.033 |               1.113 |
| csv       | delimited_array | matrix_materialization | all                 |      1.938 |               1.113 |
| csv       | delimited_array | random_rows            | 128                 |      1.952 |               1.113 |
| csv       | delimited_array | vector_norm            | l2                  |      2.079 |               1.113 |
| csv       | delimited_array | write                  | all                 |       1.62 |               1.113 |
| csv       | json_array      | feature_projection     | 8                   |      8.903 |               1.786 |
| csv       | json_array      | full_read              | all                 |      1.686 |               1.786 |
| csv       | json_array      | matrix_materialization | all                 |      4.217 |               1.786 |
| csv       | json_array      | random_rows            | 128                 |      4.025 |               1.786 |
| csv       | json_array      | vector_norm            | l2                  |      4.283 |               1.786 |
| csv       | json_array      | write                  | all                 |      1.253 |               1.786 |
| duckdb    | duckdb_array    | feature_projection     | 8                   |       0.93 |               1.001 |
| duckdb    | duckdb_array    | full_read              | all                 |      1.036 |               1.001 |
| duckdb    | duckdb_array    | matrix_materialization | all                 |      1.011 |               1.001 |
| duckdb    | duckdb_array    | random_rows            | 128                 |      1.079 |               1.001 |
| duckdb    | duckdb_array    | vector_norm            | l2                  |      0.995 |               1.001 |
| duckdb    | duckdb_array    | write                  | all                 |      1.061 |               1.001 |
| parquet   | fixed_array     | feature_projection     | 8                   |      1.084 |               1.103 |
| parquet   | fixed_array     | full_read              | all                 |      0.483 |               1.103 |
| parquet   | fixed_array     | matrix_materialization | all                 |      0.438 |               1.103 |
| parquet   | fixed_array     | random_rows            | 128                 |      0.425 |               1.103 |
| parquet   | fixed_array     | vector_norm            | l2                  |      0.413 |               1.103 |
| parquet   | fixed_array     | write                  | all                 |       1.03 |               1.103 |

Raw results are in `results/raw_results.parquet` and `results/raw_results.csv`.
Summary results are in `results/summary.parquet` and `results/summary.csv`.

Primary side-by-side figures:

- `figures/write_absolute_comparison.png`
- `figures/full_read_absolute_comparison.png`
- `figures/matrix_materialization_absolute_comparison.png`
- `figures/random_rows_absolute_comparison.png`
- `figures/feature_projection_absolute_comparison.png`
- `figures/vector_norm_absolute_comparison.png`
- `figures/storage_size_absolute_comparison.png`

Secondary ratio figures:

- `figures/write_time_ratio.png`
- `figures/full_read_time_ratio.png`
- `figures/matrix_materialization_time_ratio.png`
- `figures/random_rows_time_ratio.png`
- `figures/feature_projection_time_ratio.png`
- `figures/vector_norm_time_ratio.png`
- `figures/storage_size_ratio.png`

<!-- array-we-there-yet-results:end -->
