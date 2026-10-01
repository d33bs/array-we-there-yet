# Array We There Yet?

Does it matter how you lay out a big table of numbers on disk? This project
tests it. Picture a table with one row for each sample, such as a well of cells
or a document, and hundreds or thousands of numeric features for each row. You
can store it **wide**, with one column for each feature, or **array-like**, with
all the features of a row in one field. The benchmark compares the two in seven
formats: CSV, Parquet, DuckDB, Zarr, TileDB, Vortex, and Lance. It measures speed,
file size, and the cost of moving the data.

**[Read the report](https://d33bs.github.io/array-we-there-yet/).** It has the
results, interactive plots you can filter, a real-world cost example, the
method, and the limits of the benchmark.

## Summary

Array-like layouts load a whole feature matrix much faster than wide layouts in
most backends. Reading only a few features is faster in some backends and
slower in others. The layouts also change file size, and so the time and
egress cost of moving a dataset. The report gives the numbers.

## What the benchmark measures

- **Write** and **full read** of the dataset.
- **Matrix materialization**: read every value into one NumPy array.
- **Random rows**, **feature projection**, and **mixed retrieval**.
- **Vector norm**: matrix materialization followed by an L2 norm of each row.
- **Storage size**, with a default and a compact write profile.
- **Row-count sweep** and a **scaling check** on a real 1.5 GB file.

## Quick start

Install the dependencies with [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
```

Rebuild the report page from the saved results in `results/`:

```bash
uv run array-we-there-yet report
```

This writes `site/index.html`. Open the file in a browser. The plots load
Plotly from a CDN, so the page needs a network connection.

## Run the benchmark

Run the full benchmark and rebuild the report:

```bash
uv run array-we-there-yet run
```

The defaults use 2,000 rows, 256 to 8,192 features, three measured repetitions,
and one warmup run. A full run takes about 4 hours on a laptop and needs about
1.2 GB of free disk space.

Run the same code twice, each time into its own directory, then pool the runs:

```bash
uv run array-we-there-yet run --output_dir=results_a --site_dir=site_a
uv run array-we-there-yet run --output_dir=results_b --site_dir=site_b
uv run array-we-there-yet combine --inputs=results_a,results_b
```

Run the row sweep, and the scaling check that writes and reads a real 1.5 GB
file. Both use the results of the main run:

```bash
uv run array-we-there-yet sweep
uv run array-we-there-yet scaling
```

The sweep takes about 4.5 hours, and the scaling check takes about 15 minutes.

Run a small smoke test:

```bash
uv run array-we-there-yet run --rows=100 --dimensions=8 --measured_repetitions=1 --warmups=0
```

## Outputs

| Path                                        | Contents                                                  |
| ------------------------------------------- | --------------------------------------------------------- |
| `results/raw_results.parquet`               | One row per measurement.                                  |
| `results/summary.parquet`                   | Median and quartile timings and artifact sizes.           |
| `results/csv_wide_ratio_summary.parquet`    | Ratios to CSV wide.                                       |
| `results/wide_layout_ratio_summary.parquet` | Wide layout ratios to CSV wide.                           |
| `results/ratio_summary.parquet`             | Array-like ratios to the wide layout of the same backend. |
| `results/profile_comparison.parquet`        | Compact profile ratios to the default profile.            |
| `results/encodings.parquet`                 | Compression and encodings that each format wrote.         |
| `results/row_sweep_summary.parquet`         | Median timings at 2,000, 20,000, and 200,000 rows.        |
| `results/row_sweep_raw.parquet`             | One row per measurement of the row sweep.                 |
| `results/scaling_check.parquet`             | Size, write time, and read time of the 1.5 GB files.      |
| `results/floor_results.parquet`             | One row per measurement of the plain NumPy file.          |
| `results/floor_summary.parquet`             | Median timings of the plain NumPy file.                   |
| `results/environment.json`                  | Configuration and package versions.                       |
| `site/index.html`                           | The report page. It is built, not committed.              |

All result tables are Parquet files. Read them with `pandas.read_parquet`.

## Development

This repository comes from the
[`CU-DBMI/template-uv-python-research-software`](https://github.com/CU-DBMI/template-uv-python-research-software)
template. It uses `uv`, `pytest`, and pre-commit.

The report page comes from `src/array_we_there_yet/site.py`. Its text is in
`src/array_we_there_yet/content/`, and its template, style, and script are in
`src/array_we_there_yet/site/`. A GitHub Actions workflow builds the page and
publishes it to GitHub Pages when changes reach `main`.

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
