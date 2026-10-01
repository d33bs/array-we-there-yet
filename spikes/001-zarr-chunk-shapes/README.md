# Zarr chunk-shape experiment

Date: October 1, 2026. Project commit: `37678fb`. These measurements are
separate from the main benchmark tables and plots.

## Question

Does a different Zarr matrix chunk shape help selected-row or selected-feature
reads without a large cost for full-matrix reads and writes?

## Method

The script uses the project's synthetic data generator and Zarr matrix readers.
It changes only the chunk shape. The format, default Zarr compression (`zstd`),
metadata, selected rows, and selected features stay the same. Every returned
array must match the source values exactly.

- Matrices: 20,000 rows × 1,024 features and 2,000 rows × 8,192 features.
- Chunks: 1,024×1,024 (the report setting), 128×1,024, 1,024×128, and 256×256.
- Reads: full matrix, 128 random rows, and 8 selected features.
- Timings: two separate process runs. Each run has one unmeasured write, three
  measured writes, one read warmup, and seven measured reads per operation.
  Short reads repeat within a sample until the sample lasts at least 50 ms.
  The table shows medians across both runs (six writes or 14 reads per cell).
- Machine: local macOS laptop. The script caps Zarr at one worker. The OS page
  cache stays warm. Temporary artifacts are deleted after each run.

The `selected_chunk_bytes` column sums stored payloads of chunks selected by
an operation. It is **not** measured disk traffic or S3 traffic.

## Results

Times are pooled median milliseconds. Smaller is faster. `full` loads the
entire matrix; `rows` reads 128 rows; `features` reads 8 features.

| Matrix         |  Chunk rows × features | Write |  Full |  Rows | Features |
| -------------- | ---------------------: | ----: | ----: | ----: | -------: |
| 2,000 × 8,192  | 1,024 × 1,024 (report) | 112.9 |  59.8 |  56.8 |     27.3 |
| 2,000 × 8,192  |            128 × 1,024 | 205.9 | 104.4 | 106.9 |     52.5 |
| 2,000 × 8,192  |            1,024 × 128 | 210.8 | 126.5 | 108.2 | **10.8** |
| 2,000 × 8,192  |              256 × 256 | 326.5 | 173.6 | 163.4 |     26.3 |
| 20,000 × 1,024 | 1,024 × 1,024 (report) | 135.2 |  64.3 |  68.2 | **67.5** |
| 20,000 × 1,024 |            128 × 1,024 | 278.3 | 129.2 |  76.1 |    127.9 |
| 20,000 × 1,024 |            1,024 × 128 | 261.8 | 132.9 | 129.2 |     68.2 |
| 20,000 × 1,024 |              256 × 256 | 400.6 | 220.7 | 154.1 |    151.6 |

For the wide matrix, 1,024×128 chunks cut the selected-feature read from
27.3 ms to 10.8 ms. That request selected 6.6 MB of chunk payload rather
than 30.3 MB. However, full reads took about twice as long and writes took
almost twice as long. For the taller matrix, the same feature chunks selected
37.9 MB instead of 75.7 MB, but read time stayed near 68 ms. The extra
chunks offset the smaller payloads on this local machine.

The 128-row chunks reduced the estimated chunk payload for a 128-row read of
the tall matrix from 75.7 MB to 42.2 MB. They increased the chunk count from
20 to 87, and did not improve read time. The four chunk shapes had nearly the
same artifact size for each matrix. The random float data compresses poorly.

## Verdict: PARTIAL

A narrow feature chunk helps feature projection on a very wide matrix, at a
large cost for writes and full or row reads. The current 1,024×1,024 chunk
shape remains the best general-purpose choice among those tested here.
There is no evidence here to change the main report's Zarr setting.

This test does not measure S3 requests, sharding, Icechunk, cold reads, or real
profile data. It cannot predict the cloud result in the Earthmover study. The
follow-up in `spikes/002-zarr-rustfs/` measures S3-compatible request counts and
response bytes with RustFS and controlled latency, not AWS S3.

## Reproduce

From the repository root:

```bash
uv run array-we-there-yet run --rows=100 --dimensions=8 --measured_repetitions=1 --warmups=0
uv run --frozen python spikes/001-zarr-chunk-shapes/run.py
uv run --frozen python spikes/001-zarr-chunk-shapes/run.py --label run2
uv run --frozen python spikes/001-zarr-chunk-shapes/analyze.py
```

The first command is a one-minute smoke test of the main benchmark. The other
three regenerate this experiment's measurements and pooled table.

The two `measurements.csv` files contain every measurement. The two
`summary.csv` files contain per-run medians. `pooled_summary.csv` contains
the medians and chunk counts in the table. Each `environment.json` file
records the versions and thread settings. The experiment did not change
`results/`. The report adds only a short Limitations note; its plots and
benchmark tables still use the original results.
