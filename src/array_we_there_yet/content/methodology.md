## Methodology

- **Data.** Each feature count uses a separate synthetic dataset with a fixed
  seed of 42. It has 2,000 rows, independent standard-normal `float32` features,
  and three string metadata columns: `sample_id`, `plate_id`, and `well_id`.
- **Protocol.** For each backend, layout, and feature count, the benchmark
  writes the dataset several times to new artifacts. It then times each read
  operation on the last artifact. One warmup run comes before the measured runs.
- **Timing.** Wall time comes from `time.perf_counter`. CPU time comes from
  `time.process_time`, which counts every thread in the process. Checking the
  returned values happens after the timer stops and is not included. A read that
  takes less than 50 ms repeats inside one sample until the sample lasts at least
  50 ms. The benchmark divides the sample time by the number of calls and stores
  the call count in the raw results. This removes most of the timer and
  thread-start noise from fast reads.
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
- **Statistics.** Each timing is the median of all repetitions. The results pool
  independent runs of the same code, so the error bars show the variation
  between runs and not only within one run. Error bars are the q25-to-q75 range.
  Ratios divide medians. Summary panels average ratios with the geometric mean.
  A measurement that varies by more than half of its median is marked with `*`.
  Pooling refuses runs from different commits or from uncommitted code.
- **Threads.** The benchmark asks for one thread. It enforces this for Arrow,
  DuckDB, TileDB, and the Blosc compressor that Zarr uses. Lance and Vortex use
  native thread pools that the benchmark cannot limit. The `median_parallelism`
  column in `results/summary.parquet` is the CPU time divided by the wall time. A
  value near 1.0 means one thread was busy. A larger value means more than one.
  For layouts that used more than one thread, the results also compare CPU time,
  which counts every thread.
- **Correctness.** Every write and read is checked for shape, `float32` type, and
  values within `1e-6` of the source data.
- **Access path check.** For each layout, matrix materialization should take
  about as long as a full read, and random rows should not take longer than
  matrix materialization. The report flags a layout that takes more than three
  times as long. This check finds readers that ask a library for far more than an
  operation needs.
- **Plain file.** A plain NumPy `.npy` file is timed for the same operations. For
  matrix materialization it is the speed of a memory copy, so no format can be
  faster. For random rows and feature projection it is a baseline: a row-major
  file is a poor layout for reading a few columns. The report divides each
  layout's time by the time of the plain file.
- **Row sweep.** The benchmark runs again at 2,000, 20,000, and 200,000 rows and
  1,024 features. CSV stops at 20,000 rows, because its files are 2.7 times the
  raw size and it is the slowest format. The report divides the time at the most
  rows by the time at the fewest, so linear growth is 100 times.
- **Scaling check.** The real-world example scales sizes and read times from the
  2,000-row benchmark. The scaling check writes and reads a real file of about
  1.5 GB for each layout of the example, and compares the measured size and read
  time with the scaled ones. It also measures the bytes that a client downloads to
  read 8 columns of a Parquet file, from the footer and the column chunk sizes.
- **Encodings.** Each format runs with its default settings. Formats with a
  verified compression setting also run a compact profile. Encoding and
  compression choices change file size and speed. The Encoding sensitivity table
  checks whether the array-versus-wide results hold under both profiles. The
  Encodings observed table lists what each format wrote.
- **Artifact size.** The size is the bytes on disk, including `feature_names.json`
  for layouts that store no column names.
