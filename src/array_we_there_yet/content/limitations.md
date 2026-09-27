## Limitations

- **Scale.** The main benchmark has 2,000 rows, about 65 MB at the largest
  feature count. The row sweep goes to 200,000 rows at 1,024 features, about
  820 MB, and the scaling check reads a real 1.5 GB file. All of these fit in
  memory. None of them shows reads that exceed memory. The sweep is one run and
  is not pooled. CSV was measured only up to 20,000 rows in the sweep.
- **Repetitions.** Each run makes 3 repetitions of each measurement. The results
  pool the runs listed in the Environment table. Write timings vary the most,
  especially TileDB writes. They can differ between runs by many times. The
  report marks measurements that vary by more than half of their median.
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
