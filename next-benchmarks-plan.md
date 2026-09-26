# Next benchmarks plan

This plan covers the next benchmark runs and the open issues from the review.
It starts from commit `cfc1ce5`.

## Where we are

The code in commit `334cc1d` fixes three problems that changed the headline
numbers:

- Lance wide reads named all 8,192 columns in one request. That took 11 s. A full
  scan and a selection took 0.6 s.
- Random rows read the whole matrix for Parquet, DuckDB, Vortex, and Lance.
- Vortex and Lance use more than one thread, and the results did not show it.

The results and figures in the repository still come from the code before these
fixes. The Lance wide numbers are about 19 times too slow. The README summary
still ranks Lance first for whole-matrix reads because of this.

Two earlier full runs of the old code agreed closely for reads. The median
measurement changed by 2%. About 1 in 20 measurements changed by more than 25%.
They were mostly write timings and timings of a few milliseconds.

## Status

Updated after the runs at commit `837ce54`.

| Phase                           | Status      | What is left                                                                                     |
| ------------------------------- | ----------- | ------------------------------------------------------------------------------------------------ |
| 0: run the fixed code           | Done        | None. Two runs of commit `80a73b6`. The Lance wide numbers are now correct.                      |
| 1: statistics                   | Mostly done | The runs are pooled to 6 repetitions. Reads repeat inside each sample. TileDB writes stay noisy. |
| 2: access path checks           | Done        | An automatic check passes for every layout. Mixed retrieval uses native row selection.           |
| 3: scale and cache              | Partly done | The row sweep and a real 1.5 GB scaling check are done. Cold-cache runs are not.                 |
| 4: real data                    | Not started | Needs a dataset choice and a license check.                                                      |
| 5: baselines and outside review | Partly done | The plain NumPy file is done. More encoding profiles and maintainer reviews are not.             |
| 6: README and reporting         | Done        | The README has a summary, a real-world example, and the review fixes.                            |

**What the runs found**

- The two pooled runs agreed within 2% for the median measurement. The Lance wide
  matrix time differed between runs by 2.6 times. The report flags such
  measurements with an asterisk.
- The scaled sizes of the real-world example were close to a real 1.5 GB file. The
  exception is DuckDB, whose array file was 1.8 times larger than scaled.
- With native row selection, Lance random rows stay near 1.5 ms from 2,000 to
  200,000 rows. Parquet with one row group does not benefit.

**Still blocked, and why**

- **Cold cache.** This needs a Linux machine or `sudo purge`.
- **Real data.** This needs a dataset choice from the project owner.
- **Outside reviews.** Opening issues on other projects speaks for the owner.
  Draft the text, then let the owner post it.
- **Row sweep runs.** The sweep is one run. Pool a second run before quoting it.

## Limits that shape this plan

| Limit                                        | Effect on the plan                                           |
| -------------------------------------------- | ------------------------------------------------------------ |
| A full run takes about 4 hours.              | Runs must be planned. Do not rerun for small report changes. |
| The disk has about 19 GB free.               | Large row counts need care. CSV is 2.7 times the raw size.   |
| The machine is a Mac.                        | The page cache cannot be dropped without `sudo purge`.       |
| Lance and Vortex use their own thread pools. | Compare CPU time as well as wall time.                       |
| The results come from one machine.           | State this in every result.                                  |

## Phase 0: Run the fixed code

**Goal.** Replace the stale results with results from the fixed code.

**Tasks**

1. Start from a clean working tree at a commit. The run records the commit hash.
   A `-dirty` suffix means the code changed after the commit.
1. Run `uv run array-we-there-yet run` with the defaults. It takes about 4 hours.
1. Compare the new summary with the last old run, measurement by measurement.
1. Check that only the expected measurements changed:
   - Lance wide: matrix materialization, random rows, and vector norm.
   - Random rows for Parquet, DuckDB, Vortex, and Lance.
1. Read the regenerated README and check the summary sentences.

**Done when**

- Lance wide matrix materialization is close to its full read time. The ratio of
  the two is below 3.
- The Environment table shows a commit hash without `-dirty`.
- No measurement outside the expected list changed by more than 25%.
- The summary says what the data says. Do not keep an old ranking.

**Cost.** About 4 hours of run time and 1 hour of checking.

## Phase 1: Make the statistics trustworthy

**Why.** Each timing uses 3 repetitions. The q25 to q75 error bars come from 3
values. A second run showed the bars are too narrow. Only 20% of the second run's
medians fell inside the first run's bars.

**Tasks**

1. Decide how to get more repetitions. Options:
   - Run the same commit twice and pool the raw results. Add a `run` column.
     Between-run variation shows in the error bars. This costs 4 more hours.
   - Run once with 6 repetitions. This costs about the same, but hides drift
     between runs.
   - Recommendation: run twice and pool.
1. Add a `combine` step to the CLI. It reads several raw result files, keeps the
   run number, and writes one summary.
1. Repeat fast operations inside each sample. Measurements of a few milliseconds
   vary by up to 4 times. Repeat the call until one sample takes at least 50 ms.
   Then divide. Keep the number of calls in the raw results.
1. Investigate TileDB write timings. They varied by up to 16 times between runs.
   Check for disk flushes and file system effects. If they stay noisy, stop
   quoting them and say why in the README.
1. Flag noisy measurements in the summary table. A measurement is noisy when its
   q25 to q75 range is more than 50% of its median.

**Tests**

- `combine` keeps every raw row and adds the run number.
- The pooled summary has 6 repetitions for each measurement.
- Inner repetition returns a per-call time and records the call count.

**Done when**

- Every measurement has at least 6 samples.
- Fewer than 1 in 50 measurements is flagged as noisy, or the README explains
  each group that is.

## Phase 2: Check that every access path is fair

**Why.** The Lance problem was found late. It came from how columns were
requested, not from the format. Other problems of this kind can still exist.

**Tasks**

1. Add an automatic sanity table to the report. For each layout, show matrix
   materialization divided by full read, and random rows divided by matrix
   materialization. Flag values outside 0.5 to 3. The Lance bug gave 19.5.
1. Make mixed retrieval use native row selection where the format has it. Today
   it reads the selected columns for all rows and then picks rows in Python.
1. Decide what to do with vector norm. Today it is matrix materialization plus a
   NumPy call. Choose one:
   - Remove it. It adds no information.
   - Compute it natively where the format can, for example a DuckDB expression.
   - Keep it and label it as "matrix plus NumPy".
1. Read every reader function once more. Look for these patterns:
   - A request that names thousands of columns.
   - A read of all data followed by a slice, where the format has a native call.
   - A Python loop over thousands of arrays or attributes. Zarr wide and TileDB
     wide do this. It is part of the layout. State it in the README.
1. Ask a maintainer of each format to review its reader and write settings. See
   Phase 5.

**Tests**

- The sanity table flags a reader that reads too much. Use a fake reader.
- Mixed retrieval does not call the matrix reader.

**Done when**

- No layout is flagged in the sanity table, or the README explains each flag.
- The README says which operations use native calls and which do not.

## Phase 3: Scale and cache state

**Why.** The largest dataset is 65 MB. It fits in memory, sits in one row group
or chunk, and is always read warm. A database expert will say this hides the
behavior of these formats at scale.

**Tasks**

1. Add a row sweep. Fix the feature count at 1,024. Use 2,000, 20,000, and
   200,000 rows. Raw data is 8 MB, 82 MB, and 820 MB.
   - Write CSV only up to 20,000 rows. Its files reach 2.2 GB at 200,000 rows and
     each layout writes several copies.
   - Check the disk before each size. The run already stops early when space is
     short.
1. Record the number of row groups, chunks, or fragments for each artifact.
   Random rows should get faster than matrix materialization once there is more
   than one.
1. Add cold-cache runs. The Mac cannot drop the page cache without `sudo purge`.
   Options:
   - Run on a Linux machine where the cache can be dropped before each read.
   - Use a machine with less memory than the data.
   - Recommendation: a Linux machine. Keep the Mac results as warm-cache results.
1. Add a plot of time against row count, one line for each layout.

**Tests**

- The sweep writes results for every row count.
- The disk check refuses a run that needs more space than is free.

**Done when**

- The README shows how each operation scales with rows.
- The README states which results are warm-cache and which are cold-cache.

**Cost.** One to two days of run time. This depends on the machine.

## Phase 4: Use real data

**Why.** Independent random values are close to the worst case for compression.
The best possible lossless size is about 3.3 bytes per value. Real profile data
has structure, missing values, and repeated values. The encoding results may
change.

**Tasks**

1. Choose a public profile dataset. Check its license first.
1. Add a loader. Keep the metadata columns and the feature columns. Record the
   number of missing values.
1. Run the full benchmark on it. Report it next to the synthetic results.
1. Repeat the encoding sensitivity table on the real data.
1. Add the real data size to the disk check.

**Decision needed.** Which dataset to use, and whether the data may be stored in
the repository or must be downloaded.

**Done when**

- The README reports results on real data.
- Encoding claims come from real data. The synthetic data is used only to test
  the code.

## Phase 5: Encodings and outside review

**Tasks**

1. Add a speed-of-light baseline. Read a `.npy` file into memory and with
   `mmap`. It shows how far each format is from a plain memory copy.
1. Add more encoding profiles where a setting is verified:
   - Parquet: zstd levels and row group sizes.
   - Zarr and TileDB: compression levels.
   - Lance and Vortex: check each release for a compression setting. Add a
     profile only when a test shows the size changes.
1. Open a short issue or discussion for each project. Include:
   - The layout and the reader code.
   - The write settings.
   - One question: is this the recommended way to read and write this data?
1. Change the code where a maintainer says the benchmark is unfair.

**Done when**

- Each format has had one outside review, or the README says it has not.

## Phase 6: README and reporting

**Tasks**

1. Remove or mark as hypotheses the "Impact on timing" column in the Backends
   table. We did not measure those claims.
1. Add one sentence to Key findings: feature projection is the wide layout's
   best case. It reads 8 columns out of thousands.
1. Define "ratio" once. It is now defined three times.
1. Define "profile data" in the first paragraph. Add one sentence on why it
   matters.
1. Change the "Same direction" label in the sensitivity table. It hides large
   changes in size. Show the size of the change as well.
1. Put the per-operation figures on a three by three grid.
1. Add a short reproducibility note: commit hash, run count, and machine.
1. Keep the real-world example current. The README section models a 1.5 GB CSV
   wide file. It scales the benchmark sizes and read times, and adds two
   assumptions: a 100 MB/s download and a $0.09 per GB egress price.
   - After Phase 0, read the section again. The times and dollars come from the
     new results.
   - Check the egress price against the provider's page before each release.
   - The example assumes that read time grows in a straight line with size. The
     benchmark only measures up to 65 MB. In Phase 3, write and read one real
     1.5 GB CSV wide file and each other layout. Compare the measured times with
     the scaled times. Say in the README how close they are.
   - Add a second price, for example a cloud provider with free egress, so the
     reader can see how much of the saving is time and how much is money.

**Done when**

- A reader who knows databases can state the main finding and its limits after
  reading the first page.
- The real-world example uses measured times, not only scaled ones.

## Order of work

| Step | Phase                         | Blocks                        | Cost               |
| ---- | ----------------------------- | ----------------------------- | ------------------ |
| 1    | Phase 0: run the fixed code   | Everything else               | 5 hours            |
| 2    | Phase 1: statistics           | Trust in every number         | 1 day              |
| 3    | Phase 2: access path checks   | Fair comparison               | 1 day              |
| 4    | Phase 6: README fixes         | Clarity                       | half a day         |
| 5    | Phase 3: scale and cache      | Reviewer concerns about scale | 1 to 2 days        |
| 6    | Phase 4: real data            | Encoding claims               | 1 to 2 days        |
| 7    | Phase 5: baselines and review | Credibility                   | 1 day plus waiting |

Phases 1 and 2 change the benchmark code. Do both before the next long run, so
the run uses the final code. Phase 0 is a check on the current code. It is worth
doing first, but its results will be replaced.

## Decisions needed

1. **Repetitions.** Two runs pooled, or one run with more repetitions?
1. **Cold cache.** Is a Linux machine available?
1. **Real data.** Which dataset, and where does it live?
1. **Row sweep size.** Is 200,000 rows enough, or is 2 million needed? Two million
   rows at 1,024 features is 8 GB raw. It does not fit next to CSV on this disk.
1. **Vector norm.** Remove, make native, or keep with a label?

## Risks

- A full run can fail after hours. The run now saves results after each feature
  count and stops early when the disk is short. Keep both.
- The Mac may run other work during a run. Do not run tests during a benchmark.
- The libraries change fast. Vortex is before version 1.0. Record versions in
  every result.
