# Zarr chunk reads through RustFS

October 1, 2026. Project commit: `37678fb`. This is a separate experiment. It
does not change the main benchmark tables, figures, or site.

## Question

The main benchmark reads local files. What happens to its Zarr chunk-shape
tradeoff when the same data is read through an S3-compatible API? How many
requests and bytes does each operation actually use?

## Method

RustFS runs in Docker on the same Mac, bound to `127.0.0.1:9000`. We wrote the
same synthetic 2,000-row × 8,192-feature matrix, feature names, and metadata as
the first chunk-shape experiment, using Zarr 3's default `zstd` codec. We tried
two chunk shapes: 1,024×1,024 (the report setting) and 1,024×128 (narrower
feature chunks). Each artifact is about 60.8 MB. No Icechunk or Iceberg is used.

A loopback HTTP proxy counts every GET and HEAD and the response-body bytes.
It also runs a second scenario that waits **20 ms before each HTTP request**.
This is a controlled latency test, not a measurement of AWS S3 latency. Reads
use one Zarr worker and one reusable S3 client, but open the Zarr group on each
call, as the main benchmark does. The OS page cache stays warm. Every result
is compared with the original matrix after timing.

We ran two independent processes, with one warmup and seven measured reads per
operation in each (14 measurements per cell). Each round shuffles the operation
order. All numbers below are pooled medians; q25–q75 values are in
`pooled_summary.csv`.

## Results

Milliseconds (lower is faster). GETs, HEADs, and response bytes are counted
from the actual RustFS request path; they stay the same in both latency
scenarios.

| Read                           | Chunk shape | 0 ms added | 20 ms/request | GETs + HEADs | Response MB |
| ------------------------------ | ----------- | ---------: | ------------: | -----------: | ----------: |
| Eight features across all rows | 1,024×1,024 |        307 |           967 |      13 + 13 |       30.47 |
| Eight features across all rows | 1,024×128   |        126 |         1,128 |      19 + 19 |        6.81 |
| Full matrix                    | 1,024×1,024 |        632 |         1,736 |      21 + 21 |       60.75 |
| Full matrix                    | 1,024×128   |      1,502 |         9,397 |    133 + 133 |       60.76 |
| 128 random rows                | 1,024×1,024 |        566 |         1,748 |      21 + 21 |       60.75 |
| 128 random rows                | 1,024×128   |      1,456 |         9,465 |    133 + 133 |       60.76 |

The eight-column read transfers much less data with narrow feature chunks.
On loopback with no added delay, that cuts the median from 307 to 126 ms.
But it needs 19 GETs and 19 HEADs rather than 13 of each. With 20 ms added
per request, the order reverses: 1,128 ms for narrow chunks against 967 ms
for the report setting. This is specific to this reader and its request
pattern, not a universal crossover point.

Full-matrix and random-row reads transfer nearly the whole matrix with either
shape. Narrow chunks need 128 data-chunk GETs instead of 16, plus five other
GETs (two metadata reads and three listings). They are slower in both latency
scenarios, sharply so when each request is delayed. Storage size barely
changes. No single chunk shape is best for every read.

## What this does and does not tell us

Request counts and response-body bytes are the useful, reproducible part of
this result. They show why feature projection can trade bandwidth for more
requests. In this particular `s3fs` + Zarr path, there is also one HEAD per
GET. A long-lived query engine that caches metadata or bundles requests may
have different counts. The 0-ms timing is noisy; the raw results and q25–q75
intervals are available rather than treating a single median as exact.

RustFS and its proxy both run on the same Mac. There is no WAN, TLS transfer,
AWS S3 service, bandwidth cap, cold cache, or multi-client load. The proxy
buffers each response before forwarding it. Its payload-byte totals exclude
HTTP headers. These times **must not** be read as AWS performance or cost.
This does not compare Icechunk with Iceberg or reproduce the Earthmover
article's numbers. It tests one mechanism from that work—chunk shape against
object-store request and byte counts—using this project's data and operations.

The previous local-disk sweep remains in `spikes/001-zarr-chunk-shapes/`.
`local_disk_context.csv` copies its matching rows for orientation, but those
measurements were made in a separate run, not a paired comparison.

## Reproduce

Start the pinned RustFS image with a loopback-only port. The short local
credentials are only for this isolated test, not a real S3 account:

```bash
docker run -d --name rustfs-s3-bench -p 127.0.0.1:9000:9000 \
  -e RUSTFS_ACCESS_KEY=b -e RUSTFS_SECRET_KEY=b \
  -v rustfs-bench-data:/data \
  rustfs/rustfs@sha256:8cc9801755448b71a786705ce76692c77e14936cccd87cf2fc31842e58f4d1ff \
  --address :9000
```

If the container already exists but is stopped, use
`docker start rustfs-s3-bench`. From the repository root, with `uv sync`:

```bash
uv run --frozen --with s3fs==2026.9.0 python spikes/002-zarr-rustfs/run.py --smoke
uv run --frozen --with s3fs==2026.9.0 python spikes/002-zarr-rustfs/run.py
uv run --frozen --with s3fs==2026.9.0 python spikes/002-zarr-rustfs/run.py --label run2
uv run --frozen python spikes/002-zarr-rustfs/analyze.py
```

The script replaces only its own `bench/002/` objects in RustFS.
`measurements.csv` and `run2_measurements.csv` hold every measured read;
`pooled_summary.csv` holds medians, quartiles, requests, and bytes.
`environment.json` and `run2_environment.json` record versions and the image
ID. The pinned `s3fs` package is installed only for the run; the project
manifest and lockfile are unchanged.
