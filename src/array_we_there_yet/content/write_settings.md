## Write settings

The benchmark writes every layout with a **default** profile. It uses only the
settings listed here. All other settings are the defaults of the package version
in the Environment table.

Formats with a compression setting that the benchmark verified also run a
**compact** profile. It writes smaller files and can cost time. The plots use
only the default profile. The Encoding sensitivity table and the appendix
compare the two.

| Backend | Default profile                                                                                                                          | Compact profile                                                        |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| CSV     | No compression. Packed arrays use 9 significant digits per value.                                                                        | `gzip` on the wide layout only.                                        |
| Parquet | `snappy` compression through `pyarrow.parquet.write_table`. Dictionary encoding is on.                                                   | `zstd`, dictionary encoding off, and byte-stream split on float data.  |
| DuckDB  | One `CREATE TABLE` followed by `CHECKPOINT`. Arrays are `FLOAT[N]`.                                                                      | None. DuckDB chooses its own compression.                              |
| Zarr    | Chunks of 1,024 rows. The matrix layout also uses at most 1,024 features. No codec is requested, so Zarr writes its own default: `zstd`. | Blosc with `zstd`, level 5, byte shuffle.                              |
| TileDB  | Dense array. Tile extents of at most 1,024 rows and 1,024 features. No attribute filters.                                                | Byte shuffle followed by `zstd` at level 5 on the attributes.          |
| Vortex  | None. The benchmark calls `vortex.io.write` with defaults.                                                                               | None. The benchmark found no setting to change.                        |
| Lance   | None. The benchmark calls `lance.write_dataset` with defaults.                                                                           | None. The compression settings that the benchmark tried had no effect. |
