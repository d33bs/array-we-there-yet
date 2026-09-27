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

Storing several values in one cell is not a shortcut around the relational
model. It is a documented alternative within it. Codd's First Normal Form
(1970) required every value to be atomic, one number or string per cell. The
NF² (non-first-normal-form) relations of Jaeschke and Schek (1982) relaxed
that on purpose, allowing an attribute's value to be a list, and SQL
standardized array-typed columns not long after. See References for the
lineage, including the paper behind Parquet's own nested encoding.
