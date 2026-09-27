## Terminology

| Term                   | Meaning                                                                                               |
| ---------------------- | ----------------------------------------------------------------------------------------------------- |
| Array-like layout      | A layout that stores all features of a row in one field.                                              |
| Artifact               | A file or directory that a benchmark run writes.                                                      |
| Backend                | A storage format and the library that reads it. DuckDB is a database engine with its own file format. |
| Compact profile        | Write settings that make smaller files and can cost time. See Write settings.                         |
| Egress                 | The fee that a cloud provider charges when data leaves its network, billed per gigabyte.              |
| Feature                | One numeric measurement for a row.                                                                    |
| Feature projection     | Reading only a few chosen features and skipping the rest.                                             |
| Matrix materialization | Reading every value into one in-memory table of numbers (an `N x D` NumPy array).                     |
| Metadata               | Descriptive columns, such as sample ID or plate ID.                                                   |
| Packed array           | An array stored as text inside one CSV field.                                                         |
| Ratio                  | One result divided by a reference result.                                                             |
| Row group              | A block of rows in a Parquet file. A reader can skip whole row groups.                                |
| Streaming              | Reading part of a file with byte-range requests, so that only those bytes are sent.                   |
| Wide layout            | A layout that stores each feature in its own column.                                                  |
