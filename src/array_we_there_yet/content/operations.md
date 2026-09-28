## Operations

Each run measures these operations:

| Operation              | Description                                                                                                     |
| ---------------------- | --------------------------------------------------------------------------------------------------------------- |
| Write                  | Write the full dataset.                                                                                         |
| Full read              | Read the full table.                                                                                            |
| Matrix materialization | Read every value into one `N x D` NumPy array, a single in-memory table of numbers.                             |
| Random rows            | Read 128 random rows, using each format's own row-selection call. CSV has none, so it reads the whole file.     |
| Feature projection     | Read only a few chosen features and skip the rest, when the format allows it. Here it is 8 features.            |
| Mixed retrieval        | Read metadata and selected features together, using each format's row-selection call. CSV reads the whole file. |
| Vector norm            | Read the matrix as in matrix materialization, then compute the L2 norm of each row in NumPy.                    |

Each result is stored as an absolute time. The report also stores ratios to CSV
wide and ratios to the wide layout of the same backend.
