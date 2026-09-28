"""Benchmarks for wide tables and fixed-size array layouts."""

import pandas as pd

# Pandas 3 infers plain string columns as its own StringDtype by default.
# Every backend's read path still returns object-dtype strings, so keep the
# legacy behavior for consistent comparisons across backends.
pd.set_option("future.infer_string", False)

from .main import run  # noqa: E402
