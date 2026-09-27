"""Tests for the row-sweep section and the measured real-world example."""

import pandas as pd
import pytest

from array_we_there_yet.report import (
    real_world_section,
    real_world_table,
    row_scaling_section,
    row_scaling_table,
)

LAYOUTS = [
    ("csv", "wide", "default", 187_500_000, 1.0),
    ("csv", "wide", "compact", 75_000_000, 1.25),
    ("parquet", "fixed_array", "default", 75_000_000, 0.05),
    ("parquet", "fixed_array", "compact", 60_000_000, 0.07),
]


def _summary() -> pd.DataFrame:
    rows = []
    for backend, layout, profile, size, matrix in LAYOUTS:
        for operation, seconds in [
            ("write", 1.0),
            ("matrix_materialization", matrix),
        ]:
            rows.append(
                {
                    "backend": backend,
                    "layout": layout,
                    "profile": profile,
                    "rows": 2_000,
                    "dimensions": 8_192,
                    "operation": operation,
                    "operation_parameter": "all",
                    "median_seconds": seconds,
                    "q25_seconds": seconds,
                    "q75_seconds": seconds,
                    "artifact_bytes": size,
                    "repetitions": 3,
                }
            )
    return pd.DataFrame(rows)


def _scaling() -> pd.DataFrame:
    """Measured values for two layouts. CSV wide is 1.6 GB and reads in 9 s."""
    return pd.DataFrame(
        [
            {
                "backend": "csv",
                "layout": "wide",
                "profile": "default",
                "rows": 16_000,
                "dimensions": 8_192,
                "size_bytes": 1.6e9,
                "write_seconds": 30.0,
                "matrix_seconds": 9.0,
            },
            {
                "backend": "parquet",
                "layout": "fixed_array",
                "profile": "default",
                "rows": 16_000,
                "dimensions": 8_192,
                "size_bytes": 0.7e9,
                "write_seconds": 2.0,
                "matrix_seconds": 0.5,
            },
        ]
    )


def test_measured_values_replace_scaled_values_for_measured_layouts() -> None:
    """Layouts in the scaling check use measured size and read time."""
    table = real_world_table(_summary(), _scaling())

    csv = table[(table["layout"] == "wide") & (table["profile"] == "default")].iloc[0]
    parquet = table[
        (table["layout"] == "fixed_array") & (table["profile"] == "default")
    ].iloc[0]
    assert csv["size_gb"] == pytest.approx(1.6)
    assert csv["read_seconds"] == pytest.approx(9.0)
    assert csv["download_seconds"] == pytest.approx(16.0)
    assert bool(csv["measured"]) is True
    assert parquet["size_gb"] == pytest.approx(0.7)
    assert parquet["read_seconds"] == pytest.approx(0.5)


def test_layouts_missing_from_the_scaling_check_keep_scaled_values() -> None:
    """A layout that was not measured falls back to the linear scaling."""
    table = real_world_table(_summary(), _scaling())

    compact = table[(table["layout"] == "wide") & (table["profile"] == "compact")].iloc[
        0
    ]

    assert bool(compact["measured"]) is False
    assert compact["size_gb"] == pytest.approx(0.6)


def test_section_compares_predicted_and_measured_values() -> None:
    """A table shows how close the linear prediction was to the real file."""
    text = "\n".join(real_world_section(_summary(), _scaling()))

    assert "### Scaling check" in text
    assert (
        "| Layout | Scaled size | Measured size | Scaled read | Measured read |" in text
    )
    assert "| CSV `wide` | 1.5 GB | 1.6 GB | 8 s | 9 s |" in text
    assert "| Parquet `fixed_array` | 0.6 GB | 0.7 GB | 0.4 s | 0.5 s |" in text
    assert "measured on a real file of about 16,000 rows" in text
    assert "scale linearly" not in text


def test_section_without_a_scaling_check_says_it_scales_linearly() -> None:
    """Without measurements the section keeps its linear-scaling assumption."""
    text = "\n".join(real_world_section(_summary(), None))

    assert "### Scaling check" not in text
    assert "scale linearly" in text


def _sweep() -> pd.DataFrame:
    rows = []
    for count, matrix, random_rows, projection in [
        (2_000, 0.01, 0.005, 0.002),
        (200_000, 1.2, 0.1, 0.3),
    ]:
        for operation, seconds in [
            ("matrix_materialization", matrix),
            ("random_rows", random_rows),
            ("feature_projection", projection),
        ]:
            rows.append(
                {
                    "backend": "parquet",
                    "layout": "fixed_array",
                    "profile": "default",
                    "rows": count,
                    "dimensions": 1_024,
                    "operation": operation,
                    "median_seconds": seconds,
                    "q25_seconds": seconds,
                    "q75_seconds": seconds,
                    "artifact_bytes": count * 4_096,
                }
            )
    return pd.DataFrame(rows)


def test_row_scaling_table_gives_the_growth_from_the_smallest_to_the_largest() -> None:
    """Growth is the time at the most rows divided by the time at the fewest."""
    table = row_scaling_table(_sweep())

    row = table.iloc[0]
    assert row["matrix_materialization"] == pytest.approx(120.0)
    assert row["random_rows"] == pytest.approx(20.0)
    assert row["feature_projection"] == pytest.approx(150.0)
    assert row["row_growth"] == pytest.approx(100.0)


def test_row_scaling_section_explains_linear_growth() -> None:
    """The section names the row growth and shows each layout's time growth."""
    text = "\n".join(row_scaling_section(_sweep()))

    assert text.startswith("### Scaling with row count")
    assert "The row count grows 100x, from 2,000 to 200,000 rows" in text
    assert "1,024 features" in text
    assert "Time that grows 100x is linear in the row count." in text
    assert "| Parquet `fixed_array` | 120x | 20x | 150x |" in text


def test_row_scaling_section_is_absent_without_a_sweep() -> None:
    """Results without a sweep get no section."""
    assert row_scaling_section(None) == []
    assert row_scaling_section(pd.DataFrame()) == []


def _wide_summary() -> pd.DataFrame:
    """Add a Parquet wide layout that is 0.8 GB in the scaled file."""
    extra = _summary()[
        (_summary()["backend"] == "parquet") & (_summary()["profile"] == "default")
    ].assign(layout="wide", artifact_bytes=100_000_000)
    projection = extra[extra["operation"] == "write"].assign(
        operation="feature_projection", median_seconds=0.002
    )
    return pd.concat([_summary(), extra, projection], ignore_index=True)


def _wide_scaling() -> pd.DataFrame:
    """Measured Parquet wide: 0.8 GB, a 4.2 MB footer, and 5 MB for 8 features."""
    row = {
        "backend": "parquet",
        "layout": "wide",
        "profile": "default",
        "rows": 16_000,
        "dimensions": 8_192,
        "size_bytes": 0.8e9,
        "write_seconds": 3.0,
        "matrix_seconds": 1.0,
        "footer_bytes": 4.2e6,
        "bytes_8_features": 5.0e6,
    }
    return pd.DataFrame([row])


def test_streaming_table_uses_the_measured_bytes_when_they_exist() -> None:
    """A measured partial read replaces the estimate and is labeled as measured."""
    text = "\n".join(real_world_section(_wide_summary(), _wide_scaling()))

    assert "| 8 features of 8,192 | 1.5 GB | 5 MB (measured) | 0.6 GB |" in text
    assert "The footer is 4.2 MB (measured)." in text
    assert "(estimate)" in text


def test_streamed_row_uses_the_measured_bytes() -> None:
    """The streamed row in the main table is built from the measured bytes."""
    table = real_world_table(_wide_summary(), _wide_scaling())

    streamed = table[table["variant"] == "streamed"].iloc[0]
    assert streamed["size_gb"] == pytest.approx(0.005)
    assert bool(streamed["measured"]) is True


def test_streamed_row_stays_an_estimate_without_a_measurement() -> None:
    """Without scaling data the streamed row is marked as not measured."""
    table = real_world_table(_wide_summary(), None)

    streamed = table[table["variant"] == "streamed"].iloc[0]
    assert bool(streamed["measured"]) is False


def test_scaling_check_says_where_the_scaled_numbers_were_close_and_where_not() -> None:
    """Sizes within 10% and reads within 1.5x are called close. Misses are named."""
    text = "\n".join(real_world_section(_summary(), _scaling()))

    assert (
        "Measured sizes are within 10% of the scaled sizes, except Parquet "
        "`fixed_array` (1.2x larger). Measured read times are within 1.5x of the "
        "scaled times."
    ) in text


def test_scaling_check_names_a_read_time_miss() -> None:
    """A layout that reads twice as slowly as scaled is named."""
    scaling = _scaling()
    scaling.loc[scaling["layout"] == "fixed_array", "matrix_seconds"] = 2.0

    text = "\n".join(real_world_section(_summary(), scaling))

    assert (
        "Measured read times are within 1.5x of the scaled times, except Parquet "
        "`fixed_array` (5x slower)."
    ) in text


def _two_layout_sweep() -> pd.DataFrame:
    rows = []
    for layout, growths in [
        ("fixed_array", (120.0, 20.0, 150.0)),
        ("wide", (30.0, 2.0, 40.0)),
    ]:
        for count, factor in [(2_000, 0.0), (200_000, 1.0)]:
            for operation, growth in zip(
                ["matrix_materialization", "random_rows", "feature_projection"],
                growths,
                strict=True,
            ):
                rows.append(
                    {
                        "backend": "parquet",
                        "layout": layout,
                        "profile": "default",
                        "rows": count,
                        "dimensions": 1_024,
                        "operation": operation,
                        "median_seconds": 1.0 if factor == 0.0 else growth,
                        "q25_seconds": 1.0,
                        "q75_seconds": 1.0,
                        "artifact_bytes": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_row_scaling_section_names_the_fastest_and_slowest_growth() -> None:
    """The section says which layout grew most and least."""
    text = "\n".join(row_scaling_section(_two_layout_sweep()))

    assert (
        "Matrix materialization grew the most in Parquet `fixed_array` (120x) and "
        "the least in Parquet `wide` (30x). Random rows grew the most in Parquet "
        "`fixed_array` (20x) and the least in Parquet `wide` (2x)."
    ) in text
