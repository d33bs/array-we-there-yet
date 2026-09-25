"""Tests for the real-world example section."""

import pandas as pd
import pytest

from array_we_there_yet.report import (
    _dollars,
    _duration,
    real_world_section,
    real_world_table,
    render_results_section,
)

# CSV wide takes 1 s to read and 187.5 MB, so a 1.5 GB file is 8 times larger.
LAYOUTS = [
    ("csv", "wide", "default", 187_500_000, 1.0),
    ("csv", "wide", "compact", 75_000_000, 1.25),
    ("parquet", "fixed_array", "default", 75_000_000, 0.05),
    ("parquet", "fixed_array", "compact", 60_000_000, 0.07),
]


def _summary() -> pd.DataFrame:
    """Return a summary with four layouts at 8,192 features."""
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


def test_duration_uses_two_significant_digits_and_the_right_unit() -> None:
    """Short times read in seconds, longer ones in minutes or hours."""
    assert _duration(0.4) == "0.4 s"
    assert _duration(23.0) == "23 s"
    assert _duration(6.4) == "6.4 s"
    assert _duration(150.0) == "2.5 min"
    assert _duration(16_600.0) == "4.6 h"


def test_dollars_show_cents_for_small_amounts_and_whole_dollars_for_large() -> None:
    """Amounts under ten dollars keep cents."""
    assert _dollars(0.135) == "$0.14"
    assert _dollars(0.054) == "$0.05"
    assert _dollars(81.0) == "$81"
    assert _dollars(1_234.0) == "$1,234"


def test_real_world_table_scales_the_benchmark_to_a_1_5_gb_csv() -> None:
    """Size scales by the ratio to CSV wide and read time scales with the size."""
    table = real_world_table(_summary())

    csv = table[(table["layout"] == "wide") & (table["profile"] == "default")].iloc[0]
    parquet = table[
        (table["layout"] == "fixed_array") & (table["profile"] == "default")
    ].iloc[0]
    assert csv["size_gb"] == pytest.approx(1.5)
    assert csv["download_seconds"] == pytest.approx(15.0)
    assert csv["read_seconds"] == pytest.approx(8.0)
    assert csv["total_seconds"] == pytest.approx(23.0)
    assert csv["egress_dollars"] == pytest.approx(0.135)
    assert parquet["size_gb"] == pytest.approx(0.6)
    assert parquet["read_seconds"] == pytest.approx(0.4)
    assert parquet["time_saved_seconds"] == pytest.approx(16_600.0)
    assert parquet["egress_saved_dollars"] == pytest.approx(81.0)


def test_real_world_table_is_empty_without_a_csv_wide_baseline() -> None:
    """Without CSV wide there is nothing to compare against."""
    summary = _summary()
    summary = summary[
        ~((summary["backend"] == "csv") & (summary["profile"] == "default"))
    ]

    assert real_world_table(summary).empty
    assert real_world_section(summary) == []


def test_real_world_section_shows_time_and_egress_for_each_layout() -> None:
    """The section has one table for one use and one for many uses."""
    text = "\n".join(real_world_section(_summary()))

    assert text.startswith("## Real-world example")
    assert "| CSV `wide` | 1.5 GB | 15 s | 8 s | 23 s | $0.14 |" in text
    assert "| CSV `wide` (compact) | 0.6 GB | 6 s | 10 s | 16 s | $0.05 |" in text
    assert "| Parquet `fixed_array` | 0.6 GB | 6 s | 0.4 s | 6.4 s | $0.05 |" in text
    assert (
        "| Parquet `fixed_array` (compact) | 0.48 GB | 4.8 s | 0.56 s | 5.4 s | $0.04 |"
    ) in text
    assert "| CSV `wide` (compact) | 1.9 h | $81 |" in text
    assert "| Parquet `fixed_array` | 4.6 h | $81 |" in text
    assert "| Parquet `fixed_array` (compact) | 4.9 h | $92 |" in text


def test_real_world_section_opens_with_a_plain_takeaway() -> None:
    """One sentence says what a common binary format saves per use and over many."""
    text = "\n".join(real_world_section(_summary()))

    assert (
        "**Takeaway.** With Parquet `fixed_array`, one use takes 6.4 s instead of "
        "23 s and costs $0.05 instead of $0.14 in egress. Over 1,000 uses that "
        "saves 4.6 h and $81."
    ) in text
    assert text.index("**Takeaway.**") < text.index("### One use")


def test_real_world_section_has_no_takeaway_without_parquet() -> None:
    """Without the Parquet array layout there is no named comparison."""
    summary = _summary()
    summary = summary[summary["backend"] == "csv"]

    text = "\n".join(real_world_section(summary))

    assert "**Takeaway.**" not in text
    assert "### One use" in text


def test_real_world_section_states_its_assumptions_and_sources() -> None:
    """A reader can see every assumption and where the price comes from."""
    text = "\n".join(real_world_section(_summary()))

    assert "Egress is the fee that a cloud provider charges" in text
    assert "about 16,000 rows of 8,192 features" in text
    assert "100 MB/s" in text
    assert "$0.09 per GB" in text
    assert "1,000 uses" in text
    assert "scale linearly" in text
    assert "first 100 GB each month is free" in text
    assert "https://aws.amazon.com/s3/pricing/" in text


def test_results_section_puts_the_example_after_the_summary() -> None:
    """The example comes after the summary and before the results."""
    section = render_results_section(summary=_summary(), figure_paths=[])

    assert (
        section.index("## Summary")
        < section.index("## Real-world example")
        < section.index("## Results")
    )
