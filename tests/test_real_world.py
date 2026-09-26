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
    assert (
        "| Layout | Time spent | Egress spent | Time saved | Egress cost saved |"
        in text
    )
    assert "| CSV `wide` | 6.4 h | $135 | baseline | baseline |" in text
    assert "| CSV `wide` (compact) | 4.4 h | $54 | 1.9 h | $81 |" in text
    assert "| Parquet `fixed_array` | 1.8 h | $54 | 4.6 h | $81 |" in text
    assert "| Parquet `fixed_array` (compact) | 1.5 h | $43 | 4.9 h | $92 |" in text


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

    assert "**What egress is.**" in text
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


def test_real_world_section_defines_egress_before_the_example() -> None:
    """Egress is defined first, in plain words, with a price unit."""
    text = "\n".join(real_world_section(_summary()))

    definition = text.index("**What egress is.**")
    scenario = text.index("This example asks")
    assert definition < scenario
    assert (
        "Cloud providers charge for storing a file and, separately, for data that "
        "leaves their network. The second charge is called egress."
    ) in text
    assert "billed per gigabyte" in text


def test_real_world_section_explains_why_egress_matters_to_hosts_and_users() -> None:
    """One paragraph is for whoever hosts a dataset and one for whoever uses it."""
    text = "\n".join(real_world_section(_summary()))

    assert "**Why it matters for hosting a dataset.**" in text
    assert "**Why it matters for using a dataset.**" in text
    assert "grows with the number of users and with the file size" in text
    assert "requester-pays" in text
    assert "File size sets the download time. The format sets the read time." in text
    hosting = text.index("**Why it matters for hosting a dataset.**")
    using = text.index("**Why it matters for using a dataset.**")
    assert (
        text.index("**What egress is.**") < hosting < using < text.index("### One use")
    )


def _summary_with_wide() -> pd.DataFrame:
    """Add a Parquet wide layout of 0.8 GB, which is 100 MB in the 2,000-row data."""
    extra = _summary()[
        (_summary()["backend"] == "parquet") & (_summary()["profile"] == "default")
    ].assign(layout="wide")
    extra.loc[extra["operation"] == "write", "artifact_bytes"] = 100_000_000
    extra.loc[extra["operation"] == "matrix_materialization", "median_seconds"] = 0.1
    extra["artifact_bytes"] = 100_000_000
    projection = extra[extra["operation"] == "write"].assign(
        operation="feature_projection", median_seconds=0.002
    )
    csv_projection = _summary()[
        (_summary()["backend"] == "csv")
        & (_summary()["profile"] == "default")
        & (_summary()["operation"] == "write")
    ].assign(operation="feature_projection", median_seconds=0.5)
    return pd.concat([_summary(), extra, projection, csv_projection], ignore_index=True)


def test_section_states_the_parquet_size_and_how_much_smaller_it_is() -> None:
    """The size of the Parquet file is stated in gigabytes and as a percentage."""
    text = "\n".join(real_world_section(_summary()))

    assert (
        "**Size as Parquet.** The same data takes 0.6 GB as a Parquet file with the "
        "array layout, 60% smaller than the 1.5 GB CSV wide file. With compact "
        "settings it takes 0.48 GB, 68% smaller."
    ) in text
    assert text.index("**Size as Parquet.**") < text.index("### One use")


def test_size_sentence_names_the_wide_layout_when_it_was_measured() -> None:
    """A Parquet wide file is named too when the results include it."""
    text = "\n".join(real_world_section(_summary_with_wide()))

    assert "As a Parquet wide file it takes 0.8 GB, 47% smaller." in text


def test_streaming_section_explains_partial_reads_in_plain_words() -> None:
    """The section says how a client reads part of a Parquet file from the cloud."""
    text = "\n".join(real_world_section(_summary_with_wide()))

    assert "### Streaming a Parquet file" in text
    assert (
        "Parquet stores a footer that lists where every column and row group is."
        in text
    )
    assert "request only the byte ranges it needs" in text
    assert "Egress is billed for the bytes that are sent" in text
    assert "A CSV file cannot be read in part by column" in text
    assert "The array layout stores all features of a row in one column" in text
    assert (
        "Row selection saves egress only when the file has several row groups" in text
    )


def test_streaming_table_shows_bytes_for_a_partial_read() -> None:
    """Reading 8 of 8,192 features or 1,000 rows sends a fraction of the file."""
    text = "\n".join(real_world_section(_summary_with_wide()))

    assert "| What you read | CSV wide | Parquet wide | Parquet `fixed_array` |" in text
    assert "| Everything | 1.5 GB | 0.8 GB | 0.6 GB |" in text
    assert "| 8 features of 8,192 | 1.5 GB | 4.9 MB (estimate) | 0.6 GB |" in text
    assert "| 1,000 rows | 1.5 GB | 50 MB (estimate) | 38 MB (estimate) |" in text
    assert "Reading 8 features from Parquet wide sends 4.9 MB" in text
    assert "$0.0004" in text


def test_streaming_table_is_absent_without_a_parquet_layout() -> None:
    """Without Parquet results there is no streaming comparison."""
    summary = _summary()
    summary = summary[summary["backend"] == "csv"]

    text = "\n".join(real_world_section(summary))

    assert "### Streaming a Parquet file" not in text
    assert "**Size as Parquet.**" not in text


def test_dollars_use_four_decimals_below_one_cent() -> None:
    """Tiny egress costs do not round to zero."""
    assert _dollars(0.000439) == "$0.0004"
    assert _dollars(0.0099) == "$0.0099"
    assert _dollars(0.01) == "$0.01"


def test_streamed_row_is_added_to_the_one_use_table() -> None:
    """A user who needs 8 features downloads only those columns and the footer."""
    text = "\n".join(real_world_section(_summary_with_wide()))

    assert (
        "| Parquet `wide` (8 features streamed) | 4.9 MB | 0.049 s | 0.016 s "
        "| 0.065 s | $0.0004 |"
    ) in text


def test_streamed_row_is_added_to_the_savings_table() -> None:
    """Streaming saves nearly the whole download and read for that user."""
    text = "\n".join(real_world_section(_summary_with_wide()))

    assert (
        "| Parquet `wide` (8 features streamed) | 1.1 min | $0.44 | 5.3 h | $135 |"
    ) in text


def test_streamed_row_is_explained_under_the_table() -> None:
    """The reader is told that the last row answers a narrower question."""
    text = "\n".join(real_world_section(_summary_with_wide()))

    assert (
        "The streamed row is for a user who needs only 8 features. The other rows "
        "download and read the whole file."
    ) in text


def test_streamed_row_needs_a_feature_projection_time() -> None:
    """Without a measured projection time there is no streamed row."""
    summary = _summary_with_wide()
    summary = summary[summary["operation"] != "feature_projection"]

    text = "\n".join(real_world_section(summary))

    assert "8 features streamed" not in text
    assert "The streamed row is for" not in text


def test_streamed_saving_compares_with_csv_reading_only_8_features() -> None:
    """CSV downloads the whole file but reads only the 8 features that are needed."""
    text = "\n".join(real_world_section(_summary_with_wide()))

    assert (
        "The streamed row compares with CSV wide, which downloads the whole file "
        "and reads only 8 features."
    ) in text


def test_streamed_saving_falls_back_to_the_whole_read_without_csv_projection() -> None:
    """Without a CSV projection time the saving uses the whole-matrix read."""
    summary = _summary_with_wide()
    summary = summary[
        ~(
            (summary["backend"] == "csv")
            & (summary["operation"] == "feature_projection")
        )
    ]

    text = "\n".join(real_world_section(summary))

    assert (
        "| Parquet `wide` (8 features streamed) | 1.1 min | $0.44 | 6.4 h | $135 |"
    ) in text


def test_section_says_the_egress_cost_is_an_estimate_for_an_example() -> None:
    """The cost figures are labeled as an example, not a quote."""
    text = "\n".join(real_world_section(_summary()))

    assert (
        "**The egress costs here are an estimate.** They use one list price to show "
        "the size of the effect on a single example. Prices differ by provider, "
        "region, storage service, and volume, and they change over time."
    ) in text
    assert text.index("**The egress costs here are an estimate.**") < text.index(
        "### One use"
    )


def test_section_links_to_the_pricing_pages_of_several_providers() -> None:
    """A reader can check the price with each provider."""
    text = "\n".join(real_world_section(_summary()))

    assert "Egress pricing pages:" in text
    assert (
        "- [AWS S3](https://aws.amazon.com/s3/pricing/): the source of the "
        "$0.09 per GB used here."
    ) in text
    assert (
        "- [Google Cloud](https://cloud.google.com/vpc/network-pricing): egress "
        "is billed per GiB and depends on the source region."
    ) in text
    assert (
        "- [Azure](https://azure.microsoft.com/en-us/pricing/details/bandwidth/): "
        "the first 100 GB each month is free, and the rate depends on the region."
    ) in text
    assert (
        "- [Cloudflare R2](https://developers.cloudflare.com/r2/pricing/): no "
        "egress charges. The host pays for storage and operations instead."
    ) in text
    assert "Check the current page before you plan a budget." in text


def test_savings_table_explains_spent_and_saved() -> None:
    """The text says which columns are totals and which are differences."""
    text = "\n".join(real_world_section(_summary()))

    assert (
        "Time spent and egress spent are the totals over 1,000 uses. Time saved and "
        "egress cost saved compare a layout with CSV wide. Time is the download plus "
        "the read."
    ) in text
    assert "Egress saved" not in text
