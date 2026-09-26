"""Tests for the fixes from reading the page as a first-time reader."""

from pathlib import Path

import pandas as pd

from array_we_there_yet.report import (
    _dollars,
    _noise_note,
    _run_description,
    profile_change_table,
    real_world_bullet,
    render_results_section,
    sensitivity_table_markdown,
    summary_section,
    write_figures,
)
from tests.test_real_world import _summary as _real_world_summary
from tests.test_report import _figure_summary
from tests.test_summary import _noisy_story, _story_summary


def test_quote_gives_the_range_and_leaves_the_detail_to_the_bullets() -> None:
    """The quote no longer repeats the per-backend feature-selection sentence."""
    lines = summary_section(_story_summary())

    quote = next(line for line in lines if line.startswith("> "))
    assert quote == (
        "> **Main finding.** Array-like layouts read whole feature matrices 4x to "
        "1,000x faster than wide layouts in 2 of 3 backends. Reading only 8 "
        "features gives mixed results."
    )
    text = "\n".join(lines)
    assert "faster in Lance (10x)" in text
    assert text.count("faster in Lance (10x)") == 1


def test_real_world_bullet_states_time_and_egress_per_use() -> None:
    """The summary carries the real-world result in one sentence."""
    bullet = real_world_bullet(_real_world_summary())

    assert bullet == (
        "- **Real-world impact.** For a 1.5 GB CSV wide file, Parquet "
        "`fixed_array` takes 6.4 s per use instead of 23 s and costs $0.054 "
        "instead of $0.135 in egress. See Real-world example."
    )


def test_real_world_bullet_is_absent_without_the_example_data() -> None:
    """Without CSV wide and Parquet results there is no bullet."""
    summary = _real_world_summary()
    summary = summary[summary["backend"] == "csv"]

    assert real_world_bullet(summary) is None


def test_the_summary_lists_the_real_world_bullet_before_the_limits() -> None:
    """The bullet sits with the other findings and the limits stay last."""
    lines = summary_section(_story_summary(), extra_bullets=["- **Extra.** Text."])

    assert lines[-1].startswith("- **Limits.**")
    assert lines[-2] == "- **Extra.** Text."


def test_dollars_keep_enough_digits_that_per_use_and_total_agree() -> None:
    """$0.135 a use times 1,000 uses is $135, not $130."""
    assert _dollars(0.135) == "$0.135"
    assert _dollars(0.054) == "$0.054"
    assert _dollars(0.0432) == "$0.043"
    assert _dollars(0.000439) == "$0.0004"
    assert _dollars(1.5) == "$1.50"
    assert _dollars(135.0) == "$135"


def test_the_noise_key_does_not_start_with_a_list_marker() -> None:
    """A line that starts with an asterisk renders as a bullet and hides the key."""
    note = _noise_note(_noisy_story())

    assert note is not None
    assert note.startswith("An asterisk (`*`) marks a cell")
    assert not note.startswith("*")


def test_run_description_counts_repetitions_not_runs() -> None:
    """The summary says repetitions, because the runs are pooled."""
    summary = pd.DataFrame({"rows": [2000], "dimensions": [256], "repetitions": [6]})

    text = _run_description(summary)

    assert "median of 6 repetitions" in text
    assert "repeated runs" not in text


def test_sensitivity_table_uses_signed_percentages_only() -> None:
    """A 99x reduction reads as a percentage, like the rest of the tables."""
    table = pd.DataFrame(
        [
            {
                "backend": "zarr",
                "layout": "zarr_matrix",
                "storage_size_default": 0.9,
                "storage_size_compact": 0.9,
                "matrix_materialization_default": 0.0101,
                "matrix_materialization_compact": 0.0172,
                "feature_projection_default": 16.6,
                "feature_projection_compact": 28.0,
                "write_default": 0.0059,
                "write_compact": 0.0167,
                "holds": True,
            }
        ]
    )

    markdown = sensitivity_table_markdown(table)

    assert (
        "| Zarr | `zarr_matrix` | -10% → -10% | -99% → -98% | +1,600% → +2,700% "
        "| -99% → -98% | Same side of wide |"
    ) in markdown
    assert "x lower" not in markdown
    assert "x higher" not in markdown


def test_profile_change_table_uses_signed_percentages_only() -> None:
    """The appendix table has no multipliers."""
    comparison = pd.DataFrame(
        [
            {
                "backend": "csv",
                "layout": "wide",
                "size_ratio": 0.43,
                "matrix_materialization_ratio": 1.25,
                "write_ratio": 4.3,
            }
        ]
    )

    markdown = profile_change_table(comparison)

    assert "| CSV `wide` | -57% | +25% | +330% |" in markdown


def test_page_order_puts_plots_before_the_real_world_example(tmp_path) -> None:  # noqa: ANN001
    """The order is summary, plots, real-world example, then detailed results."""
    summary = _real_world_summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    order = [
        section.index("## Summary"),
        section.index("## Plots"),
        section.index("### How to read the figures"),
        section.index("## Real-world example"),
        section.index("## Detailed results"),
        section.index("### Key findings"),
    ]
    assert order == sorted(order)
    assert "## Results" not in section


def test_compact_rows_are_explained_where_they_first_appear() -> None:
    """A reader meets `compact` in the tables, so the note sits under them."""
    section = render_results_section(summary=_real_world_summary(), figure_paths=[])

    note = (
        "Compact rows use the compact write profile, which writes smaller files "
        "and can cost time. See Write settings."
    )
    assert note in section
    assert section.index(note) < section.index("### Savings over 1,000 uses")


def test_encoding_table_intro_says_the_sizes_are_for_a_given_row_count() -> None:
    """Sizes can change with more rows, so the row count is stated."""
    summary = _real_world_summary()
    encodings = pd.DataFrame(
        [
            {
                "backend": "csv",
                "layout": "wide",
                "profile": "default",
                "dimensions": 8192,
                "encoding": "uncompressed decimal text",
            }
        ]
    )

    section = render_results_section(
        summary=summary, figure_paths=[], encodings=encodings
    )

    assert "Sizes are for 2,000 rows and can change with more rows." in section


def test_every_layout_against_csv_wide_is_the_first_figure(tmp_path) -> None:  # noqa: ANN001
    """The reference comparison comes first, and its note points to the fairer one."""
    summary = _figure_summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    every = section.index("### Every layout against CSV wide")
    own = section.index("### Array-like layouts against their own wide layout")
    wide = section.index("### Wide layouts")
    assert every < own < wide
    assert "Figure 1. Every layout divided by CSV wide" in section
    assert "The next figure is the fairer comparison" in section
    assert "The figure before this one" not in section


def test_figure_notes_refer_to_sections_by_name_not_by_number() -> None:
    """Renumbering a figure must not leave a wrong reference behind."""
    section = render_results_section(summary=_real_world_summary(), figure_paths=[])

    assert "Figure 2 shows" not in section
    assert "Figure 3 is" not in section


def test_summary_defines_the_two_operations_in_plain_words() -> None:
    """A reader meets both terms in the bullets, so the summary explains them first."""
    lines = summary_section(_story_summary())

    gloss = next(line for line in lines if line.startswith("**Matrix"))
    assert gloss == (
        "**Matrix materialization** loads every value into one in-memory table of "
        "numbers, ready for analysis or model training. **Feature projection** "
        "loads only a few chosen features, and skips the rest."
    )
    assert lines.index(gloss) < next(
        index for index, line in enumerate(lines) if line.startswith("- **")
    )


def test_readme_defines_both_operations_in_plain_words() -> None:
    """The static tables use the same plain wording as the summary."""
    readme = Path("README.md").read_text()

    assert "Read every value into one `N x D` NumPy array" in readme
    assert "Read only a few chosen features and skip the rest." in readme
    terminology = readme[readme.index("## Terminology") :]
    assert "| Feature projection" in terminology
    assert "| Matrix materialization" in terminology
