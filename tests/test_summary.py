"""Tests for the summary, key findings, and setup sections of the README."""

from pathlib import Path

import pandas as pd
import pytest

from array_we_there_yet.report import (
    RESULTS_END,
    RESULTS_START,
    SETUP_END,
    SETUP_START,
    _access_path_note,
    _cpu_time_note,
    _noise_note,
    _percent_change,
    array_vs_wide_markdown,
    array_vs_wide_table,
    encoding_findings,
    environment_table,
    render_results_section,
    render_setup_section,
    summary_section,
    update_readme,
)

OPERATIONS = [
    "write",
    "full_read",
    "matrix_materialization",
    "random_rows",
    "feature_projection",
    "mixed_retrieval",
    "vector_norm",
]
# Array-like time divided by wide time, and array-like size divided by wide size.
STORY = {
    ("parquet", "fixed_array"): (
        {
            "write": 0.5,
            "full_read": 0.3,
            "matrix_materialization": 0.25,
            "random_rows": 0.25,
            "feature_projection": 0.95,
            "mixed_retrieval": 0.9,
            "vector_norm": 0.25,
        },
        0.7,
    ),
    ("lance", "fixed_array"): (
        {
            "write": 1.0,
            "full_read": 0.01,
            "matrix_materialization": 0.001,
            "random_rows": 0.001,
            "feature_projection": 0.1,
            "mixed_retrieval": 0.1,
            "vector_norm": 0.001,
        },
        0.9,
    ),
    ("csv", "delimited_array"): (
        {
            "write": 1.4,
            "full_read": 0.8,
            "matrix_materialization": 1.5,
            "random_rows": 1.4,
            "feature_projection": 5.0,
            "mixed_retrieval": 3.0,
            "vector_norm": 1.3,
        },
        1.1,
    ),
}


def _story_summary() -> pd.DataFrame:
    """Return a summary where every wide layout takes 1.0 seconds and 1,000 bytes."""
    rows = []
    for dimensions in [8, 16]:
        for backend, layout, ratios, size in [
            *(
                (backend, layout, values[0], values[1])
                for (backend, layout), values in STORY.items()
            ),
            *(
                (backend, "wide", dict.fromkeys(OPERATIONS, 1.0), 1.0)
                for backend, _ in STORY
            ),
        ]:
            for operation in OPERATIONS:
                seconds = ratios[operation]
                rows.append(
                    {
                        "backend": backend,
                        "layout": layout,
                        "profile": "default",
                        "rows": 10,
                        "dimensions": dimensions,
                        "operation": operation,
                        "operation_parameter": (
                            "8" if operation == "feature_projection" else "all"
                        ),
                        "median_seconds": seconds,
                        "q25_seconds": seconds,
                        "q75_seconds": seconds,
                        "artifact_bytes": 1_000 * size,
                        "repetitions": 3,
                    }
                )
    return pd.DataFrame(rows)


def test_array_vs_wide_table_divides_each_array_layout_by_its_own_wide_layout() -> None:
    """Each row compares one array-like layout with the wide layout of its backend."""
    table = array_vs_wide_table(_story_summary())

    assert list(zip(table["backend"], table["layout"])) == [
        ("csv", "delimited_array"),
        ("parquet", "fixed_array"),
        ("lance", "fixed_array"),
    ]
    largest_dimensions = 16
    lance = table[table["backend"] == "lance"].iloc[0]
    assert lance["dimensions"] == largest_dimensions
    assert lance["matrix_materialization"] == pytest.approx(0.001)
    assert lance["feature_projection"] == pytest.approx(0.1)
    assert lance["storage_size"] == pytest.approx(0.9)


def test_array_vs_wide_markdown_reads_in_percentages_and_multipliers() -> None:
    """The table says how much faster or smaller, not raw ratios."""
    markdown = array_vs_wide_markdown(array_vs_wide_table(_story_summary()))

    assert (
        "| Backend | Layout | Matrix materialization | Feature projection "
        "| Write | Storage size |"
    ) in markdown
    assert "| Lance | `fixed_array` | -99.9% | -90% | 0% | -10% |" in markdown
    assert "| Parquet | `fixed_array` | -75% | -5% | -50% | -30% |" in markdown
    assert "| CSV | `delimited_array` | +50% | +400% | +40% | +10% |" in markdown


def test_summary_section_leads_with_the_main_finding_as_a_quote() -> None:
    """The first content of the summary is one quoted sentence with the finding."""
    lines = summary_section(_story_summary())

    assert lines[0] == "## Summary"
    quote = next(line for line in lines if line.startswith("> "))
    assert quote == (
        "> **Main finding.** Array-like layouts read whole feature matrices 4x to "
        "1,000x faster than wide layouts in 2 of 3 backends. Reading only 8 "
        "features gives mixed results."
    )
    quote_position = 2
    assert lines.index(quote) == quote_position


def test_summary_section_backs_the_finding_with_concrete_bullets() -> None:
    """Each bullet names backends and numbers from the data."""
    text = "\n".join(summary_section(_story_summary()))

    assert (
        "- **Whole-matrix reads.** Matrix materialization is faster with the "
        "array-like layout in Lance (1,000x) and Parquet (4x). It is slower in "
        "CSV (1.5x)."
    ) in text
    assert (
        "- **Selecting a few features.** Reading 8 features is faster in Lance "
        "(10x). It is about the same in Parquet. It is slower in CSV "
        "(5x)."
    ) in text
    assert (
        "- **Text packing.** CSV packed arrays are slower than CSV wide for write, "
        "matrix materialization, random rows, feature projection, mixed retrieval, "
        "and vector norm, and faster only for full read. They are 10% larger."
    ) in text
    assert "single machine" in text
    assert "See Limitations." in text


def test_summary_section_skips_bullets_it_has_no_data_for() -> None:
    """A summary without CSV rows has no text-packing bullet."""
    summary = _story_summary()
    summary = summary[summary["backend"] != "csv"]

    text = "\n".join(summary_section(summary))

    assert "Text packing" not in text
    assert "> **Main finding.**" in text


def _compact_summary() -> pd.DataFrame:
    """Return Parquet results where the compact profile narrows the storage gap."""
    rows = []
    for profile, wide_bytes, array_bytes in [
        ("default", 1_000, 700),
        ("compact", 800, 680),
    ]:
        for layout, size in [("wide", wide_bytes), ("fixed_array", array_bytes)]:
            for operation in ["write", "matrix_materialization"]:
                rows.append(
                    {
                        "backend": "parquet",
                        "layout": layout,
                        "profile": profile,
                        "rows": 10,
                        "dimensions": 16,
                        "operation": operation,
                        "operation_parameter": "all",
                        "median_seconds": 1.0,
                        "q25_seconds": 1.0,
                        "q75_seconds": 1.0,
                        "artifact_bytes": size,
                        "repetitions": 3,
                    }
                )
    return pd.DataFrame(rows)


def test_summary_section_reports_the_encoding_effect_on_storage() -> None:
    """The encoding bullet shows the storage gap under both profiles."""
    text = "\n".join(summary_section(_compact_summary()))

    assert (
        "- **Encoding settings matter.** Parquet's array layout is 30% smaller "
        "than its wide layout by default and 15% smaller with the compact "
        "profile. Default dictionary encoding inflates the wide layout."
    ) in text


def test_encoding_findings_states_the_dictionary_and_duckdb_sizes() -> None:
    """The encodings text quotes bytes per value for Parquet wide and DuckDB wide."""
    summary = _compact_summary()
    duckdb = summary[
        (summary["layout"] == "wide") & (summary["profile"] == "default")
    ].assign(backend="duckdb", artifact_bytes=1_650)
    summary = pd.concat([summary, duckdb], ignore_index=True)

    text = encoding_findings(summary)

    assert text is not None
    assert "Parquet turns on dictionary encoding by default." in text
    assert (
        "Parquet wide takes 6.25 bytes per value by default and 5.00 with the" in text
    )
    assert "DuckDB wide takes 10.31 bytes per value, about 2.6x the raw size" in text
    assert "did not investigate" in text


def test_encoding_findings_is_absent_without_parquet_compact_rows() -> None:
    """Without a Parquet compact profile there is nothing to quote."""
    assert encoding_findings(_story_summary()) is None


def _environment() -> dict:
    """Return environment metadata shaped like `environment.json`."""
    return {
        "python": "3.11.13",
        "platform": "macOS-26.7-arm64-arm-64bit",
        "git_commit": "abc1234-dirty",
        "hardware": {
            "cpu_model": "Test CPU",
            "logical_cores": 12,
            "memory_bytes": 34_359_738_368,
        },
        "packages": {"pyarrow": "25.0.1", "tiledb": [0, 36, 1]},
        "thread_limits": {
            "arrow_cpu_threads": 1,
            "arrow_io_threads": 1,
            "duckdb_threads": 1,
            "lance": "library default",
            "vortex": "library default",
        },
    }


def test_environment_table_puts_the_thread_limits_in_one_row() -> None:
    """The table has one thread-limit row instead of one row per library."""
    table = environment_table(_environment())

    assert "| CPU | Test CPU (12 logical cores) |" in table
    assert "| Memory | 32 GiB |" in table
    assert "| Code version | `abc1234-dirty` |" in table
    assert "| `tiledb` | 0.36.1 |" in table
    assert table.count("Thread limit") == 0
    assert (
        "| Threads | Arrow CPU 1, Arrow I/O 1, DuckDB 1; "
        "Lance and Vortex use their library defaults |"
    ) in table


def test_setup_section_holds_the_environment_and_the_backends() -> None:
    """Setup facts live in their own section, apart from the results."""
    lines = render_setup_section(_story_summary(), _environment())

    headings = [line for line in lines if line.startswith("#")]
    assert headings == ["## Environment", "## Backends and access paths"]
    text = "\n".join(lines)
    assert "| Backend | Package or binding | Layouts measured |" in text
    assert (
        "| CSV | [`pandas` CSV I/O]"
        "(https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html) "
        "| `wide`, `delimited_array` |"
    ) in text
    assert "Impact on timing" not in text
    assert "not only the storage layout" in text


def test_setup_section_omits_the_environment_when_unknown() -> None:
    """Without metadata, only the backends table remains."""
    lines = render_setup_section(_story_summary(), None)

    headings = [line for line in lines if line.startswith("#")]
    assert headings == ["## Backends and access paths"]


def test_update_readme_replaces_the_results_and_setup_blocks(tmp_path: Path) -> None:
    """Both generated blocks are replaced and the text around them is kept."""
    readme = tmp_path / "README.md"
    readme.write_text(
        "# Title\n\nIntro.\n\n"
        f"{RESULTS_START}\n\nold results\n\n{RESULTS_END}\n\n"
        "## Methodology\n\nStatic text.\n\n"
        f"{SETUP_START}\n\nold setup\n\n{SETUP_END}\n\n"
        "## Limitations\n",
        encoding="utf-8",
    )

    update_readme(
        readme_path=readme,
        summary=_story_summary(),
        figure_paths=[],
        environment=_environment(),
    )

    text = readme.read_text(encoding="utf-8")
    assert "old results" not in text
    assert "old setup" not in text
    assert text.index("## Summary") < text.index("## Methodology")
    assert text.index("## Methodology") < text.index("## Environment")
    assert text.index("## Environment") < text.index("## Limitations")
    assert "Static text." in text
    assert text.startswith("# Title\n\nIntro.")


def test_update_readme_leaves_a_readme_without_setup_markers_alone(
    tmp_path: Path,
) -> None:
    """A README with no setup markers gets no setup section appended."""
    readme = tmp_path / "README.md"
    readme.write_text(
        f"# Title\n\n{RESULTS_START}\n\nold\n\n{RESULTS_END}\n", encoding="utf-8"
    )

    update_readme(
        readme_path=readme,
        summary=_story_summary(),
        figure_paths=[],
        environment=_environment(),
    )

    text = readme.read_text(encoding="utf-8")
    assert "## Summary" in text
    assert "## Environment" not in text


def _cpu_summary() -> pd.DataFrame:
    """Return Vortex results where the array layout uses three threads."""
    rows = []
    for layout, wall, cpu in [("wide", 4.0, 4.0), ("fixed_array", 1.0, 3.0)]:
        rows.append(
            {
                "backend": "vortex",
                "layout": layout,
                "profile": "default",
                "rows": 10,
                "dimensions": 16,
                "operation": "matrix_materialization",
                "operation_parameter": "all",
                "median_seconds": wall,
                "median_cpu_seconds": cpu,
                "median_parallelism": cpu / wall,
                "artifact_bytes": 1_000,
                "repetitions": 3,
            }
        )
    return pd.DataFrame(rows)


def test_cpu_time_note_compares_layouts_in_cpu_time_for_multithreaded_backends() -> (
    None
):
    """CPU time counts every thread, so it gives a thread-fair comparison."""
    note = _cpu_time_note(_cpu_summary())

    assert note == (
        "Measured in CPU time, which counts every thread, matrix materialization "
        "with the array-like layout is 1.3x faster in Vortex (4x faster in wall "
        "time)."
    )


def test_cpu_time_note_is_absent_without_cpu_columns_or_multithreading() -> None:
    """Single-threaded runs and older results need no CPU-time note."""
    summary = _cpu_summary()

    assert _cpu_time_note(summary.drop(columns="median_cpu_seconds")) is None
    single = summary.assign(median_parallelism=1.0)
    assert _cpu_time_note(single) is None


def _noisy_story() -> pd.DataFrame:
    """Make Lance wide matrix materialization vary by 70% of its median."""
    summary = _story_summary()
    target = (
        (summary["backend"] == "lance")
        & (summary["layout"] == "wide")
        & (summary["operation"] == "matrix_materialization")
    )
    summary.loc[target, "q25_seconds"] = 0.2
    summary.loc[target, "q75_seconds"] = 0.9
    summary["noisy"] = (summary["q75_seconds"] - summary["q25_seconds"]) > (
        0.5 * summary["median_seconds"]
    )
    return summary


def test_key_findings_mark_cells_whose_measurement_is_noisy() -> None:
    """A cell built from a noisy measurement gets an asterisk."""
    markdown = array_vs_wide_markdown(array_vs_wide_table(_noisy_story()))

    assert "| Lance | `fixed_array` | -99.9%* | -90% | 0% | -10% |" in markdown
    assert "| Parquet | `fixed_array` | -75% | -5% | -50% | -30% |" in markdown


def test_noise_note_counts_noisy_measurements() -> None:
    """The note says how many measurements vary by more than half their median."""
    note = _noise_note(_noisy_story())

    assert note is not None
    assert note.startswith("An asterisk (`*`) marks a cell")
    assert "2 of 84 measurements (2%)" in note


def test_noise_note_is_absent_when_nothing_is_noisy() -> None:
    """Quiet results need no note, and neither do old results without the flag."""
    summary = _noisy_story()

    assert _noise_note(summary.assign(noisy=False)) is None
    assert _noise_note(summary.drop(columns="noisy")) is None


def test_access_path_note_confirms_layouts_are_in_range() -> None:
    """Every layout is within three times of the reference operation."""
    note = _access_path_note(_story_summary())

    assert note == (
        "Access path check: for every layout, matrix materialization takes at most "
        "3 times as long as a full read, and random rows take at most 3 times as "
        "long as matrix materialization."
    )


def test_access_path_note_names_a_layout_that_reads_too_much() -> None:
    """A layout that is 20 times slower than its full read is named."""
    summary = _story_summary()
    target = (
        (summary["backend"] == "lance")
        & (summary["layout"] == "wide")
        & (summary["operation"] == "matrix_materialization")
    )
    summary.loc[target, "median_seconds"] = 20.0

    note = _access_path_note(summary)

    assert note is not None
    assert note.startswith("Access path check: these layouts are outside the range.")
    assert (
        "Lance `wide`: matrix materialization takes 20x as long as a full read" in note
    )


def test_environment_table_shows_when_runs_were_pooled() -> None:
    """A pooled result says how many independent runs it contains."""
    environment = _environment() | {"runs": 2}

    table = environment_table(environment)

    assert "| Runs | 2 independent runs of the same code, pooled |" in table
    assert "Runs" not in environment_table(_environment())


def test_key_findings_say_feature_projection_is_the_wide_layouts_best_case() -> None:
    """A reader learns why array layouts can lose at feature projection."""
    section = render_results_section(summary=_story_summary(), figure_paths=[])

    assert (
        "Feature projection reads 8 of 16 features, which is the best case for a "
        "wide layout."
    ) in section


def test_sensitivity_table_says_what_the_conclusion_column_means() -> None:
    """The label names the side of wide and does not imply an unchanged effect."""
    summary = _compact_summary()
    section = render_results_section(summary=summary, figure_paths=[])

    assert "Same side of wide" in section or "Reverses" in section
    assert "Same direction" not in section
    assert "does not mean that the size of the effect stays the same" in section


def test_percent_change_is_always_a_signed_percentage() -> None:
    """Every value is a percentage with a sign, never a multiplier."""
    assert _percent_change(1.0) == "0%"
    assert _percent_change(1.004) == "0%"
    assert _percent_change(0.5) == "-50%"
    assert _percent_change(0.9) == "-10%"
    assert _percent_change(1.3) == "+30%"
    assert _percent_change(1.5) == "+50%"


def test_percent_change_rounds_large_increases_to_two_significant_digits() -> None:
    """A fivefold increase is +400%, and a 17-fold increase is +1,600%."""
    assert _percent_change(5.0) == "+400%"
    assert _percent_change(5.19) == "+420%"
    assert _percent_change(17.4) == "+1,600%"
    assert _percent_change(6.21) == "+520%"


def test_percent_change_keeps_decimals_for_almost_total_reductions() -> None:
    """A reduction of 99.98% must not round to 100%, which would read as zero."""
    assert _percent_change(0.001) == "-99.9%"
    assert _percent_change(0.0102) == "-99%"
    assert _percent_change(1 / 5400) == "-99.98%"
    assert _percent_change(0.99999) == "0%"


def test_key_findings_intro_explains_the_signs() -> None:
    """The intro says what a negative and a positive percentage mean."""
    section = render_results_section(summary=_story_summary(), figure_paths=[])

    assert "A negative percentage means faster or smaller." in section
    assert "A positive percentage means slower or larger." in section
    assert '"lower" mean' not in section
