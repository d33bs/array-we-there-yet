"""Tests for report generation."""

from pathlib import Path

import pandas as pd
import pytest

from array_we_there_yet.report import (
    SERIES_COLORS,
    _direction_title,
    _errorbar_interval,
    _facet_grid_shape,
    _facet_panels,
    _ordered_backends,
    _parallelism_note,
    _ratio_axis_label,
    _series_legend_handle,
    _time_summary_ratios,
    _with_reference_series,
    csv_wide_ratio_table,
    default_profile,
    encoding_table,
    profile_comparison_table,
    profile_sensitivity_table,
    ratio_table,
    render_results_section,
    write_combined_facet_overview,
    write_figures,
    write_profile_figure,
    write_profile_tables,
    write_ratio_tables,
)


def _figure_summary() -> pd.DataFrame:
    """Return a small two-backend summary for figure and README tests."""
    return pd.DataFrame(
        [
            {
                "backend": backend,
                "layout": layout,
                "dataset": "synthetic",
                "rows": 10,
                "dimensions": dimensions,
                "dtype": "float32",
                "operation": operation,
                "operation_parameter": "all",
                "threads": 1,
                "compression": "none",
                "median_seconds": seconds,
                "q25_seconds": seconds,
                "q75_seconds": seconds,
                "artifact_bytes": artifact_bytes,
                "repetitions": 1,
            }
            for backend in ["csv", "parquet"]
            for dimensions in [4, 8]
            for operation in ["write", "matrix_materialization", "mixed_retrieval"]
            for layout, seconds, artifact_bytes in [
                ("wide", 2.0, 2_000),
                ("fixed_array", 1.0, 1_000),
            ]
        ]
    )


def test_write_figures_puts_the_csv_wide_comparison_first(tmp_path: Path) -> None:
    """The main figure compares every layout with CSV wide."""
    names = [path.name for path in write_figures(_figure_summary(), tmp_path)]

    assert names[:3] == [
        "combined_facet_overview.png",
        "wide_layouts_facet_overview.png",
        "backend_wide_facet_overview.png",
    ]
    assert "write_absolute_comparison.png" in names
    assert "matrix_materialization_absolute_comparison.png" in names
    assert "parquet_performance_tracking.png" not in names
    assert "csv_wide_facet_overview.png" not in names


def test_results_section_has_a_professional_structure(tmp_path: Path) -> None:
    """The results section uses ordered headings and no filler labels."""
    summary = _figure_summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    headings = [line for line in section.splitlines() if line.startswith("#")]
    assert headings == [
        "## Summary",
        "## Plots",
        "### How to read the figures",
        "### Every layout against CSV wide",
        "### Array-like layouts against their own wide layout",
        "### Wide layouts",
        "## Real-world example",
        "### One use",
        "### Savings over 1,000 uses",
        "### Streaming a Parquet file",
        "## Detailed results",
        "### Key findings",
    ]
    for filler in [
        "Primary facet plots",
        "starter",
        "Use the plot images below",
        "Access paths:",
        "Detailed result files:",
        "### Result files",
        "### Environment",
        "### Backends and access paths",
        "Parquet tracking",
        "parquet_performance",
        "dotted",
        "Dotted",
    ]:
        assert filler not in section


def test_results_section_describes_the_run_and_findings(tmp_path: Path) -> None:
    """The opening paragraph states the data, and key findings are a table."""
    summary = _figure_summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    assert "synthetic data with 10 rows and 4 to 8 features" in section
    assert "median of 1 repetition" in section
    assert "Geometric Mean Time is the geometric mean" in section
    assert "Median Time" not in section
    assert "compares the array-like layout with the wide layout of the same" in section
    assert (
        "| Backend | Layout | Matrix materialization | Feature projection "
        "| Write | Storage size |"
    ) in section
    assert "| CSV | `fixed_array` | -50% | n/a | -50% | -50% |" in section


def test_results_section_explains_figures_once(tmp_path: Path) -> None:
    """Shared figure terms appear once, not in every caption."""
    summary = _figure_summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    assert section.count("q25-to-q75") == 1
    assert section.count("log scale") == 1
    assert "Overall Time Index" not in section
    for figure_number in range(1, 4):
        assert f"Figure {figure_number}." in section
    for image in [
        "combined_facet_overview.png",
        "wide_layouts_facet_overview.png",
        "backend_wide_facet_overview.png",
    ]:
        assert f"(figures/{image})" in section.replace(str(tmp_path), "figures")


def test_ratio_tables_are_written(tmp_path: Path) -> None:
    """Ratio tables are written only as Parquet."""
    summary = _figure_summary()

    ratio_table_paths = write_ratio_tables(summary, tmp_path)

    assert {path.name for path in ratio_table_paths} == {
        "ratio_summary.parquet",
        "csv_wide_ratio_summary.parquet",
        "wide_layout_ratio_summary.parquet",
    }
    assert all(path.exists() for path in ratio_table_paths)
    assert not list(tmp_path.glob("*.csv"))


def test_ordered_backends_puts_csv_first_and_fast_formats_later() -> None:
    """Plot panels use the requested benchmark-reading order."""
    assert _ordered_backends(
        ["lance", "vortex", "tiledb", "zarr", "duckdb", "csv", "parquet"]
    ) == [
        "csv",
        "parquet",
        "duckdb",
        "zarr",
        "tiledb",
        "vortex",
        "lance",
    ]


def test_combined_facet_grid_stays_compact() -> None:
    """Combined overview uses no more than three facets per row."""
    assert _facet_grid_shape(10) == (4, 3)
    assert _direction_title("Write", better="lower") == "Write\n(lower is better)"
    assert _ratio_axis_label("time_ratio") == "Time ratio (array / wide)"
    assert _ratio_axis_label("artifact_size_ratio") == "Size ratio (array / wide)"


def test_time_summary_ratios_use_the_geometric_mean() -> None:
    """Summary facets average time ratios with the geometric mean."""
    expected_geometric_mean = (2.0 * 0.5 * 0.25) ** (1 / 3)
    ratios = pd.DataFrame(
        [
            {
                "backend": "parquet",
                "layout": "fixed_array",
                "dimensions": 8,
                "operation": operation,
                "time_ratio": ratio,
            }
            for operation, ratio in [
                ("write", 2.0),
                ("full_read", 0.5),
                ("matrix_materialization", 0.25),
            ]
        ]
    )

    summary = _time_summary_ratios(ratios)

    assert summary.loc[0, "geometric_mean_time_ratio"] == pytest.approx(
        expected_geometric_mean
    )


def test_ratio_table_carries_conservative_time_error_intervals() -> None:
    """Ratio tables keep repeat dispersion for error bars."""
    expected_time_ratio = 0.5
    expected_time_ratio_q25 = 0.2
    expected_time_ratio_q75 = 1.5
    summary = pd.DataFrame(
        [
            {
                "backend": "parquet",
                "layout": "wide",
                "dimensions": 8,
                "operation": "mixed_retrieval",
                "operation_parameter": "3_rows_2_features",
                "median_seconds": 4.0,
                "q25_seconds": 2.0,
                "q75_seconds": 5.0,
                "artifact_bytes": 200,
            },
            {
                "backend": "parquet",
                "layout": "fixed_array",
                "dimensions": 8,
                "operation": "mixed_retrieval",
                "operation_parameter": "3_rows_2_features",
                "median_seconds": 2.0,
                "q25_seconds": 1.0,
                "q75_seconds": 3.0,
                "artifact_bytes": 100,
            },
        ]
    )

    ratios = ratio_table(summary)

    assert ratios.loc[0, "time_ratio"] == expected_time_ratio
    assert ratios.loc[0, "time_ratio_q25"] == expected_time_ratio_q25
    assert ratios.loc[0, "time_ratio_q75"] == expected_time_ratio_q75
    assert _errorbar_interval(ratios, "time_ratio") == [[0.3], [1.0]]
    assert _errorbar_interval(summary, "median_seconds") == [[2.0, 1.0], [1.0, 1.0]]


def _csv_wide_summary() -> pd.DataFrame:
    """Return a small summary with CSV wide and other layouts."""
    return pd.DataFrame(
        [
            {
                "backend": backend,
                "layout": layout,
                "dimensions": 8,
                "operation": operation,
                "operation_parameter": "all",
                "median_seconds": seconds,
                "q25_seconds": seconds / 2,
                "q75_seconds": seconds * 2,
                "artifact_bytes": artifact_bytes,
            }
            for operation in ["write", "full_read"]
            for backend, layout, seconds, artifact_bytes in [
                ("csv", "wide", 8.0, 800),
                ("csv", "json_array", 16.0, 1_600),
                ("parquet", "wide", 2.0, 400),
                ("parquet", "fixed_array", 1.0, 200),
                ("zarr", "wide", 4.0, 600),
                ("zarr", "zarr_matrix", 1.0, 300),
            ]
        ]
    )


def test_csv_wide_ratio_table_keeps_only_csv_and_parquet_wide_layouts() -> None:
    """Each layout is divided by CSV wide, and only Parquet wide stays as wide."""
    expected_parquet_wide_ratio = 0.25
    expected_size_ratio = 0.5
    expected_csv_wide_seconds = 8.0
    ratios = csv_wide_ratio_table(_csv_wide_summary())

    layouts = {tuple(pair) for pair in ratios[["backend", "layout"]].to_numpy()}
    assert layouts == {
        ("csv", "json_array"),
        ("parquet", "wide"),
        ("parquet", "fixed_array"),
        ("zarr", "zarr_matrix"),
    }
    parquet_wide = ratios[
        (ratios["backend"] == "parquet")
        & (ratios["layout"] == "wide")
        & (ratios["operation"] == "write")
    ].iloc[0]
    assert parquet_wide["time_ratio"] == expected_parquet_wide_ratio
    assert parquet_wide["time_ratio_q25"] == expected_parquet_wide_ratio / 4
    assert parquet_wide["time_ratio_q75"] == expected_parquet_wide_ratio * 4
    assert parquet_wide["artifact_size_ratio"] == expected_size_ratio
    assert parquet_wide["csv_wide_median_seconds"] == expected_csv_wide_seconds


def test_csv_wide_ratio_table_is_empty_without_csv_wide() -> None:
    """Runs without a CSV wide baseline produce no CSV wide ratios."""
    summary = _csv_wide_summary()
    summary = summary[~((summary["backend"] == "csv") & (summary["layout"] == "wide"))]

    assert csv_wide_ratio_table(summary).empty


def test_ratio_axis_label_names_the_baseline() -> None:
    """Axis labels can name a baseline other than the backend wide layout."""
    assert _ratio_axis_label("time_ratio", baseline="vs CSV wide") == (
        "Time ratio (vs CSV wide)"
    )
    assert _ratio_axis_label("artifact_size_ratio", baseline="vs CSV wide") == (
        "Size ratio (vs CSV wide)"
    )


def test_reference_series_plots_the_baseline_at_one() -> None:
    """CSV wide appears as a flat series at 1.0 next to the compared layouts."""
    ratios = csv_wide_ratio_table(_csv_wide_summary())

    augmented = _with_reference_series(ratios, backend="csv", layout="wide")

    reference = augmented[
        (augmented["backend"] == "csv") & (augmented["layout"] == "wide")
    ]
    assert len(reference) == len(ratios[["dimensions", "operation"]].drop_duplicates())
    for column in [
        "time_ratio",
        "time_ratio_q25",
        "time_ratio_q75",
        "artifact_size_ratio",
    ]:
        assert set(reference[column]) == {1.0}
    assert len(augmented) == len(ratios) + len(reference)
    assert len(ratios[ratios["layout"] == "wide"]) > 0


def test_reference_series_is_not_added_twice() -> None:
    """Adding the reference series again leaves the frame unchanged."""
    ratios = csv_wide_ratio_table(_csv_wide_summary())
    once = _with_reference_series(ratios, backend="csv", layout="wide")

    twice = _with_reference_series(once, backend="csv", layout="wide")

    assert len(twice) == len(once)


def _parallelism_summary() -> pd.DataFrame:
    """Return a summary with measured parallelism for three layouts."""
    return pd.DataFrame(
        [
            {"backend": backend, "layout": layout, "operation": op, "value": value}
            for backend, layout, op, value in [
                ("csv", "wide", "write", 1.0),
                ("csv", "wide", "full_read", 1.0),
                ("vortex", "fixed_array", "write", 3.0),
                ("vortex", "fixed_array", "full_read", 8.2),
                ("lance", "wide", "write", 1.0),
                ("lance", "wide", "full_read", 3.8),
            ]
        ]
    ).rename(columns={"value": "median_parallelism"})


def test_parallelism_note_names_layouts_that_used_more_than_one_thread() -> None:
    """The note lists multi-threaded layouts with their median and maximum."""
    note = _parallelism_note(_parallelism_summary())

    assert note is not None
    assert "Vortex `fixed_array` (median 5.6, up to 8.2)" in note
    assert "Lance `wide` (median 2.4, up to 3.8)" in note
    assert "CSV `wide`" not in note
    assert "CPU time divided by wall time" in note


def test_parallelism_note_is_absent_for_single_threaded_runs() -> None:
    """No note appears when every layout stayed near one thread."""
    summary = _parallelism_summary()
    summary["median_parallelism"] = 1.0

    assert _parallelism_note(summary) is None
    assert _parallelism_note(summary.drop(columns="median_parallelism")) is None


def test_results_section_places_the_parallelism_note_after_findings(
    tmp_path: Path,
) -> None:
    """The note sits under Key findings, after the plots and the example."""
    summary = _figure_summary()
    summary["median_parallelism"] = 4.0
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    guide = section.index("### How to read the figures")
    findings = section.index("### Key findings")
    note = section.index("used more than one thread")
    assert guide < findings < note


def _all_operations_summary() -> pd.DataFrame:
    """Return a summary that covers every operation for CSV and Parquet."""
    operations = [
        "write",
        "full_read",
        "matrix_materialization",
        "random_rows",
        "feature_projection",
        "mixed_retrieval",
        "vector_norm",
    ]
    return pd.DataFrame(
        [
            {
                "backend": backend,
                "layout": layout,
                "dimensions": dimensions,
                "operation": operation,
                "operation_parameter": "all",
                "median_seconds": seconds,
                "q25_seconds": seconds,
                "q75_seconds": seconds,
                "artifact_bytes": artifact_bytes,
            }
            for operation in operations
            for dimensions in [4, 8]
            for backend, layout, seconds, artifact_bytes in [
                ("csv", "wide", 4.0, 800),
                ("csv", "json_array", 8.0, 900),
                ("parquet", "wide", 2.0, 400),
                ("parquet", "fixed_array", 1.0, 200),
            ]
        ]
    )


def test_facet_panels_fill_exactly_a_three_by_three_grid() -> None:
    """Each facet figure has seven operations, one time summary, and storage."""
    ratios = csv_wide_ratio_table(_all_operations_summary())

    panels = _facet_panels(ratios=ratios, baseline="vs CSV wide")

    titles = [title.split("\n")[0] for title, *_ in panels]
    assert titles == [
        "Write",
        "Full Read",
        "Matrix Materialization",
        "Random Rows",
        "Feature Projection",
        "Mixed Retrieval",
        "Vector Norm",
        "Geometric Mean Time",
        "Storage Size",
    ]
    assert _facet_grid_shape(len(panels)) == (3, 3)


def _profile_summary() -> pd.DataFrame:
    """Return a summary with default and compact profiles for two layouts."""
    rows = []
    for layout in ["wide", "fixed_array"]:
        for profile, size, matrix, write in [
            ("default", 1_000, 1.0, 2.0),
            ("compact", 500, 1.5, 8.0),
        ]:
            for dimensions in [8, 16]:
                for operation, seconds in [
                    ("matrix_materialization", matrix),
                    ("feature_projection", matrix),
                    ("write", write),
                ]:
                    rows.append(
                        {
                            "backend": "parquet",
                            "layout": layout,
                            "profile": profile,
                            "rows": 10,
                            "dimensions": dimensions,
                            "operation": operation,
                            "operation_parameter": "all",
                            "median_seconds": seconds,
                            "q25_seconds": seconds,
                            "q75_seconds": seconds,
                            "artifact_bytes": size,
                            "repetitions": 1,
                        }
                    )
    return pd.DataFrame(rows)


def test_default_profile_keeps_only_default_rows() -> None:
    """Only default-profile rows feed the existing tables and figures."""
    summary = _profile_summary()

    default = default_profile(summary)

    assert "profile" not in default.columns
    assert len(default) == len(summary) // 2
    assert default_profile(default) is default


def test_ratio_tables_ignore_compact_rows() -> None:
    """Compact rows do not duplicate the ratio table rows."""
    summary = _profile_summary()

    mixed = ratio_table(summary)
    default_only = ratio_table(default_profile(summary))

    assert len(mixed) == len(default_only)
    assert (
        mixed["time_ratio"].to_numpy() == default_only["time_ratio"].to_numpy()
    ).all()


def test_profile_comparison_table_divides_compact_by_default() -> None:
    """Each layout reports compact-to-default ratios at the largest feature count."""
    expected_dimensions = 16
    expected_size_ratio = 0.5
    expected_matrix_ratio = 1.5
    expected_write_ratio = 4.0
    expected_bytes_per_value = 1_000 / (10 * 16)

    table = profile_comparison_table(_profile_summary())

    assert list(table["layout"]) == ["wide", "fixed_array"]
    fixed = table[table["layout"] == "fixed_array"].iloc[0]
    assert fixed["dimensions"] == expected_dimensions
    assert fixed["size_ratio"] == expected_size_ratio
    assert fixed["matrix_materialization_ratio"] == expected_matrix_ratio
    assert fixed["write_ratio"] == expected_write_ratio
    assert fixed["default_bytes_per_value"] == expected_bytes_per_value


def test_profile_comparison_table_is_empty_without_compact_rows() -> None:
    """Runs without a compact profile have nothing to compare."""
    summary = _profile_summary()
    summary = summary[summary["profile"] == "default"]

    assert profile_comparison_table(summary).empty
    assert profile_comparison_table(summary.drop(columns="profile")).empty


def test_write_profile_figure_needs_compact_rows(tmp_path: Path) -> None:
    """The size-versus-time figure exists only when a compact profile ran."""
    summary = _profile_summary()

    figure = write_profile_figure(summary, tmp_path)
    missing = write_profile_figure(summary[summary["profile"] == "default"], tmp_path)

    assert figure == tmp_path / "encoding_profiles.png"
    assert figure.exists()
    assert missing is None


def test_encoding_table_lists_each_profile_with_bytes_per_value() -> None:
    """The encodings table joins observed encodings to stored bytes per value."""
    summary = _profile_summary()
    encodings = pd.DataFrame(
        [
            {
                "backend": "parquet",
                "layout": "fixed_array",
                "profile": profile,
                "dimensions": 16,
                "encoding": text,
            }
            for profile, text in [
                ("default", "snappy; PLAIN"),
                ("compact", "zstd; BYTE_STREAM_SPLIT"),
            ]
        ]
    )

    table = encoding_table(encodings, summary)

    assert (
        "| Backend | Layout | Profile | Compression and encodings | Bytes per value |"
        in table
    )
    assert "| Parquet | `fixed_array` | default | snappy; PLAIN | 6.25 |" in table
    assert (
        "| Parquet | `fixed_array` | compact | zstd; BYTE_STREAM_SPLIT | 3.12 |"
        in table
    )


def _profile_encodings() -> pd.DataFrame:
    """Return observed encodings for the profile summary."""
    return pd.DataFrame(
        [
            {
                "backend": "parquet",
                "layout": layout,
                "profile": profile,
                "dimensions": 16,
                "encoding": text,
            }
            for layout in ["wide", "fixed_array"]
            for profile, text in [("default", "snappy; PLAIN"), ("compact", "zstd")]
        ]
    )


def test_results_section_puts_encoding_sensitivity_before_the_appendix(
    tmp_path: Path,
) -> None:
    """Sensitivity and observed encodings lead. The profile plot is an appendix."""
    summary = _profile_summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(
        summary=summary,
        figure_paths=paths,
        encodings=_profile_encodings(),
    )

    headings = [line for line in section.splitlines() if line.startswith("### ")]
    assert headings.index("### Key findings") < headings.index(
        "### Encoding sensitivity"
    )
    assert headings.index("### Encoding sensitivity") + 1 == headings.index(
        "### Encodings observed"
    )
    assert "### Parquet tracking" not in headings
    assert headings[-1] == "### Appendix: storage size against time"
    assert "Encoding profiles" not in headings
    appendix = section[section.index("### Appendix: storage size against time") :]
    assert "synthetic" in appendix
    assert "not a ranking" in appendix
    assert "| Parquet `fixed_array` | -50% | +50% | +300% |" in appendix
    assert "![Storage size against time" in appendix
    assert ".csv" not in section
    assert "| Parquet | `fixed_array` | compact | zstd | 3.12 |" in section
    assert "Most size differences between the binary formats" in section


def test_results_section_states_the_sensitivity_conclusion(tmp_path: Path) -> None:
    """The sensitivity table compares array-to-wide ratios under both profiles."""
    summary = _profile_summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    assert (
        "| Backend | Layout | Storage size | Matrix materialization "
        "| Feature projection | Write | Conclusion |"
    ) in section
    assert (
        "| Parquet | `fixed_array` | 0% → 0% | 0% → 0% | 0% → 0% | 0% → 0% |"
    ) in section
    assert "first with the default profile and then with the compact profile" in section


def test_results_section_skips_profiles_without_a_compact_run(tmp_path: Path) -> None:
    """Runs with only the default profile show no profile sections."""
    summary = _figure_summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    assert "Encoding profiles" not in section
    assert "profile_comparison" not in section
    assert "encodings.parquet" not in section


def _sensitivity_summary() -> pd.DataFrame:
    """Return a summary where the compact profile reverses one conclusion."""
    rows = []
    for profile, wide, array in [
        ("default", {"size": 2000, "time": 4.0}, {"size": 1000, "time": 2.0}),
        ("compact", {"size": 1000, "time": 2.0}, {"size": 1200, "time": 3.0}),
    ]:
        for layout, values in [("wide", wide), ("fixed_array", array)]:
            for operation in ["write", "matrix_materialization", "feature_projection"]:
                rows.append(
                    {
                        "backend": "parquet",
                        "layout": layout,
                        "profile": profile,
                        "rows": 10,
                        "dimensions": 16,
                        "operation": operation,
                        "operation_parameter": "all",
                        "median_seconds": values["time"],
                        "q25_seconds": values["time"],
                        "q75_seconds": values["time"],
                        "artifact_bytes": values["size"],
                        "repetitions": 1,
                    }
                )
    return pd.DataFrame(rows)


def test_profile_sensitivity_table_flags_a_conclusion_that_reverses() -> None:
    """A ratio that crosses 1.0 between profiles is marked as reversed."""
    expected_default_size = 0.5
    expected_compact_size = 1.2

    table = profile_sensitivity_table(_sensitivity_summary())

    row = table.iloc[0]
    assert (row["backend"], row["layout"]) == ("parquet", "fixed_array")
    assert row["storage_size_default"] == expected_default_size
    assert row["storage_size_compact"] == expected_compact_size
    assert row["write_default"] == expected_default_size
    assert bool(row["holds"]) is False


def test_profile_sensitivity_table_marks_a_stable_conclusion() -> None:
    """Ratios that stay below 1.0 in both profiles are marked as holding."""
    summary = _sensitivity_summary()
    compact_array = (summary["profile"] == "compact") & (
        summary["layout"] == "fixed_array"
    )
    summary.loc[compact_array, ["median_seconds", "q25_seconds", "q75_seconds"]] = 0.5
    summary.loc[compact_array, "artifact_bytes"] = 400

    table = profile_sensitivity_table(summary)

    assert bool(table.iloc[0]["holds"]) is True


def test_profile_sensitivity_table_needs_wide_and_array_in_both_profiles() -> None:
    """Backends without a compact wide and array pair are skipped."""
    summary = _sensitivity_summary()
    summary = summary[
        ~((summary["profile"] == "compact") & (summary["layout"] == "wide"))
    ]

    assert profile_sensitivity_table(summary).empty
    assert profile_sensitivity_table(summary.drop(columns="profile")).empty


def test_profile_tables_are_written_only_as_parquet(tmp_path: Path) -> None:
    """The compact-versus-default comparison has no CSV copy."""
    paths = write_profile_tables(_profile_summary(), tmp_path)

    assert [path.name for path in paths] == ["profile_comparison.parquet"]
    assert not list(tmp_path.glob("*.csv"))
    assert write_profile_tables(_figure_summary(), tmp_path) == []


def test_series_legend_handle_shows_only_line_color_and_style() -> None:
    """Legend keys are plain lines: no markers and no error-bar caps."""
    wide = _series_legend_handle(("csv", "wide"))
    array = _series_legend_handle(("parquet", "fixed_array"))

    assert wide.get_label() == "csv/wide"
    assert wide.get_color() == SERIES_COLORS[("csv", "wide")]
    assert wide.get_linestyle() == "--"
    assert array.get_label() == "parquet/fixed_array"
    assert array.get_color() == SERIES_COLORS[("parquet", "fixed_array")]
    assert array.get_linestyle() == "-"
    for handle in (wide, array):
        assert handle.get_marker() == "None"


def test_facet_figure_legend_uses_plain_line_keys(tmp_path: Path) -> None:
    """The saved figure builds its shared legend from plain line keys."""
    ratios = csv_wide_ratio_table(_all_operations_summary())

    figure = write_combined_facet_overview(_all_operations_summary(), tmp_path)

    assert figure is not None
    assert figure.exists()
    labels = {"/".join(pair) for pair in ratios[["backend", "layout"]].to_numpy()}
    assert "parquet/wide" in labels


def test_readme_text_refers_only_to_sections_that_exist(tmp_path: Path) -> None:
    """The static README names no section that the report does not produce."""
    text = (Path(__file__).parents[1] / "README.md").read_text(encoding="utf-8")
    readme = " ".join(text.split())

    assert "The Encoding profiles section" not in readme
    assert "Repetitions." in readme
    assert "A full run takes about" in readme
