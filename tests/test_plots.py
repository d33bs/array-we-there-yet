"""Tests for the JSON specs that the report page draws with Plotly."""

import json

import pandas as pd
import pytest

from array_we_there_yet.plots import (
    catalog,
    explorer_spec,
    facet_specs,
    profile_spec,
    real_world_spec,
    row_scaling_spec,
)
from tests.test_real_world import _summary as _real_world_summary
from tests.test_report import _figure_summary
from tests.test_scale_report import _sweep


def _series(panel: dict, key: str) -> dict:
    return next(series for series in panel["series"] if series["key"] == key)


def _panel(spec: dict, panel_id: str) -> dict:
    return next(panel for panel in spec["panels"] if panel["id"] == panel_id)


def test_facet_specs_hold_the_three_views_in_page_order() -> None:
    """Every layout against CSV wide comes first, as on the page."""
    specs = facet_specs(_figure_summary())

    assert list(specs) == ["combined", "backend_wide", "wide_layouts"]
    assert specs["combined"]["baseline"] == "CSV wide"
    assert specs["combined"]["axisNote"] == "vs CSV wide"
    assert specs["backend_wide"]["axisNote"] == "array / wide"
    assert specs["backend_wide"]["baseline"] == "the wide layout of the same backend"


def test_a_facet_has_a_panel_per_operation_then_the_summaries() -> None:
    """The seven operations come first, then geometric mean time and size."""
    spec = facet_specs(_figure_summary())["combined"]

    assert [panel["id"] for panel in spec["panels"]] == [
        "write",
        "matrix_materialization",
        "mixed_retrieval",
        "geometric_mean_time",
        "storage_size",
    ]
    assert _panel(spec, "matrix_materialization")["title"] == "Matrix Materialization"
    assert _panel(spec, "storage_size")["title"] == "Storage Size"


def test_series_carry_the_values_the_page_needs_to_draw_and_filter() -> None:
    """Each series has a key, a label, a style, and sorted points with a range."""
    spec = facet_specs(_figure_summary())["backend_wide"]

    series = _series(_panel(spec, "matrix_materialization"), "parquet/fixed_array")
    assert series["label"] == "Parquet fixed_array"
    assert series["backend"] == "parquet"
    assert series["layout"] == "fixed_array"
    assert series["dash"] == "solid"
    assert series["color"].startswith("#")
    assert series["x"] == [4, 8]
    assert series["y"] == pytest.approx([0.5, 0.5])
    assert series["low"] == pytest.approx([0.5, 0.5])
    assert series["high"] == pytest.approx([0.5, 0.5])


def test_wide_layouts_are_dashed_and_csv_wide_is_a_flat_reference() -> None:
    """CSV wide appears as a line at 1.0 so that the reader sees the baseline."""
    spec = facet_specs(_figure_summary())["combined"]

    reference = _series(_panel(spec, "write"), "csv/wide")
    assert reference["y"] == [1.0, 1.0]
    assert reference["dash"] == "dash"
    assert _series(_panel(spec, "write"), "parquet/wide")["dash"] == "dash"


def test_summary_panels_have_no_range() -> None:
    """The geometric mean and the size are single values, so they have no range."""
    spec = facet_specs(_figure_summary())["backend_wide"]

    for panel_id in ["geometric_mean_time", "storage_size"]:
        series = _series(_panel(spec, panel_id), "csv/fixed_array")
        assert series["low"] is None
        assert series["high"] is None
    assert _series(_panel(spec, "storage_size"), "csv/fixed_array")["y"] == (
        pytest.approx([0.5, 0.5])
    )


def test_specs_are_plain_json() -> None:
    """The page embeds the specs, so they must not hold NumPy types."""
    text = json.dumps(facet_specs(_figure_summary()))

    assert "NaN" not in text


def test_facets_without_ratios_are_left_out() -> None:
    """A summary with one layout has nothing to compare."""
    summary = _figure_summary()
    summary = summary[(summary["backend"] == "csv") & (summary["layout"] == "wide")]

    assert facet_specs(summary) == {}


def test_row_scaling_spec_has_a_panel_per_operation() -> None:
    """Time against rows for matrix, random rows, and projection."""
    spec = row_scaling_spec(_sweep())
    assert spec is not None

    assert [panel["id"] for panel in spec["panels"]] == [
        "matrix_materialization",
        "random_rows",
        "feature_projection",
    ]
    series = _series(_panel(spec, "random_rows"), "parquet/fixed_array")
    assert series["x"] == [2_000, 200_000]
    assert series["y"] == pytest.approx([0.005, 0.1])
    assert spec["dimensions"] == pytest.approx(1_024)


def test_row_scaling_spec_needs_two_row_counts() -> None:
    """A sweep with one row count draws nothing."""
    sweep = _sweep()

    assert row_scaling_spec(None) is None
    assert row_scaling_spec(sweep[sweep["rows"] == sweep["rows"].min()]) is None


def test_profile_spec_pairs_default_and_compact_points() -> None:
    """Each layout has a default and a compact point, sized against raw float32."""
    spec = profile_spec(_real_world_summary())
    assert spec is not None

    assert [panel["id"] for panel in spec["panels"]] == [
        "matrix_materialization",
        "write",
    ]
    points = _panel(spec, "write")["points"]
    csv_default = next(
        point
        for point in points
        if point["key"] == "csv/wide" and point["profile"] == "default"
    )
    assert csv_default["x"] == pytest.approx(187_500_000 / (2_000 * 8_192 * 4))
    assert csv_default["y"] == pytest.approx(1.0)
    assert {point["profile"] for point in points} == {"default", "compact"}


def test_profile_spec_is_absent_without_a_compact_profile() -> None:
    """Without compact results there is nothing to compare."""
    summary = _real_world_summary()
    summary = summary[summary["profile"] == "default"]

    assert profile_spec(summary) is None


def test_real_world_spec_gives_sizes_and_reads_at_the_example_file_size() -> None:
    """The page scales these to the file size that a reader picks."""
    spec = real_world_spec(_real_world_summary(), None)
    assert spec is not None

    assert spec["datasetGb"] == pytest.approx(1.5)
    assert spec["downloadMbPerSecond"] == pytest.approx(100)
    assert spec["egressDollarsPerGb"] == pytest.approx(0.09)
    assert spec["uses"] == pytest.approx(1_000)
    layouts = {layout["label"]: layout for layout in spec["layouts"]}
    assert layouts["CSV wide"]["sizeGb"] == pytest.approx(1.5)
    assert layouts["CSV wide"]["readSeconds"] == pytest.approx(8.0)
    assert layouts["Parquet fixed_array"]["sizeGb"] == pytest.approx(0.6)
    assert layouts["Parquet fixed_array (compact)"]["sizeGb"] == pytest.approx(0.48)
    assert not any("streamed" in label for label in layouts)


def test_real_world_spec_is_absent_without_the_example_data() -> None:
    """Without CSV wide there is no baseline."""
    assert real_world_spec(pd.DataFrame(), None) is None


def test_catalog_lists_backends_and_layouts_in_plot_order() -> None:
    """The filter bar groups layouts by backend, wide layout first."""
    summary = _figure_summary()

    result = catalog(summary)

    assert result["backends"] == [
        {"id": "csv", "label": "CSV"},
        {"id": "parquet", "label": "Parquet"},
    ]
    assert [layout["key"] for layout in result["layouts"]] == [
        "csv/wide",
        "csv/fixed_array",
        "parquet/wide",
        "parquet/fixed_array",
    ]
    assert result["layouts"][0]["dash"] == "dash"
    assert result["layouts"][1]["label"] == "CSV fixed_array"


def test_explorer_spec_holds_every_result_for_the_absolute_view() -> None:
    """One row per layout, profile, feature count, and operation."""
    spec = explorer_spec(_real_world_summary())

    assert spec["operations"][0] == {"id": "write", "title": "Write"}
    assert {"id": "storage_size", "title": "Storage Size"} in spec["operations"]
    assert spec["profiles"] == ["default", "compact"]
    row = next(
        row
        for row in spec["rows"]
        if row["key"] == "csv/wide"
        and row["profile"] == "compact"
        and row["operation"] == "matrix_materialization"
    )
    assert row["dimensions"] == pytest.approx(8_192)
    assert row["median"] == pytest.approx(1.25)
    assert row["bytes"] == pytest.approx(75_000_000)
    assert "NaN" not in json.dumps(spec)
