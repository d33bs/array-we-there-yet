"""JSON specs for the interactive plots on the report page.

Each function returns plain Python data. The page embeds the data as JSON and
draws it with Plotly, so the specs hold numbers and labels but no drawing code.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from array_we_there_yet.report import (
    BACKEND_DISPLAY_NAMES,
    CSV_WIDE_LABEL,
    EXAMPLE_DATASET_GB,
    EXAMPLE_DOWNLOAD_MB_PER_SECOND,
    EXAMPLE_EGRESS_DOLLARS_PER_GB,
    EXAMPLE_USES,
    OPERATIONS,
    ROW_SCALING_OPERATIONS,
    SERIES_COLORS,
    _example_name,
    _ordered_groups,
    _time_summary_ratios,
    _with_reference_series,
    csv_wide_ratio_table,
    default_profile,
    profile_comparison_table,
    ratio_table,
    real_world_table,
    wide_layout_ratio_table,
)

FLOAT32_BYTES = 4
NEUTRAL_COLOR = "#666666"


def series_label(backend: str, layout: str) -> str:
    """Return a backend and layout as plain text, such as ``Parquet fixed_array``."""
    return f"{BACKEND_DISPLAY_NAMES.get(backend, backend)} {layout}"


def _style(backend: str, layout: str) -> dict[str, str]:
    """Return the key, label, color, and line style of a layout."""
    return {
        "key": f"{backend}/{layout}",
        "label": series_label(backend, layout),
        "backend": backend,
        "layout": layout,
        "color": SERIES_COLORS.get((backend, layout), NEUTRAL_COLOR),
        "dash": "dash" if layout == "wide" else "solid",
    }


def _numbers(values: pd.Series, *, as_int: bool = False) -> list[float]:
    """Return a column as plain Python numbers."""
    return [int(value) if as_int else float(value) for value in values]


def _series(
    data: pd.DataFrame,
    *,
    x_column: str,
    value_column: str,
    range_columns: tuple[str, str] | None,
) -> list[dict[str, Any]]:
    """Return one series per layout, sorted by the x column."""
    series = []
    for (backend, layout), group in _ordered_groups(data):
        ordered = group.sort_values(x_column)
        has_range = range_columns is not None and set(range_columns).issubset(
            ordered.columns
        )
        series.append(
            {
                **_style(backend, layout),
                "x": _numbers(ordered[x_column], as_int=True),
                "y": _numbers(ordered[value_column]),
                "low": _numbers(ordered[range_columns[0]]) if has_range else None,
                "high": _numbers(ordered[range_columns[1]]) if has_range else None,
            }
        )
    return series


def _facet_panels(ratios: pd.DataFrame) -> list[dict[str, Any]]:
    """Return a panel for each operation, then geometric mean time and size."""
    panels = []
    for operation in OPERATIONS:
        data = ratios[ratios["operation"] == operation]
        if not data.empty:
            panels.append(
                {
                    "id": operation,
                    "title": operation.replace("_", " ").title(),
                    "yLabel": "Time ratio",
                    "series": _series(
                        data,
                        x_column="dimensions",
                        value_column="time_ratio",
                        range_columns=("time_ratio_q25", "time_ratio_q75"),
                    ),
                }
            )
    time_summary = _time_summary_ratios(ratios)
    if not time_summary.empty:
        panels.append(
            {
                "id": "geometric_mean_time",
                "title": "Geometric Mean Time",
                "yLabel": "Time ratio",
                "series": _series(
                    time_summary,
                    x_column="dimensions",
                    value_column="geometric_mean_time_ratio",
                    range_columns=None,
                ),
            }
        )
    storage = ratios[ratios["operation"] == "write"]
    if not storage.empty:
        panels.append(
            {
                "id": "storage_size",
                "title": "Storage Size",
                "yLabel": "Size ratio",
                "series": _series(
                    storage,
                    x_column="dimensions",
                    value_column="artifact_size_ratio",
                    range_columns=None,
                ),
            }
        )
    return panels


def facet_specs(summary: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return the three facet views, in page order, that have data to compare."""
    csv_reference = {"backend": "csv", "layout": "wide"}
    views = [
        (
            "combined",
            "Every layout against CSV wide",
            CSV_WIDE_LABEL,
            _with_reference_series_if_any(csv_wide_ratio_table(summary), csv_reference),
        ),
        (
            "backend_wide",
            "Array-like layouts against their own wide layout",
            "the wide layout of the same backend",
            ratio_table(summary),
        ),
        (
            "wide_layouts",
            "Wide layouts",
            CSV_WIDE_LABEL,
            _with_reference_series_if_any(
                wide_layout_ratio_table(summary), csv_reference
            ),
        ),
    ]
    specs = {}
    for view_id, title, baseline, ratios in views:
        panels = _facet_panels(ratios) if not ratios.empty else []
        if panels:
            note = "array / wide" if view_id == "backend_wide" else f"vs {baseline}"
            specs[view_id] = {
                "id": view_id,
                "title": title,
                "baseline": baseline,
                "axisNote": note,
                "panels": panels,
            }
    return specs


def _with_reference_series_if_any(
    ratios: pd.DataFrame, reference: dict[str, str]
) -> pd.DataFrame:
    """Add the reference layout as a flat line, unless there is nothing to compare."""
    if ratios.empty:
        return ratios
    return _with_reference_series(ratios, **reference)


def row_scaling_spec(sweep: pd.DataFrame | None) -> dict[str, Any] | None:
    """Return time against row count for each layout, or None without a sweep."""
    if sweep is None or sweep.empty:
        return None
    data = default_profile(sweep)
    if data["rows"].nunique() < 2:  # noqa: PLR2004
        return None
    panels = []
    for operation, title in ROW_SCALING_OPERATIONS:
        points = data[data["operation"] == operation]
        if points.empty:
            continue
        panels.append(
            {
                "id": operation,
                "title": title,
                "series": _series(
                    points,
                    x_column="rows",
                    value_column="median_seconds",
                    range_columns=("q25_seconds", "q75_seconds"),
                ),
            }
        )
    return {"dimensions": int(data["dimensions"].max()), "panels": panels}


def profile_spec(summary: pd.DataFrame) -> dict[str, Any] | None:
    """Return storage size against time for the default and compact profiles."""
    if profile_comparison_table(summary).empty:
        return None
    largest = summary["dimensions"].max()
    top = summary[summary["dimensions"] == largest]
    sizes = top[top["operation"] == "write"].set_index(["backend", "layout", "profile"])
    values = float(sizes["rows"].iloc[0]) * float(largest)
    panels = []
    for operation, title in [
        ("matrix_materialization", "Matrix Materialization"),
        ("write", "Write"),
    ]:
        timed = top[top["operation"] == operation].set_index(
            ["backend", "layout", "profile"]
        )
        points = []
        for (backend, layout), _ in _ordered_groups(
            top[["backend", "layout"]].drop_duplicates()
        ):
            for profile in ["default", "compact"]:
                key = (backend, layout, profile)
                if key in timed.index and key in sizes.index:
                    points.append(
                        {
                            **_style(backend, layout),
                            "profile": profile,
                            "x": float(sizes.loc[key, "artifact_bytes"])
                            / (values * FLOAT32_BYTES),
                            "y": float(timed.loc[key, "median_seconds"]),
                        }
                    )
        panels.append({"id": operation, "title": title, "points": points})
    return {"dimensions": int(largest), "panels": panels}


def real_world_spec(
    summary: pd.DataFrame,
    scaling: pd.DataFrame | None = None,
) -> dict[str, Any] | None:
    """Return the sizes and read times of the real-world example.

    The values are for a file of ``datasetGb``. The page scales them to the file
    size that a reader picks, and adds the download time and egress cost.
    """
    if summary.empty:
        return None
    table = real_world_table(summary, scaling)
    if table.empty:
        return None
    layouts = [
        {
            **_style(row["backend"], row["layout"]),
            "key": f"{row['backend']}/{row['layout']}/{row['profile']}",
            "label": _example_name(row).replace("`", ""),
            "profile": row["profile"],
            "sizeGb": float(row["size_gb"]),
            "readSeconds": float(row["read_seconds"]),
            "measured": bool(row["measured"]),
        }
        for row in table.to_dict("records")
        if row.get("variant") != "streamed"
    ]
    return {
        "datasetGb": EXAMPLE_DATASET_GB,
        "downloadMbPerSecond": EXAMPLE_DOWNLOAD_MB_PER_SECOND,
        "egressDollarsPerGb": EXAMPLE_EGRESS_DOLLARS_PER_GB,
        "uses": EXAMPLE_USES,
        "layouts": layouts,
    }


def catalog(summary: pd.DataFrame) -> dict[str, list[dict[str, str]]]:
    """Return the backends and layouts of the summary, in plot order.

    The filter bar uses this list, so every layout can be shown or hidden.
    """
    data = default_profile(summary)
    layouts = [
        _style(backend, layout) for (backend, layout), _ in _ordered_groups(data)
    ]
    backends = []
    for layout in layouts:
        entry = {
            "id": layout["backend"],
            "label": BACKEND_DISPLAY_NAMES.get(layout["backend"], layout["backend"]),
        }
        if entry not in backends:
            backends.append(entry)
    return {"backends": backends, "layouts": layouts}


def explorer_spec(summary: pd.DataFrame) -> dict[str, Any]:
    """Return every result, so that the page can show absolute values.

    Each row is one layout, profile, feature count, and operation.
    """
    if "profile" not in summary.columns:
        summary = summary.assign(profile="default")
    operations = [
        {"id": operation, "title": operation.replace("_", " ").title()}
        for operation in OPERATIONS
        if (summary["operation"] == operation).any()
    ]
    if (summary["operation"] == "write").any():
        operations.append({"id": "storage_size", "title": "Storage Size"})
    rows = [
        {
            **_style(row["backend"], row["layout"]),
            "profile": row["profile"],
            "dimensions": int(row["dimensions"]),
            "operation": row["operation"],
            "median": float(row["median_seconds"]),
            "low": float(row["q25_seconds"]),
            "high": float(row["q75_seconds"]),
            "bytes": int(row["artifact_bytes"]),
        }
        for row in summary.to_dict("records")
    ]
    profiles = [
        profile
        for profile in ["default", "compact"]
        if (summary["profile"] == profile).any()
    ]
    return {"operations": operations, "profiles": profiles, "rows": rows}
