"""Report generation for benchmark outputs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Literal, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

RESULTS_START = "<!-- array-we-there-yet-results:start -->"
RESULTS_END = "<!-- array-we-there-yet-results:end -->"
SETUP_START = "<!-- array-we-there-yet-setup:start -->"
SETUP_END = "<!-- array-we-there-yet-setup:end -->"
SIMILAR_LOW = 0.9
NOISY_RANGE = 0.5
ACCESS_PATH_LIMIT = 3
SIMILAR_HIGH = 1.1
RAW_FLOAT32_BYTES = 4
EXAMPLE_DATASET_GB = 1.5
EXAMPLE_DOWNLOAD_MB_PER_SECOND = 100
EXAMPLE_EGRESS_DOLLARS_PER_GB = 0.09
EXAMPLE_USES = 1_000
EXAMPLE_LAYOUTS = [
    ("csv", "wide", "default"),
    ("csv", "wide", "compact"),
    ("parquet", "wide", "default"),
    ("parquet", "fixed_array", "default"),
    ("parquet", "fixed_array", "compact"),
    ("duckdb", "duckdb_array", "default"),
    ("zarr", "zarr_matrix", "default"),
    ("tiledb", "tiledb_dense", "default"),
    ("vortex", "fixed_array", "default"),
    ("lance", "fixed_array", "default"),
]
THREAD_LABELS = {
    "arrow_cpu_threads": "Arrow CPU",
    "arrow_io_threads": "Arrow I/O",
    "duckdb_threads": "DuckDB",
    "tiledb_concurrency_level": "TileDB",
    "zarr_blosc_threads": "Zarr Blosc",
    "lance": "Lance",
    "vortex": "Vortex",
}
Panel = tuple[str, pd.DataFrame, str, str]


class PlotStyle(TypedDict, total=False):
    """Line style fields shared by plot and errorbar calls."""

    label: str
    color: str | None
    linestyle: str


OPERATIONS = [
    "write",
    "full_read",
    "matrix_materialization",
    "random_rows",
    "feature_projection",
    "mixed_retrieval",
    "vector_norm",
]
BACKEND_ORDER = ["csv", "parquet", "duckdb", "zarr", "tiledb", "vortex", "lance"]
BACKEND_DISPLAY_NAMES = {
    "csv": "CSV",
    "parquet": "Parquet",
    "duckdb": "DuckDB",
    "zarr": "Zarr",
    "tiledb": "TileDB",
    "vortex": "Vortex",
    "lance": "Lance",
}
BACKEND_PACKAGES = {
    "csv": "`pandas` CSV I/O",
    "parquet": "`pyarrow.parquet`",
    "duckdb": "`duckdb` Python package",
    "zarr": "`zarr` Python package",
    "tiledb": "`tiledb` Python package",
    "vortex": "`vortex-data` (`vortex` import)",
    "lance": "`lance` Python package",
}
LAYOUT_COLORS = {
    "wide": "#4D4D4D",
    "fixed_array": "#009E73",
    "delimited_array": "#0072B2",
    "json_array": "#D55E00",
    "duckdb_array": "#CC79A7",
    "zarr_matrix": "#56B4E9",
    "tiledb_dense": "#E69F00",
}
SERIES_COLORS = {
    ("csv", "delimited_array"): "#0072B2",
    ("csv", "json_array"): "#D55E00",
    ("parquet", "fixed_array"): "#009E73",
    ("duckdb", "duckdb_array"): "#CC79A7",
    ("zarr", "zarr_matrix"): "#56B4E9",
    ("tiledb", "tiledb_dense"): "#E69F00",
    ("vortex", "fixed_array"): "#999999",
    ("lance", "fixed_array"): "#000000",
    ("csv", "wide"): "#882255",
    ("parquet", "wide"): "#009E73",
    ("duckdb", "wide"): "#CC79A7",
    ("zarr", "wide"): "#56B4E9",
    ("tiledb", "wide"): "#E69F00",
    ("vortex", "wide"): "#999999",
    ("lance", "wide"): "#000000",
}
KEY_COLUMNS = ["dimensions", "operation", "operation_parameter"]
CSV_WIDE_LABEL = "CSV wide"
GAIN_MULTIPLIER_LIMIT = 0.1
LOSS_MULTIPLIER_LIMIT = 2.0
PARALLELISM_LIMIT = 2.0
WIDE_REFERENCE_BACKENDS = ("csv", "parquet")


def default_profile(summary: pd.DataFrame) -> pd.DataFrame:
    """Return the default-profile rows without the profile column."""
    if "profile" not in summary.columns:
        return summary
    return summary[summary["profile"] == "default"].drop(columns="profile")


def ratio_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate array-to-wide ratios for comparable backend operations."""
    summary = default_profile(summary)
    return _ratios_against(
        candidates=summary[summary["layout"] != "wide"],
        baseline=summary[summary["layout"] == "wide"],
        keys=["backend", *KEY_COLUMNS],
        prefix="wide",
    )


def csv_wide_ratio_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate ratios of every layout to the CSV wide layout.

    Parquet wide is the only other wide layout kept as a candidate.
    """
    summary = default_profile(summary)
    is_csv_wide = (summary["backend"] == "csv") & (summary["layout"] == "wide")
    return _ratios_against(
        candidates=summary[~is_csv_wide & _is_reference_layout(summary)],
        baseline=summary[is_csv_wide],
        keys=KEY_COLUMNS,
        prefix="csv_wide",
    )


def wide_layout_ratio_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate ratios of every backend's wide layout to the CSV wide layout."""
    summary = default_profile(summary)
    wide = summary[summary["layout"] == "wide"]
    is_csv = wide["backend"] == "csv"
    return _ratios_against(
        candidates=wide[~is_csv],
        baseline=wide[is_csv],
        keys=KEY_COLUMNS,
        prefix="csv_wide",
    )


def _is_reference_layout(frame: pd.DataFrame) -> pd.Series:
    """Return rows that are array-like or a CSV or Parquet wide layout."""
    return (frame["layout"] != "wide") | frame["backend"].isin(WIDE_REFERENCE_BACKENDS)


def _ratios_against(
    *,
    candidates: pd.DataFrame,
    baseline: pd.DataFrame,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Divide candidate timings and sizes by matching baseline rows."""
    value_columns = {
        "median_seconds": f"{prefix}_median_seconds",
        "q25_seconds": f"{prefix}_q25_seconds",
        "q75_seconds": f"{prefix}_q75_seconds",
        "artifact_bytes": f"{prefix}_artifact_bytes",
    }
    baseline = baseline[[*keys, *value_columns]].rename(columns=value_columns)
    ratios = candidates.merge(baseline, on=keys, how="inner")
    ratios["time_ratio"] = ratios["median_seconds"] / ratios[f"{prefix}_median_seconds"]
    ratios["time_ratio_q25"] = ratios["q25_seconds"] / ratios[f"{prefix}_q75_seconds"]
    ratios["time_ratio_q75"] = ratios["q75_seconds"] / ratios[f"{prefix}_q25_seconds"]
    ratios["artifact_size_ratio"] = (
        ratios["artifact_bytes"] / ratios[f"{prefix}_artifact_bytes"]
    )
    return ratios


def write_figures(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> list[Path]:
    """Write absolute comparison figures and compact ratio figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    all_profiles = summary
    summary = default_profile(summary)
    paths: list[Path] = []
    combined = write_combined_facet_overview(summary, figure_dir)
    if combined is not None:
        paths.append(combined)
    wide_layouts = write_wide_layouts_facet_overview(summary, figure_dir)
    if wide_layouts is not None:
        paths.append(wide_layouts)
    backend_wide = write_backend_wide_facet_overview(summary, figure_dir)
    if backend_wide is not None:
        paths.append(backend_wide)
    profiles = write_profile_figure(all_profiles, figure_dir)
    if profiles is not None:
        paths.append(profiles)
    paths.extend(write_absolute_figures(summary, figure_dir))
    paths.extend(write_ratio_figures(summary, figure_dir))
    return paths


def write_ratio_tables(
    summary: pd.DataFrame,
    output_dir: Path = Path("results"),
) -> list[Path]:
    """Write array-to-wide and CSV-wide ratio tables as Parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "ratio_summary": ratio_table(summary),
        "csv_wide_ratio_summary": csv_wide_ratio_table(summary),
        "wide_layout_ratio_summary": wide_layout_ratio_table(summary),
    }
    paths = []
    for name, table in tables.items():
        path = output_dir / f"{name}.parquet"
        table.to_parquet(path, index=False)
        paths.append(path)
    return paths


def write_combined_facet_overview(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> Path | None:
    """Write the main faceted overview with every layout compared with CSV wide."""
    ratios = csv_wide_ratio_table(summary)
    if ratios.empty:
        return None
    baseline = f"vs {CSV_WIDE_LABEL}"
    panels = _facet_panels(
        ratios=_with_reference_series(ratios, backend="csv", layout="wide"),
        baseline=baseline,
    )
    return _write_facet_figure(
        panels,
        figure_dir / "combined_facet_overview.png",
        baseline=baseline,
        legend_columns=4,
    )


def write_wide_layouts_facet_overview(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> Path | None:
    """Write the same facets for wide layouts only, compared with CSV wide."""
    ratios = wide_layout_ratio_table(summary)
    if ratios.empty:
        return None
    baseline = f"vs {CSV_WIDE_LABEL}"
    panels = _facet_panels(
        ratios=_with_reference_series(ratios, backend="csv", layout="wide"),
        baseline=baseline,
    )
    return _write_facet_figure(
        panels,
        figure_dir / "wide_layouts_facet_overview.png",
        baseline=baseline,
    )


def write_backend_wide_facet_overview(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> Path | None:
    """Write the same facets with each array-like layout compared with its own wide."""
    ratios = ratio_table(summary)
    panels = _facet_panels(ratios=ratios, baseline="array / wide")
    return _write_facet_figure(
        panels,
        figure_dir / "backend_wide_facet_overview.png",
        baseline="array / wide",
    )


def _facet_panels(*, ratios: pd.DataFrame, baseline: str) -> list[Panel]:
    """Return the nine panels of a facet figure.

    There is one panel for each of the seven operations, then the geometric mean
    time and the storage size. Together they fill a three-by-three grid.
    """
    time_label = _ratio_axis_label("time_ratio", baseline=baseline)
    panels: list[Panel] = []
    for operation in OPERATIONS:
        data = ratios[ratios["operation"] == operation]
        if not data.empty:
            panels.append(
                (
                    _direction_title(
                        operation.replace("_", " ").title(),
                        better="lower",
                    ),
                    data,
                    "time_ratio",
                    time_label,
                )
            )

    time_summary = _time_summary_ratios(ratios)
    if not time_summary.empty:
        panels.append(
            (
                _direction_title("Geometric Mean Time", better="lower"),
                time_summary,
                "geometric_mean_time_ratio",
                time_label,
            )
        )
    storage = ratios[ratios["operation"] == "write"]
    if not storage.empty:
        panels.append(
            (
                _direction_title("Storage Size", better="lower"),
                storage,
                "artifact_size_ratio",
                _ratio_axis_label("artifact_size_ratio", baseline=baseline),
            )
        )
    return panels


def _write_facet_figure(
    panels: list[Panel],
    path: Path,
    *,
    baseline: str,
    legend_columns: int = 3,
) -> Path | None:
    """Draw panels on a compact grid with one shared legend."""
    if not panels:
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    row_count, column_count = _facet_grid_shape(len(panels))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.0 * column_count, 3.0 * row_count),
        squeeze=False,
        sharex=True,
    )
    for ax, (title, data, value_column, ylabel) in zip(
        axes.ravel(),
        panels,
        strict=False,
    ):
        for label, group in _ordered_groups(data):
            sorted_group = group.sort_values("dimensions")
            label_text = "/".join(label)
            _plot_with_errorbars(
                ax,
                sorted_group["dimensions"],
                sorted_group[value_column],
                style={
                    "label": label_text,
                    "color": _series_color(label),
                    "linestyle": _series_linestyle(label),
                },
                yerr=_errorbar_interval(sorted_group, value_column),
            )
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Feature count")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes.ravel()[len(panels) :]:
        ax.set_axis_off()

    labels = list(
        _ordered_group_labels(
            pd.concat([data[["backend", "layout"]] for _, data, _, _ in panels])
        )
    )
    if labels:
        legend_columns = min(legend_columns, len(labels))
        legend_rows = -(-len(labels) // legend_columns)
        fig.legend(
            handles=[_series_legend_handle(label) for label in labels],
            loc="lower center",
            ncols=legend_columns,
            fontsize="small",
        )
        fig.tight_layout(rect=(0, 0.02 + 0.025 * legend_rows, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def profile_comparison_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare the compact profile with the default profile at the top feature count.

    Each ratio divides the compact result by the default result for the same layout.
    """
    if "profile" not in summary.columns or not (summary["profile"] == "compact").any():
        return pd.DataFrame()
    largest = summary["dimensions"].max()
    data = summary[
        (summary["dimensions"] == largest)
        & summary["operation"].isin(["write", "matrix_materialization"])
    ]
    rows = []
    compact_layouts = data[data["profile"] == "compact"][["backend", "layout"]]
    for (backend, layout), _ in _ordered_groups(compact_layouts.drop_duplicates()):
        layout_data = data[(data["backend"] == backend) & (data["layout"] == layout)]
        values = {}
        for profile in ["default", "compact"]:
            profile_data = layout_data[layout_data["profile"] == profile]
            write = profile_data[profile_data["operation"] == "write"].iloc[0]
            matrix = profile_data[profile_data["operation"] == "matrix_materialization"]
            values[profile] = {
                "bytes": float(write["artifact_bytes"]),
                "write": float(write["median_seconds"]),
                "matrix": float(matrix.iloc[0]["median_seconds"]),
                "values": float(write["rows"]) * float(write["dimensions"]),
            }
        default, compact = values["default"], values["compact"]
        rows.append(
            {
                "backend": backend,
                "layout": layout,
                "dimensions": int(largest),
                "default_bytes_per_value": default["bytes"] / default["values"],
                "compact_bytes_per_value": compact["bytes"] / compact["values"],
                "size_ratio": compact["bytes"] / default["bytes"],
                "matrix_materialization_ratio": compact["matrix"] / default["matrix"],
                "write_ratio": compact["write"] / default["write"],
            }
        )
    return pd.DataFrame(rows)


def write_profile_tables(
    summary: pd.DataFrame,
    output_dir: Path = Path("results"),
) -> list[Path]:
    """Write the compact-versus-default comparison, if a compact profile ran."""
    comparison = profile_comparison_table(summary)
    if comparison.empty:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "profile_comparison.parquet"
    comparison.to_parquet(path, index=False)
    return [path]


def write_profile_figure(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> Path | None:
    """Plot storage size against time for the default and compact profiles.

    Each arrow starts at the default result and ends at the compact result.
    Layouts without a compact profile appear as single points.
    """
    if profile_comparison_table(summary).empty:
        return None
    largest = summary["dimensions"].max()
    top = summary[summary["dimensions"] == largest]
    sizes = top[top["operation"] == "write"].set_index(["backend", "layout", "profile"])
    values = float(sizes["rows"].iloc[0]) * float(largest)
    panels = [
        ("matrix_materialization", "Matrix Materialization"),
        ("write", "Write"),
    ]
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(panels), figsize=(11, 4.8), squeeze=False)
    handles: dict[str, object] = {}
    for ax, (operation, title) in zip(axes.ravel(), panels, strict=True):
        timed = top[top["operation"] == operation].set_index(
            ["backend", "layout", "profile"]
        )
        for (backend, layout), _ in _ordered_groups(
            top[["backend", "layout"]].drop_duplicates()
        ):
            color = _series_color((backend, layout))
            points = {}
            for profile in ["default", "compact"]:
                key = (backend, layout, profile)
                if key in timed.index and key in sizes.index:
                    points[profile] = (
                        float(sizes.loc[key, "artifact_bytes"]) / (values * 4),
                        float(timed.loc[key, "median_seconds"]),
                    )
            for profile, point in points.items():
                marker = ax.scatter(
                    *point,
                    marker="s" if layout == "wide" else "o",
                    s=55,
                    color=color if profile == "default" else "white",
                    edgecolors=color,
                    linewidths=1.6,
                    zorder=3,
                )
                handles.setdefault(f"{backend}/{layout}", marker)
            if len(points) == 2:  # noqa: PLR2004
                ax.annotate(
                    "",
                    xy=points["compact"],
                    xytext=points["default"],
                    arrowprops={"arrowstyle": "->", "color": color, "lw": 1.4},
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(_direction_title(title, better="lower"))
        ax.set_xlabel("Storage size (x raw float32)")
        ax.set_ylabel("Median seconds")
        ax.grid(True, alpha=0.25)

    marker_key = [
        Line2D([], [], marker="o", linestyle="", color="black", label="default"),
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            markerfacecolor="white",
            color="black",
            label="compact",
        ),
    ]
    series = []
    for name in handles:
        backend, _, layout = name.partition("/")
        series.append(
            Line2D(
                [],
                [],
                marker="s" if layout == "wide" else "o",
                linestyle="",
                color=_series_color((backend, layout)),
                label=name,
            )
        )
    fig.legend(
        handles=[*series, *marker_key],
        loc="lower center",
        ncols=5,
        fontsize="small",
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    path = figure_dir / "encoding_profiles.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


SENSITIVITY_MEASURES = [
    ("storage_size", "write", "artifact_bytes"),
    ("matrix_materialization", "matrix_materialization", "median_seconds"),
    ("feature_projection", "feature_projection", "median_seconds"),
    ("write", "write", "median_seconds"),
]


def profile_sensitivity_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare array-to-wide ratios under the default and compact profiles.

    A backend appears when both its wide and array-like layouts ran in both
    profiles. The `holds` column is true when every measure stays on the same
    side of 1.0 in both profiles.
    """
    if "profile" not in summary.columns:
        return pd.DataFrame()
    top = summary[summary["dimensions"] == summary["dimensions"].max()]
    rows = []
    for backend in _ordered_backends(top["backend"].unique()):
        backend_rows = top[top["backend"] == backend]
        for layout in _ordered_layouts(backend_rows["layout"].unique()):
            if layout == "wide":
                continue
            row = _sensitivity_row(backend_rows, backend, layout)
            if row is not None:
                rows.append(row)
    return pd.DataFrame(rows)


def _sensitivity_row(
    backend_rows: pd.DataFrame,
    backend: str,
    layout: str,
) -> dict[str, Any] | None:
    """Return one sensitivity row, or None when a needed profile is missing."""
    pairs = {}
    for profile in ["default", "compact"]:
        in_profile = backend_rows[backend_rows["profile"] == profile]
        wide = in_profile[in_profile["layout"] == "wide"]
        array = in_profile[in_profile["layout"] == layout]
        if wide.empty or array.empty:
            return None
        pairs[profile] = (wide, array)

    row: dict[str, Any] = {
        "backend": backend,
        "layout": layout,
        "dimensions": int(backend_rows["dimensions"].max()),
    }
    holds = True
    for name, operation, column in SENSITIVITY_MEASURES:
        ratios = {}
        for profile, (wide, array) in pairs.items():
            wide_value = wide[wide["operation"] == operation][column]
            array_value = array[array["operation"] == operation][column]
            ratios[profile] = (
                float(array_value.iloc[0]) / float(wide_value.iloc[0])
                if not wide_value.empty and not array_value.empty
                else float("nan")
            )
            row[f"{name}_{profile}"] = ratios[profile]
        if not any(np.isnan(value) for value in ratios.values()):
            holds = holds and (ratios["default"] < 1.0) == (ratios["compact"] < 1.0)
    row["holds"] = holds
    return row


def sensitivity_table_markdown(table: pd.DataFrame) -> str:
    """Return a Markdown table of array-to-wide ratios in both profiles."""
    lines = [
        "| Backend | Layout | Storage size | Matrix materialization "
        "| Feature projection | Write | Conclusion |",
        "| ------- | ------ | ------------ | ---------------------- "
        "| ------------------ | ----- | ---------- |",
    ]
    for row in table.to_dict("records"):
        cells = []
        for name, _, _ in SENSITIVITY_MEASURES:
            default, compact = row[f"{name}_default"], row[f"{name}_compact"]
            cells.append(
                "n/a"
                if np.isnan(default) or np.isnan(compact)
                else f"{_change_text(default)} → {_change_text(compact)}"
            )
        backend = BACKEND_DISPLAY_NAMES.get(row["backend"], row["backend"])
        conclusion = "Same side of wide" if row["holds"] else "Reverses"
        lines.append(
            f"| {backend} | `{row['layout']}` | "
            + " | ".join(cells)
            + f" | {conclusion} |"
        )
    return "\n".join(lines)


def _change_text(ratio: float) -> str:
    """Return a signed percentage for small changes and a multiplier otherwise."""
    if GAIN_MULTIPLIER_LIMIT < ratio < LOSS_MULTIPLIER_LIMIT:
        percent = round((ratio - 1) * 100)
        return "0%" if percent == 0 else f"{percent:+d}%"
    if ratio <= GAIN_MULTIPLIER_LIMIT:
        return f"{_multiplier(1 / ratio)}x lower"
    return f"{_multiplier(ratio)}x higher"


def _multiplier(value: float) -> str:
    """Format a multiplier with two significant digits and no exponent.

    The third digit is not reliable, because repeated runs differ by more than 1%.
    """
    rounded = float(f"{value:.2g}")
    if rounded >= 100:  # noqa: PLR2004
        return f"{rounded:,.0f}"
    return f"{rounded:.2g}"


def profile_change_table(comparison: pd.DataFrame) -> str:
    """Return a Markdown table of compact-versus-default changes."""
    lines = [
        "| Layout | Storage size | Matrix materialization | Write |",
        "| ------ | ------------ | ---------------------- | ----- |",
    ]
    for row in comparison.to_dict("records"):
        name = BACKEND_DISPLAY_NAMES.get(row["backend"], row["backend"])
        lines.append(
            f"| {name} `{row['layout']}` | {_change_text(row['size_ratio'])} "
            f"| {_change_text(row['matrix_materialization_ratio'])} "
            f"| {_change_text(row['write_ratio'])} |"
        )
    return "\n".join(lines)


def encoding_table(encodings: pd.DataFrame, summary: pd.DataFrame) -> str:
    """Return a Markdown table of observed encodings and stored bytes per value."""
    largest = encodings["dimensions"].max()
    if "profile" not in summary.columns:
        summary = summary.assign(profile="default")
    writes = summary[
        (summary["operation"] == "write") & (summary["dimensions"] == largest)
    ].set_index(["backend", "layout", "profile"])
    latest = encodings[encodings["dimensions"] == largest]
    lines = [
        "| Backend | Layout | Profile | Compression and encodings | Bytes per value |",
        "| ------- | ------ | ------- | ------------------------- | --------------- |",
    ]
    for backend in _ordered_backends(latest["backend"].unique()):
        backend_rows = latest[latest["backend"] == backend]
        for layout in _ordered_layouts(backend_rows["layout"].unique()):
            layout_rows = backend_rows[backend_rows["layout"] == layout]
            for profile in ["default", "compact"]:
                match = layout_rows[layout_rows["profile"] == profile]
                key = (backend, layout, profile)
                if match.empty or key not in writes.index:
                    continue
                row = writes.loc[key]
                per_value = row["artifact_bytes"] / (row["rows"] * row["dimensions"])
                name = BACKEND_DISPLAY_NAMES.get(backend, backend)
                lines.append(
                    f"| {name} | `{layout}` | {profile} "
                    f"| {match.iloc[0]['encoding']} | {per_value:.2f} |"
                )
    return "\n".join(lines)


def write_absolute_figures(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> list[Path]:
    """Write side-by-side absolute comparison figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    for operation in OPERATIONS:
        data = summary[summary["operation"] == operation]
        if data.empty:
            continue
        path = figure_dir / f"{operation}_absolute_comparison.png"
        _write_absolute_comparison(
            data=data,
            value_column="median_seconds",
            ylabel="Median seconds",
            title=f"{operation.replace('_', ' ').title()} Time",
            path=path,
        )
        paths.append(path)

    storage = summary[summary["operation"] == "write"].copy()
    if not storage.empty:
        storage["artifact_megabytes"] = storage["artifact_bytes"] / 1_000_000
        path = figure_dir / "storage_size_absolute_comparison.png"
        _write_absolute_comparison(
            data=storage,
            value_column="artifact_megabytes",
            ylabel="Artifact size (MB)",
            title="Artifact Size",
            path=path,
        )
        paths.append(path)
    return paths


def write_ratio_figures(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> list[Path]:
    """Write compact ratio figures from summarized benchmark data."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    ratios = ratio_table(summary)
    paths: list[Path] = []
    for operation in OPERATIONS:
        data = ratios[ratios["operation"] == operation]
        if data.empty:
            continue
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, group in _ordered_groups(data):
            sorted_group = group.sort_values("dimensions")
            _plot_with_errorbars(
                ax,
                sorted_group["dimensions"],
                sorted_group["time_ratio"],
                style={"label": "/".join(label)},
                yerr=_errorbar_interval(sorted_group, "time_ratio"),
            )
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_title(f"{operation.replace('_', ' ').title()} Time Ratio")
        ax.set_xlabel("Feature count")
        ax.set_ylabel("array / wide")
        ax.legend(fontsize="small", ncols=2)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = figure_dir / f"{operation}_time_ratio.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)

    storage = ratios[ratios["operation"] == "write"]
    if not storage.empty:
        fig, ax = plt.subplots(figsize=(9, 4.8))
        for label, group in _ordered_groups(storage):
            sorted_group = group.sort_values("dimensions")
            _plot_with_errorbars(
                ax,
                sorted_group["dimensions"],
                sorted_group["artifact_size_ratio"],
                style={"label": "/".join(label)},
                yerr=_errorbar_interval(sorted_group, "artifact_size_ratio"),
            )
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_title("Artifact Size Ratio")
        ax.set_xlabel("Feature count")
        ax.set_ylabel("array / wide")
        ax.legend(fontsize="small", ncols=2)
        ax.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        path = figure_dir / "storage_size_ratio.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(path)
    return paths


def _write_absolute_comparison(
    *,
    data: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """Write one side-by-side figure with one backend per panel."""
    backends = _ordered_backends(data["backend"].unique())
    panel_count = len(backends)
    fig, axes = plt.subplots(
        1,
        panel_count,
        figsize=(max(6.0, 3.3 * panel_count), 4.8),
        squeeze=False,
        sharey=False,
    )
    legend_labels: list[str] = []
    for index, backend in enumerate(backends):
        ax = axes[0][index]
        backend_data = data[data["backend"] == backend]
        dimensions = sorted(backend_data["dimensions"].unique())
        layouts = _ordered_layouts(backend_data["layout"].unique())
        x_positions = list(range(len(dimensions)))
        bar_width = min(0.8 / max(len(layouts), 1), 0.28)
        center_offset = (len(layouts) - 1) * bar_width / 2

        for layout_index, layout in enumerate(layouts):
            layout_data = backend_data[backend_data["layout"] == layout]
            values = [
                _value_for_dimension(
                    data=layout_data,
                    dimension=dimension,
                    value_column=value_column,
                )
                for dimension in dimensions
            ]
            offsets = [
                position - center_offset + layout_index * bar_width
                for position in x_positions
            ]
            bars = ax.bar(
                offsets,
                values,
                width=bar_width,
                label=layout,
                color=LAYOUT_COLORS.get(layout),
                yerr=_errorbar_for_dimensions(
                    data=layout_data,
                    dimensions=dimensions,
                    value_column=value_column,
                ),
                capsize=3,
            )
            if bars and layout not in legend_labels:
                legend_labels.append(layout)

        ax.set_title(str(backend))
        ax.set_xlabel("Feature count")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([str(dimension) for dimension in dimensions])
        ax.grid(True, axis="y", alpha=0.25)
        if index == 0:
            ax.set_ylabel(ylabel)

    fig.suptitle(f"{title}: Wide and Array-Like Layouts")
    if legend_labels:
        legend_handles = [
            Patch(
                facecolor=LAYOUT_COLORS.get(label, "#7f7f7f"),
                label=label,
            )
            for label in legend_labels
        ]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncols=min(len(legend_labels), 4),
        )
    fig.tight_layout(rect=(0, 0.12, 1, 0.94))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _ordered_layouts(layouts: Iterable[object]) -> list[str]:
    """Return wide first, then other layouts alphabetically."""
    names = sorted(str(layout) for layout in layouts)
    if "wide" in names:
        names.remove("wide")
        return ["wide", *names]
    return names


def _ordered_backends(backends: Iterable[object]) -> list[str]:
    """Return known backends in plot order, then unknown backends."""
    names = [str(backend) for backend in backends]
    known = [backend for backend in BACKEND_ORDER if backend in names]
    unknown = sorted(backend for backend in names if backend not in BACKEND_ORDER)
    return [*known, *unknown]


def access_path_table(summary: pd.DataFrame) -> str:
    """Return a Markdown table for package and binding access paths."""
    lines = [
        "| Backend | Package or binding | Layouts measured |",
        "| ------- | ------------------ | ---------------- |",
    ]
    for backend in _ordered_backends(summary["backend"].unique()):
        backend_rows = summary[summary["backend"] == backend]
        layouts = ", ".join(
            f"`{layout}`"
            for layout in _ordered_layouts(backend_rows["layout"].unique())
        )
        package = BACKEND_PACKAGES.get(backend, "Unknown")
        display_backend = BACKEND_DISPLAY_NAMES.get(backend, backend)
        lines.append(f"| {display_backend} | {package} | {layouts} |")
    return "\n".join(lines)


def _with_reference_series(
    ratios: pd.DataFrame,
    *,
    backend: str,
    layout: str,
) -> pd.DataFrame:
    """Add the reference layout as a flat series at 1.0 so it appears in the plots."""
    already_present = (ratios["backend"] == backend) & (ratios["layout"] == layout)
    if already_present.any():
        return ratios
    reference = ratios[KEY_COLUMNS].drop_duplicates().copy()
    reference["backend"] = backend
    reference["layout"] = layout
    for column in [
        "time_ratio",
        "time_ratio_q25",
        "time_ratio_q75",
        "artifact_size_ratio",
    ]:
        reference[column] = 1.0
    return pd.concat([ratios, reference], ignore_index=True)


def _facet_grid_shape(panel_count: int) -> tuple[int, int]:
    """Return row and column counts with at most three columns."""
    column_count = min(3, max(panel_count, 1))
    row_count = (panel_count + column_count - 1) // column_count
    return row_count, column_count


def _geometric_mean(values: pd.Series) -> float:
    """Return the geometric mean, the standard average for ratios."""
    return float(np.exp(np.log(values).mean()))


def _time_summary_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    """Return the geometric mean time ratio across timed operations."""
    timed = ratios[ratios["operation"].isin(OPERATIONS)]
    if timed.empty:
        return pd.DataFrame()
    return (
        timed.groupby(["backend", "layout", "dimensions"], as_index=False)
        .agg(
            geometric_mean_time_ratio=("time_ratio", _geometric_mean),
        )
        .sort_values(["backend", "layout", "dimensions"])
    )


def _direction_title(title: str, *, better: str) -> str:
    """Return a plot title with a direction cue."""
    return f"{title}\n({better} is better)"


def _ratio_axis_label(value_column: str, *, baseline: str = "array / wide") -> str:
    """Return the y-axis label for a ratio plot."""
    if value_column == "artifact_size_ratio":
        return f"Size ratio ({baseline})"
    return f"Time ratio ({baseline})"


def _ordered_groups(
    data: pd.DataFrame,
) -> Iterator[tuple[tuple[str, str], pd.DataFrame]]:
    """Yield backend and layout groups in plot order."""
    for backend in _ordered_backends(data["backend"].unique()):
        backend_data = data[data["backend"] == backend]
        for layout in _ordered_layouts(backend_data["layout"].unique()):
            layout_data = backend_data[backend_data["layout"] == layout]
            if not layout_data.empty:
                yield (backend, layout), layout_data


def _ordered_group_labels(data: pd.DataFrame) -> Iterator[tuple[str, str]]:
    """Yield backend and layout labels in plot order."""
    for label, _group in _ordered_groups(data):
        yield label


def _series_color(label: tuple[str, str]) -> str:
    """Return the color for a combined-plot series."""
    return SERIES_COLORS.get(label, "#666666")


def _series_legend_handle(label: tuple[str, str]) -> Line2D:
    """Return a legend key that shows only the line color and the line style."""
    return Line2D(
        [],
        [],
        color=_series_color(label),
        linestyle=_series_linestyle(label),
        linewidth=1.7,
        label="/".join(label),
    )


def _series_linestyle(label: tuple[str, str]) -> Literal["--", "-"]:
    """Return a dashed line for wide layouts and a solid line otherwise."""
    return "--" if label[1] == "wide" else "-"


def _plot_with_errorbars(
    ax: plt.Axes,
    x_values: pd.Series,
    y_values: pd.Series,
    *,
    style: PlotStyle,
    yerr: list[list[float]] | None,
) -> object:
    """Plot one line and include error bars when repeat intervals exist."""
    label = style.get("label")
    color = style.get("color")
    linestyle = style.get("linestyle", "-")
    if yerr is None:
        return ax.plot(
            x_values,
            y_values,
            marker="o",
            markersize=4.5,
            linewidth=1.7,
            linestyle=linestyle,
            label=label,
            color=color,
        )[0]
    container = ax.errorbar(
        x_values,
        y_values,
        yerr=yerr,
        fmt=f"{linestyle}o",
        markersize=4.5,
        linewidth=1.7,
        elinewidth=1.5,
        capsize=4,
        capthick=1.5,
        label=label,
        color=color,
        markeredgecolor="white",
        markeredgewidth=0.6,
    )
    return container


def _errorbar_interval(
    data: pd.DataFrame,
    value_column: str,
) -> list[list[float]] | None:
    """Return asymmetric error bars from repeat quantiles."""
    columns = _errorbar_columns(value_column)
    if columns is None:
        return None
    lower_column, upper_column = columns
    required = {value_column, lower_column, upper_column}
    if not required.issubset(data.columns):
        return None

    lower_errors: list[float] = []
    upper_errors: list[float] = []
    for _, row in data.iterrows():
        value = float(row[value_column])
        lower_errors.append(max(value - float(row[lower_column]), 0.0))
        upper_errors.append(max(float(row[upper_column]) - value, 0.0))
    return [lower_errors, upper_errors]


def _errorbar_for_dimensions(
    *,
    data: pd.DataFrame,
    dimensions: list[int],
    value_column: str,
) -> list[list[float]] | None:
    """Return bar error intervals in the same order as plotted dimensions."""
    columns = _errorbar_columns(value_column)
    if columns is None:
        return None
    lower_errors: list[float] = []
    upper_errors: list[float] = []
    found_interval = False
    for dimension in dimensions:
        dimension_data = data[data["dimensions"] == dimension]
        interval = _errorbar_interval(dimension_data, value_column)
        if interval is None or dimension_data.empty:
            lower_errors.append(0.0)
            upper_errors.append(0.0)
            continue
        lower_errors.append(interval[0][0])
        upper_errors.append(interval[1][0])
        found_interval = True
    if not found_interval:
        return None
    return [lower_errors, upper_errors]


def _errorbar_columns(value_column: str) -> tuple[str, str] | None:
    """Return lower and upper interval columns for a plotted value."""
    if value_column == "median_seconds":
        return "q25_seconds", "q75_seconds"
    if value_column == "time_ratio":
        return "time_ratio_q25", "time_ratio_q75"
    return None


def _value_for_dimension(
    *,
    data: pd.DataFrame,
    dimension: int,
    value_column: str,
) -> float:
    """Return one plotted value for a dimension."""
    values = data[data["dimensions"] == dimension][value_column]
    if values.empty:
        return 0.0
    return float(values.iloc[0])


def update_readme(  # noqa: PLR0913
    *,
    readme_path: Path,
    summary: pd.DataFrame,
    figure_paths: list[Path],
    environment: dict[str, Any] | None = None,
    encodings: pd.DataFrame | None = None,
    floor: pd.DataFrame | None = None,
) -> None:
    """Insert the latest benchmark results and setup into the README.

    The results block is appended when its markers are missing. The setup block is
    replaced only when its markers exist.
    """
    text = readme_path.read_text(encoding="utf-8")
    results = render_results_section(
        summary=summary,
        figure_paths=figure_paths,
        encodings=encodings,
        floor=floor,
    )
    text = _replace_block(text, RESULTS_START, RESULTS_END, results, append=True)
    setup = "\n".join(render_setup_section(summary, environment))
    text = _replace_block(text, SETUP_START, SETUP_END, setup, append=False)
    readme_path.write_text(text, encoding="utf-8")


def _replace_block(
    text: str,
    start: str,
    end: str,
    content: str,
    *,
    append: bool,
) -> str:
    """Replace the text between two markers, or append a block when asked."""
    block = f"{start}\n\n{content}\n\n{end}"
    if start in text and end in text:
        before = text.split(start, maxsplit=1)[0]
        after = text.split(end, maxsplit=1)[1]
        return before + block + after
    return text.rstrip() + "\n\n" + block + "\n" if append else text


def render_results_section(
    *,
    summary: pd.DataFrame,
    figure_paths: list[Path],
    encodings: pd.DataFrame | None = None,
    floor: pd.DataFrame | None = None,
) -> str:
    """Render the Markdown summary and results for the README."""
    all_profiles = summary
    summary = default_profile(summary)
    max_dimension = int(summary["dimensions"].max())
    lines = [
        *summary_section(all_profiles),
        *_blank_before(real_world_section(all_profiles)),
        "",
        "## Results",
        "",
        _run_description(summary),
        "",
        "### Key findings",
        "",
        (
            "Each cell compares the array-like layout with the wide layout of the "
            f"same backend at {max_dimension:,} features. Negative percentages and "
            '"lower" mean faster or smaller. '
            f"{_feature_projection_note(summary, max_dimension)}"
        ),
        "",
        array_vs_wide_markdown(array_vs_wide_table(summary)),
        *_optional_paragraph(_noise_note(summary)),
        *_optional_paragraph(_parallelism_note(summary)),
        *_optional_paragraph(_cpu_time_note(summary)),
        *_optional_paragraph(_access_path_note(summary)),
        *_blank_before(floor_section(all_profiles, floor)),
        "",
        "### How to read the figures",
        "",
        "- Each panel shows one operation or one summary measure.",
        "- Lower is better in every panel.",
        (
            "- Ratio panels divide one result by a reference result. "
            "A value of 1.0 means the same as the reference."
        ),
        (
            "- Geometric Mean Time is the geometric mean of the time ratios "
            "across all operations. The geometric mean is the standard way "
            "to average ratios."
        ),
        "- Error bars show the q25-to-q75 range across repeated runs.",
        "- The y-axis uses a log scale to show small and large changes.",
        "- Dashed lines are wide layouts. Solid lines are array-like layouts.",
    ]

    figures = _named_figures(figure_paths)
    figure_number = 0
    for filename, heading, alt_text, caption in [
        (
            "combined_facet_overview.png",
            "Every layout against CSV wide",
            "Time and storage ratios for every layout against CSV wide",
            (
                "Every layout divided by CSV wide, which is the flat line at 1.0. "
                "Parquet wide is the only other wide layout shown here. "
                "Figure 2 shows all wide layouts."
            ),
        ),
        (
            "wide_layouts_facet_overview.png",
            "Wide layouts",
            "Time and storage ratios for wide layouts against CSV wide",
            ("Wide layouts only, divided by CSV wide, which is the flat line at 1.0."),
        ),
        (
            "backend_wide_facet_overview.png",
            "Array-like layouts against their own wide layout",
            "Time and storage ratios for array-like layouts against wide layouts",
            (
                "Each array-like layout divided by the wide layout of the same "
                "backend. Values below 1.0 favor the array-like layout."
            ),
        ),
    ]:
        path = figures.get(filename)
        if path is None:
            continue
        figure_number += 1
        lines.extend(
            [
                "",
                f"### {heading}",
                "",
                f"![{alt_text}]({path.as_posix()})",
                "",
                f"Figure {figure_number}. {caption}",
            ]
        )
        if filename == "combined_facet_overview.png":
            lines.extend(["", REFERENCE_NOTE])

    lines.extend(_encoding_lines(all_profiles, encodings))
    lines.extend(
        _appendix_lines(
            all_profiles,
            figures.get("encoding_profiles.png"),
            figure_number + 1,
        )
    )
    return "\n".join(lines)


REFERENCE_NOTE = (
    "CSV wide is the reference because it is the most common way to share this "
    "kind of data. It is a text format, so ratios against it exaggerate the gain "
    "of any binary format. Figure 3 is the fairer comparison: each array-like "
    "layout against a wide layout in the same format."
)


def render_setup_section(
    summary: pd.DataFrame,
    environment: dict[str, Any] | None,
) -> list[str]:
    """Return the environment and backend sections, which describe the setup."""
    lines: list[str] = []
    if environment:
        lines.extend(["## Environment", "", environment_table(environment), ""])
    lines.extend(
        [
            "## Backends and access paths",
            "",
            access_path_table(default_profile(summary)),
            "",
            (
                "The timings include each package and binding, not only the "
                "storage layout. Different language layers can change the result."
            ),
        ]
    )
    return lines


def _join_words(items: list[str]) -> str:
    """Join words with commas and a final `and`."""
    if len(items) <= 2:  # noqa: PLR2004
        return " and ".join(items)
    return ", ".join(items[:-1]) + ", and " + items[-1]


def _size_words(ratio: float) -> str:
    """Describe a size ratio as a percentage or a multiplier."""
    if ratio < 1:
        return f"{round((1 - ratio) * 100)}% smaller"
    if ratio < LOSS_MULTIPLIER_LIMIT:
        return f"{round((ratio - 1) * 100)}% larger"
    return f"{_multiplier(ratio)}x larger"


def array_vs_wide_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare each array-like layout with the wide layout of its own backend.

    The table uses the largest feature count. Each value is an array-like result
    divided by the wide result. Missing operations give missing values.
    """
    ratios = ratio_table(summary)
    if ratios.empty:
        return pd.DataFrame()
    top = ratios[ratios["dimensions"] == ratios["dimensions"].max()]
    rows = []
    for (backend, layout), group in _ordered_groups(top):

        def ratio(operation: str, column: str, rows: pd.DataFrame = group) -> float:
            values = rows[rows["operation"] == operation][column]
            return float(values.iloc[0]) if not values.empty else float("nan")

        def noisy(operation: str, rows: pd.DataFrame = group) -> bool:
            match = rows[rows["operation"] == operation]
            if match.empty:
                return False
            row = match.iloc[0]
            return _varies(
                row["median_seconds"], row["q25_seconds"], row["q75_seconds"]
            ) or (
                _varies(
                    row["wide_median_seconds"],
                    row["wide_q25_seconds"],
                    row["wide_q75_seconds"],
                )
            )

        rows.append(
            {
                "backend": backend,
                "layout": layout,
                "dimensions": int(top["dimensions"].max()),
                "matrix_materialization": ratio("matrix_materialization", "time_ratio"),
                "feature_projection": ratio("feature_projection", "time_ratio"),
                "write": ratio("write", "time_ratio"),
                "storage_size": ratio("write", "artifact_size_ratio"),
                "matrix_materialization_noisy": noisy("matrix_materialization"),
                "feature_projection_noisy": noisy("feature_projection"),
                "write_noisy": noisy("write"),
                "storage_size_noisy": False,
            }
        )
    return pd.DataFrame(rows)


def array_vs_wide_markdown(table: pd.DataFrame) -> str:
    """Return the array-versus-wide table as Markdown."""
    lines = [
        "| Backend | Layout | Matrix materialization | Feature projection "
        "| Write | Storage size |",
        "| ------- | ------ | ---------------------- | ------------------ "
        "| ----- | ------------ |",
    ]
    for row in table.to_dict("records"):
        cells = [
            "n/a"
            if np.isnan(row[name])
            else _change_text(row[name]) + ("*" if row.get(f"{name}_noisy") else "")
            for name in [
                "matrix_materialization",
                "feature_projection",
                "write",
                "storage_size",
            ]
        ]
        backend = BACKEND_DISPLAY_NAMES.get(row["backend"], row["backend"])
        lines.append(f"| {backend} | `{row['layout']}` | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _backend_ratios(table: pd.DataFrame, column: str) -> dict[str, float]:
    """Return one geometric-mean ratio per backend, skipping missing values."""
    ratios: dict[str, float] = {}
    for backend in table["backend"].unique():
        values = table[table["backend"] == backend][column].dropna()
        if not values.empty:
            ratios[backend] = _geometric_mean(values)
    return ratios


def _named(backend: str) -> str:
    return BACKEND_DISPLAY_NAMES.get(backend, backend)


def _feature_projection_note(summary: pd.DataFrame, dimensions: int) -> str:
    """Say that feature projection is the best case for a wide layout."""
    count = _feature_count_text(summary).split()[0]
    if not count.isdigit():
        return ""
    return (
        f"Feature projection reads {count} of {dimensions:,} features, "
        "which is the best case for a wide layout."
    )


def _feature_count_text(summary: pd.DataFrame) -> str:
    """Return the number of projected features, or a vague phrase."""
    values = summary[summary["operation"] == "feature_projection"][
        "operation_parameter"
    ]
    digits = [value for value in values if str(value).isdigit()]
    return f"{digits[0]} features" if digits else "a few features"


def summary_section(summary: pd.DataFrame) -> list[str]:
    """Return the summary: one main finding as a quote, then concrete bullets."""
    all_profiles = summary
    summary = default_profile(summary)
    table = array_vs_wide_table(summary)
    lines = ["## Summary", ""]
    if table.empty:
        return lines

    matrix = _backend_ratios(table, "matrix_materialization")
    projection = _backend_ratios(table, "feature_projection")
    features = _feature_count_text(summary)
    faster = sorted((r, b) for b, r in matrix.items() if r < 1)
    slower = sorted(((r, b) for b, r in matrix.items() if r > 1), reverse=True)
    groups = _projection_groups(projection)

    quote = _main_finding(faster, len(matrix), groups, features)
    if quote:
        lines.extend([f"> **Main finding.** {quote}", ""])
    if faster:
        lines.append(_matrix_bullet(faster, slower))
    if groups:
        lines.append(_projection_bullet(groups, features))
    lines.extend(
        bullet
        for bullet in [
            _text_packing_bullet(summary),
            _encoding_bullet(all_profiles),
        ]
        if bullet
    )
    repetitions = int(summary["repetitions"].max())
    lines.append(
        "- **Limits.** These results come from a single machine, synthetic data, "
        f"warm caches, Python bindings, and {repetitions} repetitions per timing. "
        "See Limitations."
    )
    return lines


def _projection_groups(projection: dict[str, float]) -> dict[str, list[str]]:
    """Group backends by how feature projection compares with their wide layout."""
    faster = sorted((r, b) for b, r in projection.items() if r < SIMILAR_LOW)
    same = [b for b, r in projection.items() if SIMILAR_LOW <= r <= SIMILAR_HIGH]
    slower = sorted(
        ((r, b) for b, r in projection.items() if r > SIMILAR_HIGH), reverse=True
    )
    groups = {
        "faster": [f"{_named(b)} ({_multiplier(1 / r)}x)" for r, b in faster],
        "about the same": [_named(b) for b in same],
        "slower": [f"{_named(b)} ({_multiplier(r)}x)" for r, b in slower],
    }
    return {name: items for name, items in groups.items() if items}


def _main_finding(
    faster: list[tuple[float, str]],
    backend_count: int,
    groups: dict[str, list[str]],
    features: str,
) -> str:
    """Return the one-paragraph main finding, or an empty string."""
    if not faster:
        return ""
    speedups = [1 / ratio for ratio, _ in faster]
    low, high = _multiplier(min(speedups)), _multiplier(max(speedups))
    span = f"{low}x" if low == high else f"{low}x to {high}x"
    text = (
        f"Array-like layouts read whole feature matrices {span} faster than wide "
        f"layouts in {len(faster)} of {backend_count} backends."
    )
    if not groups:
        return text
    parts = [f"{name} in {_join_words(items)}" for name, items in groups.items()]
    joined = _join_words(parts)
    if len(parts) > 1:
        return (
            f"{text} Reading only {features} gives mixed results: array-like "
            f"layouts are {joined}."
        )
    return f"{text} Reading only {features}, array-like layouts are {joined}."


def _matrix_bullet(
    faster: list[tuple[float, str]],
    slower: list[tuple[float, str]],
) -> str:
    """Return the bullet about whole-matrix reads."""
    fast_names = _join_words(
        [f"{_named(b)} ({_multiplier(1 / r)}x)" for r, b in faster]
    )
    text = (
        "- **Whole-matrix reads.** Matrix materialization is faster with the "
        f"array-like layout in {fast_names}."
    )
    if slower:
        slow_names = _join_words(
            [f"{_named(b)} ({_multiplier(r)}x)" for r, b in slower]
        )
        text += f" It is slower in {slow_names}."
    return text


def _projection_bullet(groups: dict[str, list[str]], features: str) -> str:
    """Return the bullet about selecting a few features."""
    sentences = [
        f"It is {name} in {_join_words(items)}." for name, items in groups.items()
    ]
    first = sentences[0].replace("It is", f"Reading {features} is", 1)
    return "- **Selecting a few features.** " + " ".join([first, *sentences[1:]])


def _text_packing_bullet(summary: pd.DataFrame) -> str:
    """Return the bullet about CSV packed arrays, or an empty string."""
    ratios = ratio_table(summary)
    csv = ratios[(ratios["backend"] == "csv")]
    if csv.empty:
        return ""
    csv = csv[csv["dimensions"] == csv["dimensions"].max()]
    per_operation = {
        operation: _geometric_mean(group["time_ratio"])
        for operation, group in csv.groupby("operation")
    }
    slower = [_plain(op) for op in OPERATIONS if per_operation.get(op, 1) > 1]
    faster = [_plain(op) for op in OPERATIONS if per_operation.get(op, 1) < 1]
    if not slower:
        return ""
    text = (
        "- **Text packing.** CSV packed arrays are slower than CSV wide for "
        f"{_join_words(slower)}"
    )
    text += f", and faster only for {_join_words(faster)}." if faster else "."
    sizes = csv[csv["operation"] == "write"]["artifact_size_ratio"]
    if not sizes.empty:
        text += f" They are {_size_words(_geometric_mean(sizes))}."
    return text


def _plain(operation: str) -> str:
    return operation.replace("_", " ")


def _encoding_bullet(summary: pd.DataFrame) -> str:
    """Return the bullet about encoding settings, or an empty string."""
    sensitivity = profile_sensitivity_table(summary)
    if sensitivity.empty:
        return ""
    parquet = sensitivity[sensitivity["backend"] == "parquet"]
    if parquet.empty:
        return ""
    row = parquet.iloc[0]
    default, compact = row["storage_size_default"], row["storage_size_compact"]
    if np.isnan(default) or np.isnan(compact):
        return ""
    text = (
        "- **Encoding settings matter.** Parquet's array layout is "
        f"{_size_words(default)} than its wide layout by default and "
        f"{_size_words(compact)} with the compact profile."
    )
    if compact > default:
        text += " Default dictionary encoding inflates the wide layout."
    return text


def encoding_findings(summary: pd.DataFrame) -> str | None:
    """Return text on Parquet dictionary encoding and DuckDB wide size, if known."""
    if "profile" not in summary.columns:
        return None
    largest = summary["dimensions"].max()
    writes = summary[
        (summary["operation"] == "write") & (summary["dimensions"] == largest)
    ]

    def per_value(backend: str, layout: str, profile: str) -> float | None:
        match = writes[
            (writes["backend"] == backend)
            & (writes["layout"] == layout)
            & (writes["profile"] == profile)
        ]
        if match.empty:
            return None
        row = match.iloc[0]
        return float(row["artifact_bytes"]) / (float(row["rows"]) * float(largest))

    default = per_value("parquet", "wide", "default")
    compact = per_value("parquet", "wide", "compact")
    if default is None or compact is None or default <= compact:
        return None
    text = (
        "Parquet turns on dictionary encoding by default. Every value in this data "
        "is unique, so the dictionary adds overhead. "
        f"Parquet wide takes {default:.2f} bytes per value by default and "
        f"{compact:.2f} with the compact profile."
    )
    duckdb = per_value("duckdb", "wide", "default")
    if duckdb is not None:
        first = writes.iloc[0]
        text += (
            f" DuckDB wide takes {duckdb:.2f} bytes per value, about "
            f"{duckdb / RAW_FLOAT32_BYTES:.1f}x the raw size. The benchmark did not "
            f"investigate why. It stores {int(largest):,} columns of "
            f"{int(first['rows']):,} values each."
        )
    return text


def _encoding_lines(
    summary: pd.DataFrame,
    encodings: pd.DataFrame | None,
) -> list[str]:
    """Return the sensitivity section and the observed-encodings section."""
    lines: list[str] = []
    sensitivity = profile_sensitivity_table(summary)
    if not sensitivity.empty:
        lines.extend(
            [
                "",
                "### Encoding sensitivity",
                "",
                (
                    "Encoding and compression choices change file size and speed. "
                    "This table checks whether the array-versus-wide results hold "
                    "under both write profiles. Each cell shows the array-like "
                    "layout relative to the wide layout of the same backend, first "
                    "with the default profile and then with the compact profile. "
                    "The last column says whether every measure stays on the same "
                    "side of wide, better or worse, in both profiles. This does "
                    "not mean that the size of the effect stays the same."
                ),
                "",
                sensitivity_table_markdown(sensitivity),
            ]
        )
    if encodings is not None:
        lines.extend(
            [
                "",
                "### Encodings observed",
                "",
                (
                    "The table lists the compression and encodings that each "
                    "format wrote at the largest feature count. Bytes per value "
                    "is the artifact size divided by the number of stored values. "
                    "A raw `float32` value takes 4 bytes. Most size differences "
                    "between the binary formats come from default compression "
                    "and encoding choices, not from the layout. Random values "
                    "compress poorly, so the sizes describe this data set and "
                    "not real profile data."
                ),
                *_optional_paragraph(encoding_findings(summary)),
                "",
                encoding_table(encodings, summary),
            ]
        )
    return lines


def _appendix_lines(
    summary: pd.DataFrame,
    path: Path | None,
    figure_number: int,
) -> list[str]:
    """Return the appendix that plots storage size against time."""
    comparison = profile_comparison_table(summary)
    if path is None or comparison.empty:
        return []
    return [
        "",
        "### Appendix: storage size against time",
        "",
        (
            "This appendix uses synthetic random data. It shows how the compact "
            "profile moves the size and the time of each format. It is not a "
            "ranking of formats for real data."
        ),
        "",
        f"![Storage size against time for the default and compact profiles]"
        f"({path.as_posix()})",
        "",
        (
            f"Figure {figure_number}. Storage size against time at the largest "
            "feature count. Each arrow goes from the default profile (filled) to "
            "the compact profile (open). Points without an arrow have no compact "
            "profile. Squares are wide layouts and circles are array-like "
            "layouts. Lower and further left is better."
        ),
        "",
        "Changes from the default profile to the compact profile:",
        "",
        profile_change_table(comparison),
        "",
        (
            "The compact profile changes only the write settings in Write "
            "settings. DuckDB, Lance, and Vortex have no compact profile."
        ),
    ]


def environment_table(environment: dict[str, Any]) -> str:
    """Return a Markdown table of hardware, versions, and thread limits."""
    hardware = environment.get("hardware", {})
    memory_gib = round(int(hardware.get("memory_bytes", 0)) / 2**30)
    rows = [
        (
            "CPU",
            f"{hardware.get('cpu_model', 'unknown')} "
            f"({hardware.get('logical_cores', '?')} logical cores)",
        ),
        ("Memory", f"{memory_gib} GiB"),
        ("Platform", environment.get("platform", "unknown")),
        ("Python", environment.get("python", "unknown")),
    ]
    if environment.get("git_commit"):
        rows.append(("Code version", f"`{environment['git_commit']}`"))
    if environment.get("runs", 1) > 1:
        rows.append(
            ("Runs", f"{environment['runs']} independent runs of the same code, pooled")
        )
    for package, version in environment.get("packages", {}).items():
        text = ".".join(map(str, version)) if isinstance(version, list) else version
        rows.append((f"`{package}`", text))
    threads = _thread_text(environment.get("thread_limits", {}))
    if threads:
        rows.append(("Threads", threads))
    lines = ["| Item | Value |", "| ---- | ----- |"]
    lines.extend(f"| {item} | {value} |" for item, value in rows)
    return "\n".join(lines)


def _thread_text(limits: dict[str, Any]) -> str:
    """Describe the thread limits in one line."""
    limited = [
        f"{THREAD_LABELS.get(name, name)} {value}"
        for name, value in limits.items()
        if value != "library default"
    ]
    defaults = [
        THREAD_LABELS.get(name, name)
        for name, value in limits.items()
        if value == "library default"
    ]
    parts = [", ".join(limited)] if limited else []
    if defaults:
        verb = (
            "use their library defaults"
            if len(defaults) > 1
            else "uses its library default"
        )
        parts.append(f"{_join_words(defaults)} {verb}")
    return "; ".join(parts)


def _named_figures(figure_paths: list[Path]) -> dict[str, Path]:
    """Return figure paths keyed by file name."""
    return {path.name: path for path in figure_paths}


def _optional_paragraph(text: str | None) -> list[str]:
    """Return a blank line and a paragraph, or nothing when there is no text."""
    return ["", text] if text else []


def _parallelism_note(summary: pd.DataFrame) -> str | None:
    """Name the layouts that used more than one thread in some operation."""
    if "median_parallelism" not in summary.columns:
        return None
    stats = summary.groupby(["backend", "layout"])["median_parallelism"].agg(
        ["median", "max"]
    )
    busy = stats[stats["max"] > PARALLELISM_LIMIT]
    if busy.empty:
        return None
    names = ", ".join(
        f"{_layout_name(backend, layout)} "
        f"(median {row['median']:.2g}, up to {row['max']:.2g})"
        for (backend, layout), row in busy.iterrows()
    )
    return (
        f"These layouts used more than one thread in at least one operation: {names}. "
        "Parallelism is CPU time divided by wall time, so 1.0 means one busy "
        "thread. The benchmark cannot limit the native thread pools of Lance and "
        "Vortex."
    )


def _short_time(seconds: float) -> str:
    """Format seconds, using milliseconds below one second."""
    if seconds < 1:
        return f"{_two_digits(seconds * 1000)} ms"
    return _duration(seconds)


def floor_table(summary: pd.DataFrame, floor: pd.DataFrame | None) -> pd.DataFrame:
    """Divide each layout's time by the time of a plain NumPy file."""
    if floor is None or floor.empty:
        return pd.DataFrame()
    summary = default_profile(summary)
    largest = summary["dimensions"].max()
    floor_top = floor[floor["dimensions"] == largest].set_index("operation")
    top = summary[summary["dimensions"] == largest]
    operations = ["matrix_materialization", "random_rows", "feature_projection"]
    if not set(operations).issubset(floor_top.index):
        return pd.DataFrame()
    rows = []
    for (backend, layout), group in _ordered_groups(top):
        row: dict[str, Any] = {"backend": backend, "layout": layout}
        for operation in operations:
            seconds = group[group["operation"] == operation]["median_seconds"]
            row[operation] = (
                float(seconds.iloc[0])
                / float(floor_top.loc[operation, "median_seconds"])
                if not seconds.empty
                else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def floor_section(summary: pd.DataFrame, floor: pd.DataFrame | None) -> list[str]:
    """Return the section that compares every layout with a plain NumPy file."""
    table = floor_table(summary, floor)
    if table.empty or floor is None:
        return []
    largest = default_profile(summary)["dimensions"].max()
    matrix = floor[
        (floor["dimensions"] == largest)
        & (floor["operation"] == "matrix_materialization")
    ]["median_seconds"].iloc[0]
    lines = [
        "| Layout | Matrix materialization | Random rows | Feature projection |",
        "| ------ | ---------------------- | ----------- | ------------------ |",
    ]
    for row in table.to_dict("records"):
        cells = [
            "n/a" if np.isnan(row[name]) else f"{_multiplier(row[name])}x"
            for name in ["matrix_materialization", "random_rows", "feature_projection"]
        ]
        lines.append(
            f"| {_layout_name(row['backend'], row['layout'])} | "
            + " | ".join(cells)
            + " |"
        )
    return [
        "### Distance from a plain NumPy file",
        "",
        (
            f"A NumPy `.npy` file loads into memory in {_short_time(float(matrix))} "
            f"at {int(largest):,} features. This is the floor, because no format "
            "can be faster than a memory copy. Each cell divides the time of a "
            "layout by the time of the floor, so 1x is as fast as the floor."
        ),
        "",
        *lines,
    ]


def _varies(median: float, q25: float, q75: float) -> bool:
    """Return whether the q25-to-q75 range is more than half of the median."""
    return bool((q75 - q25) > NOISY_RANGE * median)


def _noise_note(summary: pd.DataFrame) -> str | None:
    """Explain the asterisk and count the measurements that vary a lot."""
    if "noisy" not in summary.columns:
        return None
    default = default_profile(summary)
    count = int(default["noisy"].sum())
    if count == 0:
        return None
    share = round(100 * count / len(default))
    return (
        "* marks a cell whose measurement varies by more than half of its median. "
        f"{count} of {len(default)} measurements ({share}%) vary this much."
    )


def _access_path_note(summary: pd.DataFrame) -> str | None:
    """Check that no layout reads far more than its operation needs.

    Matrix materialization should cost about as much as a full read, and random
    rows should cost no more than matrix materialization. A layout far outside
    this range points to a reader that asks the library for too much.
    """
    default = default_profile(summary)
    top = default[default["dimensions"] == default["dimensions"].max()]
    flags = []
    checked = 0
    for (backend, layout), group in _ordered_groups(top):
        seconds = group.set_index("operation")["median_seconds"]
        if not {"full_read", "matrix_materialization", "random_rows"}.issubset(
            seconds.index
        ):
            continue
        checked += 1
        name = _layout_name(backend, layout)
        matrix = seconds["matrix_materialization"] / seconds["full_read"]
        rows = seconds["random_rows"] / seconds["matrix_materialization"]
        if matrix > ACCESS_PATH_LIMIT:
            flags.append(
                f"{name}: matrix materialization takes {_multiplier(matrix)}x as "
                "long as a full read."
            )
        if rows > ACCESS_PATH_LIMIT:
            flags.append(
                f"{name}: random rows take {_multiplier(rows)}x as long as matrix "
                "materialization."
            )
    if not checked:
        return None
    if flags:
        return "Access path check: these layouts are outside the range. " + " ".join(
            flags
        )
    return (
        "Access path check: for every layout, matrix materialization takes at most "
        f"{ACCESS_PATH_LIMIT} times as long as a full read, and random rows take at "
        f"most {ACCESS_PATH_LIMIT} times as long as matrix materialization."
    )


def _blank_before(lines: list[str]) -> list[str]:
    """Return a blank line and the lines, or nothing when there are no lines."""
    return ["", *lines] if lines else []


def _two_digits(value: float) -> str:
    """Format a number with two significant digits and no exponent."""
    return f"{value:,.0f}" if value >= 100 else f"{value:.2g}"  # noqa: PLR2004


def _duration(seconds: float) -> str:
    """Format seconds as seconds, minutes, or hours."""
    if seconds >= 3600:  # noqa: PLR2004
        return f"{_two_digits(seconds / 3600)} h"
    if seconds >= 60:  # noqa: PLR2004
        return f"{_two_digits(seconds / 60)} min"
    return f"{_two_digits(seconds)} s"


def _dollars(amount: float) -> str:
    """Format dollars with cents below ten dollars."""
    return f"${amount:.2f}" if amount < 10 else f"${amount:,.0f}"  # noqa: PLR2004


def real_world_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Scale the benchmark to a 1.5 GB CSV wide file.

    Sizes scale by each layout's size relative to CSV wide. Read times scale
    linearly with the size. Download time and egress cost follow from the size.
    """
    if "profile" not in summary.columns:
        summary = summary.assign(profile="default")
    top = summary[summary["dimensions"] == summary["dimensions"].max()]

    def value(backend: str, layout: str, profile: str, operation: str, column: str):  # noqa: ANN202
        match = top[
            (top["backend"] == backend)
            & (top["layout"] == layout)
            & (top["profile"] == profile)
            & (top["operation"] == operation)
        ]
        return float(match.iloc[0][column]) if not match.empty else None

    baseline_bytes = value("csv", "wide", "default", "write", "artifact_bytes")
    baseline_read = value(
        "csv", "wide", "default", "matrix_materialization", "median_seconds"
    )
    if baseline_bytes is None or baseline_read is None:
        return pd.DataFrame()
    factor = EXAMPLE_DATASET_GB * 1e9 / baseline_bytes
    rows_per_file = float(top.iloc[0]["rows"]) * factor

    records = []
    for backend, layout, profile in EXAMPLE_LAYOUTS:
        size = value(backend, layout, profile, "write", "artifact_bytes")
        read = value(
            backend, layout, profile, "matrix_materialization", "median_seconds"
        )
        if size is None or read is None:
            continue
        size_gb = size / baseline_bytes * EXAMPLE_DATASET_GB
        download = size_gb * 1000 / EXAMPLE_DOWNLOAD_MB_PER_SECOND
        records.append(
            {
                "backend": backend,
                "layout": layout,
                "profile": profile,
                "rows_per_file": rows_per_file,
                "size_gb": size_gb,
                "download_seconds": download,
                "read_seconds": read * factor,
                "total_seconds": download + read * factor,
                "egress_dollars": size_gb * EXAMPLE_EGRESS_DOLLARS_PER_GB,
            }
        )
    table = pd.DataFrame(records)
    first = table.iloc[0]
    table["time_saved_seconds"] = (
        first["total_seconds"] - table["total_seconds"]
    ) * EXAMPLE_USES
    table["egress_saved_dollars"] = (
        first["egress_dollars"] - table["egress_dollars"]
    ) * EXAMPLE_USES
    return table


def _example_name(row: dict[str, Any]) -> str:
    name = _layout_name(row["backend"], row["layout"])
    return f"{name} (compact)" if row["profile"] == "compact" else name


def _example_takeaway(records: list[dict[str, Any]]) -> str | None:
    """Return one sentence on what Parquet's array layout saves, if it is present."""
    baseline = records[0]
    match = [
        row
        for row in records
        if (row["backend"], row["layout"], row["profile"])
        == ("parquet", "fixed_array", "default")
    ]
    if not match:
        return None
    row = match[0]
    return (
        f"**Takeaway.** With {_example_name(row)}, one use takes "
        f"{_duration(row['total_seconds'])} instead of "
        f"{_duration(baseline['total_seconds'])} and costs "
        f"{_dollars(row['egress_dollars'])} instead of "
        f"{_dollars(baseline['egress_dollars'])} in egress. Over {EXAMPLE_USES:,} "
        f"uses that saves {_duration(row['time_saved_seconds'])} and "
        f"{_dollars(row['egress_saved_dollars'])}."
    )


def real_world_section(summary: pd.DataFrame) -> list[str]:
    """Return the real-world example: time and egress for a 1.5 GB CSV wide file."""
    table = real_world_table(summary)
    if table.empty:
        return []
    records = table.to_dict("records")
    rows_text = f"{round(records[0]['rows_per_file'], -2):,.0f}"
    features = int(summary["dimensions"].max())
    one_use = [
        "| Layout | Size | Download | Read into memory | Total time | Egress cost |",
        "| ------ | ---- | -------- | ---------------- | ---------- | ----------- |",
        *(
            f"| {_example_name(row)} | {_two_digits(row['size_gb'])} GB "
            f"| {_duration(row['download_seconds'])} "
            f"| {_duration(row['read_seconds'])} "
            f"| {_duration(row['total_seconds'])} "
            f"| {_dollars(row['egress_dollars'])} |"
            for row in records
        ),
    ]
    savings = [
        "| Layout | Time saved | Egress saved |",
        "| ------ | ---------- | ------------ |",
        *(
            f"| {_example_name(row)} | {_duration(row['time_saved_seconds'])} "
            f"| {_dollars(row['egress_saved_dollars'])} |"
            for row in records[1:]
        ),
    ]
    return [
        "## Real-world example",
        "",
        (
            "Egress is the fee that a cloud provider charges when data leaves its "
            "network. This example asks what it costs to move and read a "
            f"{EXAMPLE_DATASET_GB:g} GB CSV wide file, and how much the other "
            f"layouts save. The file holds about {rows_text} rows of "
            f"{features:,} features. A use is one download followed by one read "
            "into memory."
        ),
        *_optional_paragraph(_example_takeaway(records)),
        "",
        "### One use",
        "",
        *one_use,
        "",
        f"### Savings over {EXAMPLE_USES:,} uses",
        "",
        (
            f"Each cell compares a layout with CSV wide over {EXAMPLE_USES:,} uses. "
            "Time saved is the sum of the download and read times."
        ),
        "",
        *savings,
        "",
        "Assumptions:",
        "",
        f"- Download speed is {EXAMPLE_DOWNLOAD_MB_PER_SECOND} MB/s.",
        (
            f"- Egress costs ${EXAMPLE_EGRESS_DOLLARS_PER_GB:.2f} per GB. This is "
            "the AWS list price for data transfer out to the internet, first 10 TB "
            "each month ([AWS S3 pricing](https://aws.amazon.com/s3/pricing/)). "
            "The first 100 GB each month is free on AWS. The table ignores this, "
            "so it overstates the cost at low volume."
        ),
        (
            "- Sizes and read times scale linearly from the benchmark data. The "
            "benchmark does not measure files above 65 MB."
        ),
        (
            "- Read times use one thread and warm caches, except for Lance and "
            "Vortex. See Limitations."
        ),
        "- 1 GB is 1,000,000,000 bytes.",
    ]


def _cpu_time_note(summary: pd.DataFrame) -> str | None:
    """Compare array and wide layouts in CPU time for multi-threaded backends.

    CPU time counts every thread, so the comparison does not favor a format
    that spreads its work over several threads.
    """
    columns = {"median_cpu_seconds", "median_parallelism"}
    if not columns.issubset(summary.columns):
        return None
    top = summary[
        (summary["dimensions"] == summary["dimensions"].max())
        & (summary["operation"] == "matrix_materialization")
    ]
    busy = top[top["median_parallelism"] > PARALLELISM_LIMIT]["backend"].unique()
    clauses = []
    for backend in _ordered_backends(busy):
        layouts = top[top["backend"] == backend]
        wide = layouts[layouts["layout"] == "wide"]
        arrays = layouts[layouts["layout"] != "wide"]
        if wide.empty or arrays.empty:
            continue
        array = arrays.iloc[0]
        wall = float(array["median_seconds"]) / float(wide.iloc[0]["median_seconds"])
        cpu = float(array["median_cpu_seconds"]) / float(
            wide.iloc[0]["median_cpu_seconds"]
        )
        clauses.append(
            f"{_speed_words(cpu)} in {_named(backend)} ({_speed_words(wall)} in "
            "wall time)"
        )
    if not clauses:
        return None
    return (
        "Measured in CPU time, which counts every thread, matrix materialization "
        f"with the array-like layout is {_join_words(clauses)}."
    )


def _speed_words(ratio: float) -> str:
    """Describe a time ratio as a multiplier that is faster or slower."""
    if ratio < 1:
        return f"{_multiplier(1 / ratio)}x faster"
    return f"{_multiplier(ratio)}x slower"


def _run_description(summary: pd.DataFrame) -> str:
    """Describe the data size and repeat count of a benchmark run."""
    rows = int(summary["rows"].max())
    smallest = int(summary["dimensions"].min())
    largest = int(summary["dimensions"].max())
    repetitions = int(summary["repetitions"].max())
    unit = "run" if repetitions == 1 else "runs"
    return (
        f"The benchmark used synthetic data with {rows:,} rows and "
        f"{smallest:,} to {largest:,} features. "
        f"Each timing is the median of {repetitions} repeated {unit}."
    )


def _layout_name(backend: str, layout: str) -> str:
    """Return a backend display name with the layout in code font."""
    return f"{BACKEND_DISPLAY_NAMES.get(backend, backend)} `{layout}`"


def _figure_title(path: Path) -> str:
    """Return a readable title for a generated figure path."""
    name = path.stem.replace("_absolute_comparison", "")
    return name.replace("_", " ").title()
