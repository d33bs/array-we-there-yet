"""Report generation for benchmark outputs."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

RESULTS_START = "<!-- array-we-there-yet-results:start -->"
RESULTS_END = "<!-- array-we-there-yet-results:end -->"
OPERATIONS = [
    "write",
    "full_read",
    "matrix_materialization",
    "random_rows",
    "feature_projection",
    "vector_norm",
]
BACKEND_ORDER = ["csv", "parquet", "duckdb", "vortex", "lance"]
LAYOUT_COLORS = {
    "wide": "#1f77b4",
    "fixed_array": "#ff7f0e",
    "delimited_array": "#2ca02c",
    "json_array": "#d62728",
    "duckdb_array": "#9467bd",
}


def ratio_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Calculate array-to-wide ratios for comparable backend operations."""
    baseline = summary[summary["layout"] == "wide"][
        [
            "backend",
            "dimensions",
            "operation",
            "operation_parameter",
            "median_seconds",
            "artifact_bytes",
        ]
    ].rename(
        columns={
            "median_seconds": "wide_median_seconds",
            "artifact_bytes": "wide_artifact_bytes",
        }
    )
    candidates = summary[summary["layout"] != "wide"].copy()
    ratios = candidates.merge(
        baseline,
        on=["backend", "dimensions", "operation", "operation_parameter"],
        how="inner",
    )
    ratios["time_ratio"] = ratios["median_seconds"] / ratios["wide_median_seconds"]
    ratios["artifact_size_ratio"] = (
        ratios["artifact_bytes"] / ratios["wide_artifact_bytes"]
    )
    return ratios


def write_figures(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> list[Path]:
    """Write absolute comparison figures and compact ratio figures."""
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    combined = write_combined_facet_overview(summary, figure_dir)
    if combined is not None:
        paths.append(combined)
    paths.extend(write_absolute_figures(summary, figure_dir))
    parquet_tracking = write_parquet_performance_figure(summary, figure_dir)
    if parquet_tracking is not None:
        paths.append(parquet_tracking)
    paths.extend(write_ratio_figures(summary, figure_dir))
    return paths


def write_ratio_tables(
    summary: pd.DataFrame,
    output_dir: Path = Path("results"),
) -> list[Path]:
    """Write array-to-wide ratio tables for detailed review."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ratios = ratio_table(summary)
    parquet_path = output_dir / "ratio_summary.parquet"
    csv_path = output_dir / "ratio_summary.csv"
    ratios.to_parquet(parquet_path, index=False)
    ratios.to_csv(csv_path, index=False)
    return [parquet_path, csv_path]


def write_combined_facet_overview(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> Path | None:
    """Write one compact faceted overview for the README."""
    ratios = ratio_table(summary)
    panels: list[tuple[str, pd.DataFrame, str, str]] = []
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
                    _ratio_axis_label("time_ratio"),
                )
            )

    time_summary = _time_summary_ratios(ratios)
    if not time_summary.empty:
        panels.extend(
            [
                (
                    _direction_title("Median Time", better="lower"),
                    time_summary,
                    "median_time_ratio",
                    _ratio_axis_label("time_ratio"),
                ),
                (
                    _direction_title("Worst Time", better="lower"),
                    time_summary,
                    "worst_time_ratio",
                    _ratio_axis_label("time_ratio"),
                ),
            ]
        )

    storage = ratios[ratios["operation"] == "write"]
    if not storage.empty:
        panels.append(
            (
                _direction_title("Storage Size", better="lower"),
                storage,
                "artifact_size_ratio",
                _ratio_axis_label("artifact_size_ratio"),
            )
        )
    if not panels:
        return None

    figure_dir.mkdir(parents=True, exist_ok=True)
    row_count, column_count = _facet_grid_shape(len(panels))
    fig, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(4.0 * column_count, 3.0 * row_count),
        squeeze=False,
        sharex=True,
    )
    legend_handles: dict[str, object] = {}
    colors = plt.get_cmap("tab10")
    group_labels = list(_ordered_group_labels(ratios))
    color_by_label = {
        label: colors(index % colors.N) for index, label in enumerate(group_labels)
    }

    for ax, (title, data, value_column, ylabel) in zip(
        axes.ravel(),
        panels,
        strict=False,
    ):
        for label, group in _ordered_groups(data):
            sorted_group = group.sort_values("dimensions")
            label_text = "/".join(label)
            line = ax.plot(
                sorted_group["dimensions"],
                sorted_group[value_column],
                marker="o",
                linewidth=1.7,
                label=label_text,
                color=color_by_label.get(label),
            )[0]
            legend_handles.setdefault(label_text, line)
        ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(title)
        ax.set_xlabel("Feature count")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    for ax in axes.ravel()[len(panels) :]:
        ax.set_axis_off()

    if legend_handles:
        fig.legend(
            handles=list(legend_handles.values()),
            loc="lower center",
            ncols=min(3, len(legend_handles)),
            fontsize="small",
        )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    path = figure_dir / "combined_facet_overview.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def parquet_performance_table(summary: pd.DataFrame) -> pd.DataFrame:
    """Return Parquet-only performance rows with array-to-wide ratios."""
    parquet_rows = summary[summary["backend"] == "parquet"].copy()
    ratios = ratio_table(summary)
    parquet_ratios = ratios[ratios["backend"] == "parquet"][
        [
            "dimensions",
            "layout",
            "operation",
            "operation_parameter",
            "time_ratio",
            "artifact_size_ratio",
        ]
    ]
    tracked = parquet_rows.merge(
        parquet_ratios,
        on=["dimensions", "layout", "operation", "operation_parameter"],
        how="left",
    )
    tracked["artifact_megabytes"] = tracked["artifact_bytes"] / 1_000_000
    columns = [
        "backend",
        "layout",
        "dimensions",
        "operation",
        "operation_parameter",
        "median_seconds",
        "q25_seconds",
        "q75_seconds",
        "time_ratio",
        "artifact_bytes",
        "artifact_megabytes",
        "artifact_size_ratio",
        "repetitions",
    ]
    return tracked[columns].sort_values(["dimensions", "operation", "layout"])


def write_parquet_performance_tables(
    summary: pd.DataFrame,
    output_dir: Path = Path("results"),
) -> list[Path]:
    """Write Parquet-focused performance tracking tables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tracked = parquet_performance_table(summary)
    parquet_path = output_dir / "parquet_performance.parquet"
    csv_path = output_dir / "parquet_performance.csv"
    tracked.to_parquet(parquet_path, index=False)
    tracked.to_csv(csv_path, index=False)
    return [parquet_path, csv_path]


def write_parquet_performance_figure(
    summary: pd.DataFrame,
    figure_dir: Path = Path("figures"),
) -> Path | None:
    """Write one Parquet-only tracking figure."""
    parquet_rows = summary[summary["backend"] == "parquet"].copy()
    if parquet_rows.empty:
        return None
    figure_dir.mkdir(parents=True, exist_ok=True)
    parquet_rows["artifact_megabytes"] = parquet_rows["artifact_bytes"] / 1_000_000
    panels = [
        (
            "write",
            "median_seconds",
            _direction_title("Write time", better="lower"),
            "Median seconds",
        ),
        (
            "full_read",
            "median_seconds",
            _direction_title("Full read", better="lower"),
            "Median seconds",
        ),
        (
            "matrix_materialization",
            "median_seconds",
            _direction_title("Matrix materialization", better="lower"),
            "Median seconds",
        ),
        (
            "write",
            "artifact_megabytes",
            _direction_title("Storage size", better="lower"),
            "Artifact size (MB)",
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), squeeze=False)
    legend_labels: list[str] = []
    for ax, (operation, value_column, title, ylabel) in zip(
        axes.ravel(),
        panels,
        strict=True,
    ):
        panel_data = parquet_rows[parquet_rows["operation"] == operation]
        for layout in _ordered_layouts(panel_data["layout"].unique()):
            layout_data = panel_data[panel_data["layout"] == layout].sort_values(
                "dimensions"
            )
            if layout_data.empty:
                continue
            ax.plot(
                layout_data["dimensions"],
                layout_data[value_column],
                marker="o",
                label=layout,
                color=LAYOUT_COLORS.get(layout),
            )
            if layout not in legend_labels:
                legend_labels.append(layout)
        ax.set_title(title)
        ax.set_xlabel("Feature count")
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.25)

    if legend_labels:
        legend_handles = [
            Patch(facecolor=LAYOUT_COLORS.get(label, "#7f7f7f"), label=label)
            for label in legend_labels
        ]
        fig.legend(handles=legend_handles, loc="lower center", ncols=2)
    fig.suptitle("Parquet Performance Tracking")
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    path = figure_dir / "parquet_performance_tracking.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


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
            ax.plot(
                sorted_group["dimensions"],
                sorted_group["time_ratio"],
                marker="o",
                label="/".join(label),
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
            ax.plot(
                sorted_group["dimensions"],
                sorted_group["artifact_size_ratio"],
                marker="o",
                label="/".join(label),
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
    """Return CSV, Parquet, DuckDB, Vortex, Lance, then unknown backends."""
    names = [str(backend) for backend in backends]
    known = [backend for backend in BACKEND_ORDER if backend in names]
    unknown = sorted(backend for backend in names if backend not in BACKEND_ORDER)
    return [*known, *unknown]


def _facet_grid_shape(panel_count: int) -> tuple[int, int]:
    """Return row and column counts with at most three columns."""
    column_count = min(3, max(panel_count, 1))
    row_count = (panel_count + column_count - 1) // column_count
    return row_count, column_count


def _time_summary_ratios(ratios: pd.DataFrame) -> pd.DataFrame:
    """Return median and worst time ratios across timed operations."""
    timed = ratios[ratios["operation"].isin(OPERATIONS)]
    if timed.empty:
        return pd.DataFrame()
    return (
        timed.groupby(["backend", "layout", "dimensions"], as_index=False)
        .agg(
            median_time_ratio=("time_ratio", "median"),
            worst_time_ratio=("time_ratio", "max"),
        )
        .sort_values(["backend", "layout", "dimensions"])
    )


def _direction_title(title: str, *, better: str) -> str:
    """Return a plot title with a direction cue."""
    return f"{title}\n({better} is better)"


def _ratio_axis_label(value_column: str) -> str:
    """Return the y-axis label for a ratio plot."""
    if value_column == "artifact_size_ratio":
        return "Size ratio (array / wide)"
    return "Time ratio (array / wide)"


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


def update_readme(
    *,
    readme_path: Path,
    summary: pd.DataFrame,
    figure_paths: list[Path],
) -> None:
    """Insert the latest benchmark result summary into the README."""
    text = readme_path.read_text(encoding="utf-8")
    section = render_results_section(summary=summary, figure_paths=figure_paths)
    replacement = f"{RESULTS_START}\n\n{section}\n\n{RESULTS_END}"
    if RESULTS_START in text and RESULTS_END in text:
        before = text.split(RESULTS_START, maxsplit=1)[0]
        after = text.split(RESULTS_END, maxsplit=1)[1]
        text = before + replacement + after
    else:
        text = text.rstrip() + "\n\n" + replacement + "\n"
    readme_path.write_text(text, encoding="utf-8")


def render_results_section(
    *,
    summary: pd.DataFrame,
    figure_paths: list[Path],
) -> str:
    """Render a concise Markdown result section."""
    ratios = ratio_table(summary)
    max_dimension = int(summary["dimensions"].max())
    latest = ratios[ratios["dimensions"] == max_dimension].copy()
    latest["time_ratio"] = latest["time_ratio"].round(3)
    latest["artifact_size_ratio"] = latest["artifact_size_ratio"].round(3)
    highlights = result_highlights(latest)
    lines = [
        "## Current Results",
        "",
        f"These starter results use synthetic data with {max_dimension} features.",
        "A ratio less than 1.0 favors the array-like layout.",
        ("The primary plot shows each array-like result divided by its wide baseline."),
        "",
        "Use the plot images below to see the main comparisons.",
        "Use the linked tables when you need exact rows.",
        "",
        "Summary of findings:",
        "",
    ]
    lines.extend(f"- {highlight}" for highlight in highlights)
    lines.extend(
        [
            "",
            "Primary facet plots:",
        ]
    )
    combined_paths = [
        path for path in figure_paths if path.name == "combined_facet_overview.png"
    ]
    parquet_tracking_paths = [
        path for path in figure_paths if path.name == "parquet_performance_tracking.png"
    ]
    if combined_paths:
        path = combined_paths[0]
        lines.extend(
            [
                "",
                f"![Benchmark Ratios]({path.as_posix()})",
                "",
                (
                    "Figure 1. Time panels show time ratios. "
                    "The storage panel shows size ratios. "
                    "Matrix materialization means reading the data into one "
                    "`N x D` numeric array. "
                    "Feature projection means reading only selected features. "
                    "Vector norm means the length of each feature vector. "
                    "Median Time shows the middle time ratio. "
                    "Worst Time shows the largest time ratio. "
                    "Ratios let different operations fit in one compact figure. "
                    "The y-axis uses a log scale to show small and large changes. "
                    "Values less than 1.0 favor the array-like layout."
                ),
            ]
        )
    if parquet_tracking_paths:
        path = parquet_tracking_paths[0]
        lines.extend(
            [
                "",
                "Parquet performance tracking:",
                "",
                f"![Parquet Performance Tracking]({path.as_posix()})",
                "",
                (
                    "Figure 2. Parquet tracking shows absolute time and "
                    "storage for wide and fixed-array layouts. "
                    "Lower values are better."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "Detailed result files:",
            "",
            "- [CSV ratio table](results/ratio_summary.csv)",
            "- [Parquet ratio table](results/ratio_summary.parquet)",
            "- [CSV Parquet tracking table](results/parquet_performance.csv)",
            ("- [Parquet tracking table](results/parquet_performance.parquet)"),
            "",
            (
                "Raw results are in `results/raw_results.parquet` "
                "and `results/raw_results.csv`."
            ),
            (
                "Summary results are in `results/summary.parquet` "
                "and `results/summary.csv`."
            ),
            "",
            "Detailed per-operation plot files remain in `figures/`.",
        ]
    )
    return "\n".join(lines)


def result_highlights(latest: pd.DataFrame) -> list[str]:
    """Return short plain-language highlights for the newest result table."""
    if latest.empty:
        return ["No comparable array-like results are available yet."]

    fastest = latest.loc[latest["time_ratio"].idxmin()]
    slowest = latest.loc[latest["time_ratio"].idxmax()]
    write_rows = latest[latest["operation"] == "write"]
    smallest = write_rows.loc[write_rows["artifact_size_ratio"].idxmin()]
    largest = write_rows.loc[write_rows["artifact_size_ratio"].idxmax()]

    return [
        (
            f"The largest time gain is {_result_label(fastest)} "
            f"at {_ratio_text(float(fastest['time_ratio']))}."
        ),
        (
            f"The largest time loss is {_result_label(slowest)} "
            f"at {_ratio_text(float(slowest['time_ratio']))}."
        ),
        (
            f"The smallest array-like artifact is {_storage_label(smallest)} "
            f"at {_ratio_text(float(smallest['artifact_size_ratio']))}."
        ),
        (
            f"The largest array-like artifact is {_storage_label(largest)} "
            f"at {_ratio_text(float(largest['artifact_size_ratio']))}."
        ),
    ]


def _result_label(row: pd.Series) -> str:
    """Return one short result label."""
    return (
        f"{row['backend']} {row['layout']} "
        f"for {_plain_operation(str(row['operation']))}"
    )


def _storage_label(row: pd.Series) -> str:
    """Return one short storage label."""
    return f"{row['backend']} {row['layout']}"


def _plain_operation(operation: str) -> str:
    """Return an operation name for prose."""
    return operation.replace("_", " ")


def _ratio_text(value: float) -> str:
    """Return a rounded ratio string."""
    return f"{float(value):.3g}x wide"


def _figure_title(path: Path) -> str:
    """Return a readable title for a generated figure path."""
    name = path.stem.replace("_absolute_comparison", "")
    return name.replace("_", " ").title()
