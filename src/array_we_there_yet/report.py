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
BACKEND_ORDER = ["csv", "parquet", "duckdb", "arrow_ipc"]
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
    paths = write_absolute_figures(summary, figure_dir)
    paths.extend(write_ratio_figures(summary, figure_dir))
    return paths


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
    """Return CSV, Parquet, DuckDB, Arrow IPC, then unknown backends."""
    names = [str(backend) for backend in backends]
    known = [backend for backend in BACKEND_ORDER if backend in names]
    unknown = sorted(backend for backend in names if backend not in BACKEND_ORDER)
    return [*known, *unknown]


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
    replacement = f"{RESULTS_START}\n{section}\n{RESULTS_END}"
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
    columns = [
        "backend",
        "layout",
        "operation",
        "operation_parameter",
        "time_ratio",
        "artifact_size_ratio",
    ]
    table = latest[columns].sort_values(["backend", "layout", "operation"])
    highlights = result_highlights(latest)
    lines = [
        "## Current Results",
        "",
        f"These starter results use synthetic data with {max_dimension} features.",
        "A ratio less than 1.0 favors the array-like layout.",
        (
            "The primary figures show absolute values with wide and array-like "
            "layouts side by side."
        ),
        "",
        "Use the table below to find the largest changes.",
        (
            "Then open the side-by-side figures to see the absolute size "
            "of each difference."
        ),
        "",
        "Read this first:",
    ]
    lines.extend(f"- {highlight}" for highlight in highlights)
    lines.extend(
        [
            "",
            table.to_markdown(index=False),
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
            "Primary side-by-side figures:",
        ]
    )
    absolute_paths = [
        path for path in figure_paths if "absolute_comparison" in path.name
    ]
    ratio_paths = [path for path in figure_paths if "ratio" in path.name]
    for path in absolute_paths:
        lines.append(f"- `{path.as_posix()}`")
    lines.extend(["", "Secondary ratio figures:"])
    for path in ratio_paths:
        lines.append(f"- `{path.as_posix()}`")
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
