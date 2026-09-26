"""Tests for the real-world example figure."""

import re
from pathlib import Path

import pandas as pd
import pytest

from array_we_there_yet.report import (
    real_world_figure_data,
    render_results_section,
    write_figures,
    write_real_world_figure,
)
from tests.test_real_world import _summary
from tests.test_scale_report import _wide_scaling, _wide_summary


def test_figure_data_lists_time_and_egress_per_layout_slowest_first() -> None:
    """One row per layout with the download and read split, worst layout first."""
    data = real_world_figure_data(_summary(), None)

    assert list(data.columns) == [
        "name",
        "download_seconds",
        "read_seconds",
        "total_seconds",
        "egress_dollars_total",
    ]
    assert data["name"].iloc[0] == "CSV wide"
    assert data["total_seconds"].is_monotonic_decreasing
    assert "`" not in "".join(data["name"])
    csv = data.iloc[0]
    assert csv["egress_dollars_total"] == pytest.approx(135.0, abs=0.1)


def test_figure_is_written_and_absent_without_the_example_data(tmp_path: Path) -> None:
    """The figure needs CSV wide to be the baseline."""
    path = write_real_world_figure(_summary(), None, tmp_path)
    parquet_only = _summary()[lambda frame: frame["backend"] == "parquet"]

    assert path == tmp_path / "real_world_example.png"
    assert path.exists()
    assert write_real_world_figure(parquet_only, None, tmp_path) is None
    assert write_real_world_figure(pd.DataFrame(), None, tmp_path) is None


def test_the_figure_sits_after_the_takeaway_and_before_the_tables(
    tmp_path: Path,
) -> None:
    """A reader sees the picture first, then the one-use table."""
    summary = _summary()
    paths = write_figures(summary, tmp_path)

    section = render_results_section(summary=summary, figure_paths=paths)

    real_world = section[section.index("## Real-world example") :]
    takeaway = real_world.index("**Takeaway.**")
    image = real_world.index("![Time and egress")
    table = real_world.index("### One use")
    assert takeaway < image < table
    assert re.search(r"Figure \d+\. Left: time for one use", real_world)


def test_figure_leaves_out_the_streamed_layout_but_keeps_parquet_wide() -> None:
    """Streaming 8 features is not comparable with whole-file reads."""
    data = real_world_figure_data(_wide_summary(), _wide_scaling())

    assert "Parquet wide" in list(data["name"])
    assert not any("streamed" in name for name in data["name"])
    assert data["total_seconds"].is_monotonic_decreasing


def test_caption_does_not_mention_a_streamed_bar(tmp_path: Path) -> None:
    """The figure has no streamed bar, so the caption has no note about one."""
    text = render_results_section(
        summary=_wide_summary(),
        figure_paths=write_figures(_wide_summary(), tmp_path, scaling=_wide_scaling()),
        scaling=_wide_scaling(),
    )

    assert "The streamed bar" not in text
