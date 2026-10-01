"""Tests for the numbered Figures section and the references to it."""

import re

import pandas as pd

from array_we_there_yet.report import (
    FIGURE_IDS,
    figure_numbers,
    real_world_bullet,
    render_results_section,
    summary_section,
)
from array_we_there_yet.site import ASSET_DIR, build_page, render_markdown
from tests.test_real_world import _summary as _real_world_summary
from tests.test_scale_report import _sweep, _wide_scaling, _wide_summary
from tests.test_summary import _environment, _story_summary

TITLES = [
    "Every layout against CSV wide",
    "Array-like layouts against their own wide layout",
    "Wide layouts",
    "Absolute times and sizes",
    "Time and egress for the real-world example",
    "Time against row count",
    "Storage size against time",
]


def _section(*, sweep: pd.DataFrame | None = None) -> str:
    return render_results_section(
        summary=_wide_summary(),
        figure_ids=FIGURE_IDS,
        sweep=sweep,
        scaling=_wide_scaling(),
    )


def _headings(section: str, prefix: str) -> list[str]:
    return [line for line in section.splitlines() if line.startswith(prefix)]


def test_figures_have_their_own_section_between_summary_and_real_world() -> None:
    """The plots are no longer spread through the page."""
    headings = _headings(_section(sweep=_sweep()), "## ")

    assert headings == [
        "## Summary",
        "## Figures",
        "## Real-world example",
        "## Detailed results",
    ]


def test_every_figure_is_numbered_in_order_with_its_own_heading() -> None:
    """Each figure has a heading such as "Figure 1. Wide layouts"."""
    section = _section(sweep=_sweep())

    figures = _headings(section, "### Figure ")
    assert figures == [
        f"### Figure {number}. {title}" for number, title in enumerate(TITLES, 1)
    ]


def test_each_placeholder_sits_directly_under_its_heading() -> None:
    """A reader sees the plot right after its number and title."""
    section = _section(sweep=_sweep())

    placeholders = re.findall(
        r'### Figure (\d+)\. [^\n]+\n\n<div class="figure" '
        r'data-figure="([a-z_]+)"></div>',
        section,
    )
    assert [figure for _, figure in placeholders] == [
        "combined",
        "backend_wide",
        "wide_layouts",
        "explorer",
        "real_world",
        "row_scaling",
        "profiles",
    ]
    assert section.count('<div class="figure"') == len(TITLES)


def test_the_key_and_the_figures_share_a_section_that_the_key_sticks_within() -> None:
    """The key stays in view only while the reader is among the figures."""
    section = _section(sweep=_sweep())

    opened = section.index('<section class="figures-section">')
    key = section.index('<div id="filters" class="filters"></div>')
    first = section.index("### Figure 1.")
    closed = section.index("</section>")
    real_world = section.index("## Real-world example")
    assert opened < key < first < closed < real_world
    assert section.count('id="filters"') == 1


def test_figure_numbers_follow_the_figures_that_have_data() -> None:
    """Without a sweep the row scaling figure is skipped and the rest move up."""
    with_sweep = figure_numbers(
        FIGURE_IDS, summary=_wide_summary(), sweep=_sweep(), scaling=_wide_scaling()
    )
    without = figure_numbers(
        FIGURE_IDS, summary=_wide_summary(), sweep=None, scaling=_wide_scaling()
    )

    assert with_sweep["row_scaling"] == FIGURE_IDS.index("row_scaling") + 1
    assert with_sweep["profiles"] == with_sweep["row_scaling"] + 1
    assert "row_scaling" not in without
    assert without["profiles"] == with_sweep["row_scaling"]


def test_the_summary_points_at_the_figures_that_show_each_claim() -> None:
    """Each bullet names the figure that depicts it."""
    lines = summary_section(
        _story_summary(), figure_numbers={"backend_wide": 2, "profiles": 7}
    )

    text = "\n".join(lines)
    matrix = next(line for line in lines if "Whole-matrix reads" in line)
    assert matrix.endswith("See [Figure 2](#figure-2).")
    few = next(line for line in lines if "Selecting a few features" in line)
    assert few.endswith("See [Figure 2](#figure-2).")
    assert "[Figure 7](#figure-7)" not in text or "Encoding settings" in text


def test_the_real_world_bullet_names_the_figure() -> None:
    """The result points at the calculator."""
    bullet = real_world_bullet(_real_world_summary(), None, figure_number=5)

    assert bullet is not None
    assert bullet.endswith("See Real-world example and [Figure 5](#figure-5).")
    bullet = real_world_bullet(_real_world_summary(), None)
    assert bullet is not None
    assert bullet.endswith("See Real-world example.")


def test_the_text_refers_to_the_figures_that_depict_it() -> None:
    """Prose in the other sections links to the figure by number."""
    section = _section(sweep=_sweep())

    real_world = section[section.index("## Real-world example") :]
    assert "[Figure 5](#figure-5) shows the time and egress cost" in real_world
    key_findings = section[section.index("### Key findings") :]
    assert "[Figure 2](#figure-2) shows these ratios" in key_findings
    scaling = section[section.index("### Scaling with row count") :]
    assert "[Figure 6](#figure-6) plots" in scaling
    appendix = section[section.index("### Appendix") :]
    assert "[Figure 7](#figure-7) plots" in appendix


def test_every_figure_link_points_at_a_figure_that_exists() -> None:
    """A renumbered figure never leaves a dead link behind."""
    section = _section()

    links = set(re.findall(r"\(#figure-(\d+)\)", section))
    headings = set(re.findall(r"### Figure (\d+)\.", section))
    assert links
    assert links <= headings


def test_figure_headings_get_stable_ids_for_links() -> None:
    """A figure heading is reachable as figure-N whatever its title."""
    html, headings = render_markdown(
        "### Figure 3. Wide layouts\n\n### Figure 10. Long title here\n", used=set()
    )

    assert [heading["id"] for heading in headings] == ["figure-3", "figure-10"]
    assert '<h3 id="figure-3">' in html


def test_the_page_has_no_contents_list_and_one_key_inside_the_figures() -> None:
    """The contents list is gone. Figures are still linked by number."""
    page = build_page(
        summary=_wide_summary(),
        environment=_environment(),
        sweep=_sweep(),
        scaling=_wide_scaling(),
    )

    assert "<nav" not in page
    assert "Contents" not in page
    assert 'class="toc"' not in page
    for number in range(1, len(TITLES) + 1):
        assert f'<h3 id="figure-{number}">' in page
    assert page.count('id="filters"') == 1
    assert page.index('id="filters"') > page.index('<h2 id="figures">')


def test_the_page_is_narrower_than_before() -> None:
    """The reading width is at most 56rem."""
    css = (ASSET_DIR / "report.css").read_text(encoding="utf-8")

    match = re.search(r"--measure:\s*(\d+)rem", css)
    assert match
    assert int(match.group(1)) <= 56  # noqa: PLR2004
