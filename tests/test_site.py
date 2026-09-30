"""Tests for the single-page HTML report."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from array_we_there_yet.site import (
    AFTER_SETUP,
    ASSET_DIR,
    BEFORE_SETUP,
    PLOTLY_INTEGRITY,
    PLOTLY_URL,
    REPO_URL,
    _embed,
    build_page,
    render_markdown,
    static_section,
    write_site,
)
from tests.test_scale_report import _sweep, _wide_scaling, _wide_summary
from tests.test_summary import _environment

SECTIONS = [
    "Summary",
    "Figures",
    "Real-world example",
    "Detailed results",
    "Layouts",
    "Operations",
    "Methodology",
    "Environment",
    "Backends and access paths",
    "Write settings",
    "Limitations",
    "References",
    "Terminology",
]


def _page(sweep: pd.DataFrame | None = None) -> str:
    return build_page(
        summary=_wide_summary(),
        environment=_environment(),
        sweep=sweep,
        scaling=_wide_scaling(),
    )


def _data(page: str) -> dict:
    match = re.search(
        r'<script id="report-data" type="application/json">(.*?)</script>',
        page,
        re.DOTALL,
    )
    assert match
    return json.loads(match.group(1))


def test_page_holds_the_report_sections_in_order() -> None:
    """Results come first, then the method, as on the old README."""
    page = _page()

    titles = re.findall(r'<h2 id="[^"]+">(.*?)</h2>', page)
    assert titles == SECTIONS
    assert page.count("<h1>") == 1
    assert "<title>Array We There Yet?</title>" in page


def test_every_heading_has_a_unique_id() -> None:
    """Headings are link targets, so their ids must not repeat."""
    page = _page()

    ids = re.findall(r'<h[23] id="([^"]+)"', page)
    assert len(ids) == len(set(ids))


def test_figures_are_placeholders_that_the_script_fills() -> None:
    """The page has no images. Each figure is a div named by its id."""
    page = _page()

    figures = re.findall(r'<div class="figure" data-figure="([^"]+)"></div>', page)
    assert figures == [
        "combined",
        "backend_wide",
        "wide_layouts",
        "explorer",
        "real_world",
        "profiles",
    ]
    assert "<img" not in page
    assert "![" not in page
    assert ".png" not in page


def test_the_row_scaling_figure_appears_only_with_a_sweep() -> None:
    """Without a sweep there is no row scaling section or figure."""
    assert 'data-figure="row_scaling"' not in _page()
    with_sweep = _page(_sweep())

    assert 'data-figure="row_scaling"' in with_sweep
    assert _data(with_sweep)["rowScaling"]["dimensions"] == pytest.approx(1_024)


def test_embedded_data_holds_the_specs_and_the_filter_catalogue() -> None:
    """The script draws from one JSON block."""
    data = _data(_page())

    assert set(data) == {
        "backends",
        "layouts",
        "facets",
        "rowScaling",
        "profiles",
        "realWorld",
        "explorer",
    }
    assert data["backends"][0] == {"id": "csv", "label": "CSV"}
    assert {"combined", "backend_wide", "wide_layouts"} == set(data["facets"])
    assert data["rowScaling"] is None
    assert data["realWorld"]["datasetGb"] == pytest.approx(1.5)


def test_tables_become_html_tables() -> None:
    """Markdown tables render as tables, and code stays code."""
    page = _page()

    assert page.count("<table>") >= len(SECTIONS) // 2
    assert "<th>Backend</th>" in page
    assert "<code>fixed_array</code>" in page


def test_embedded_json_cannot_close_the_script_tag() -> None:
    """A label with a closing tag must not end the data block early."""
    text = _embed({"label": "</script><!-- x"})

    assert "</script>" not in text
    assert "<!--" not in text
    assert json.loads(text.replace("\\/", "/").replace("\\!", "!"))


def test_plotly_loads_from_a_pinned_cdn_file_with_an_integrity_hash() -> None:
    """A changed file on the CDN is refused by the browser."""
    page = _page()

    assert f'src="{PLOTLY_URL}"' in page
    assert f'integrity="{PLOTLY_INTEGRITY}"' in page
    assert "plotly-basic-2.35.2" in PLOTLY_URL
    assert PLOTLY_INTEGRITY.startswith("sha384-")


def test_every_static_section_has_a_content_file_that_starts_with_its_heading() -> None:
    """The method text lives in the package, not in the README."""
    for name in [*BEFORE_SETUP, *AFTER_SETUP]:
        assert static_section(name).startswith("## "), name


def test_markdown_headings_get_unique_ids() -> None:
    """Repeated titles get a numbered id."""
    used: set[str] = set()

    html, headings = render_markdown("## Same\n\n## Same\n", used=used)

    assert [heading["id"] for heading in headings] == ["same", "same-2"]
    assert '<h2 id="same-2">' in html


def test_write_site_writes_the_page_and_a_nojekyll_marker(tmp_path: Path) -> None:
    """GitHub Pages serves the folder as it is."""
    path = write_site(
        summary=_wide_summary(),
        environment=_environment(),
        scaling=_wide_scaling(),
        output_dir=tmp_path / "site",
    )

    assert path == tmp_path / "site" / "index.html"
    assert path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert (tmp_path / "site" / ".nojekyll").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_the_script_is_valid_javascript() -> None:
    """A syntax error would leave every plot empty."""
    result = subprocess.run(
        [str(shutil.which("node")), "--check", str(ASSET_DIR / "report.js")],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_report_text_and_diagram_fonts_gain_one_point() -> None:
    """The small text and SVG diagram grow with the main report copy."""
    css = (ASSET_DIR / "report.css").read_text(encoding="utf-8")
    assert "font-size: calc(100% + 1pt)" in css
    assert "font: 1rem/1.6" in css
    for size in ("13", "11", "12", "10.5"):
        assert f"font-size: calc({size}px + 1pt)" in css


def test_title_has_more_room_before_the_intro() -> None:
    """The masthead leaves a readable gap under the title row."""
    css = (ASSET_DIR / "report.css").read_text(encoding="utf-8")
    title_row = re.search(r"\.masthead-top\s*\{([^}]*)\}", css)
    assert title_row is not None
    assert "margin-bottom: 1rem" in title_row.group(1)


@pytest.mark.skipif(shutil.which("node") is None, reason="Node is not installed")
def test_y_axis_titles_name_the_log_scale_only_when_used(tmp_path: Path) -> None:
    """Axis titles follow the scale toggle and empty-panel placeholder axes."""
    page = tmp_path / "report.html"
    page.write_text(_page(_sweep()), encoding="utf-8")
    result = subprocess.run(
        [
            str(shutil.which("node")),
            str(Path(__file__).with_name("axis_scale.js")),
            str(page),
            str(ASSET_DIR / "report.js"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_the_intro_explains_the_idea_in_plain_words_for_a_newcomer() -> None:
    """A data scientist or generalist can follow it without the jargon first."""
    intro = static_section("intro")

    for term in ["*features*", "**wide**", "**array-like**"]:
        assert term in intro
    for backend in ["CSV", "Parquet", "DuckDB", "Zarr", "TileDB", "Vortex", "Lance"]:
        assert backend in intro
    assert "you" in intro.lower()
    assert "text" in intro and "CSV wide" in intro
    prose = re.sub(r"<[^>]+>", " ", intro)
    assert len(prose.split()) < 340  # noqa: PLR2004
    page = _page()
    assert "Picture a table where each row is one sample" in page


def test_the_intro_has_a_diagram_of_wide_versus_array_like() -> None:
    """A picture backs up the wide-versus-array-like explanation, up top."""
    intro = static_section("intro")

    assert "<figure" in intro
    assert "<svg" in intro
    assert 'role="img"' in intro
    assert "<title>" in intro or "<title " in intro
    figcaption = re.search(r"<figcaption>(.*?)</figcaption>", intro, re.DOTALL)
    assert figcaption
    caption = figcaption.group(1).lower()
    assert "wide" in caption
    assert "array-like" in caption
    # The diagram sits after the layouts are named and before "Does the choice".
    assert intro.index("<figure") > intro.index("array-like")
    assert intro.index("<figure") < intro.index("Does the choice")


def test_the_diagram_appears_on_the_built_page() -> None:
    """The diagram renders into the page, not just the source file."""
    page = _page()

    assert page.count("<svg") >= 1
    assert "layout-diagram" in page


def test_the_page_links_to_the_source_on_github_in_the_header_and_footer() -> None:
    """A reader can jump to the repository from the top and the bottom."""
    page = _page()

    assert page.count(f'href="{REPO_URL}"') >= 2  # noqa: PLR2004
    assert page.count("<svg") >= 3  # noqa: PLR2004 (diagram + 2 GitHub icons)
    header = page[: page.index("<main")]
    footer = page[page.index("</main>") :]
    assert REPO_URL in header
    assert REPO_URL in footer
    assert "<footer" in footer


def test_the_github_link_url_matches_the_citation_file() -> None:
    """The linked repository is the one this project actually publishes as."""
    citation = Path("CITATION.cff").read_text(encoding="utf-8")

    assert f'repository-code: "{REPO_URL}"' in citation


def test_layouts_frames_the_choice_as_a_documented_relational_alternative() -> None:
    """The Layouts section explains this is NF2, not a shortcut, and points on."""
    layouts = static_section("layouts")

    assert "First Normal Form" in layouts
    assert "NF" in layouts
    assert "1970" in layouts
    assert "1982" in layouts
    assert "See References" in layouts


def test_references_lists_the_six_theoretical_background_papers() -> None:
    """Each citation traces one claim: 1NF, NF2, specialization, arrays, ndarray."""
    group = static_section("references")
    for doi in [
        "10.1145/362384.362685",
        "10.1145/588111.588133",
        "10.1109/ICDE.2005.1",
        "10.1186/s40537-020-00399-2",
        "10.14778/1920841.1920886",
        "10.1038/s41586-020-2649-2",
    ]:
        assert doi in group
    for author in [
        "Codd",
        "Jaeschke",
        "Schek",
        "Stonebraker",
        "Baumann",
        "Melnik",
        "Harris",
    ]:
        assert author in group
    assert group.count("\n- ") >= 6  # noqa: PLR2004


def test_the_heading_uses_a_simple_seafoam_to_gold_gradient_with_no_red() -> None:
    """The title is a plain two-stop gradient, no glow."""
    css = (ASSET_DIR / "report.css").read_text(encoding="utf-8")

    heading = re.search(r"\.masthead h1\s*\{([^}]*)\}", css)
    assert heading
    rule = heading.group(1)
    assert "var(--title-a)" in rule
    assert "var(--title-b)" in rule
    assert "text-shadow" not in rule
    assert "text-shadow" not in css

    css_no_hash = css.replace("dashed", "").replace("gh-link", "")
    assert "#e11d48" not in css_no_hash
    assert "#fb7185" not in css_no_hash


def test_the_first_four_paragraphs_link_to_sources_also_in_references() -> None:
    """Conceptual claims in the intro link out, and the same source is cited below."""
    intro = static_section("intro")
    references = static_section("references")

    paragraphs = intro.split("\n\n")
    body = "\n\n".join(p for p in paragraphs if not p.startswith("<figure"))

    links = {
        "https://doi.org/10.1038/s41586-020-2649-2": "NumPy",
        "https://doi.org/10.1145/362384.362685": "**wide**",
        "https://doi.org/10.1145/588111.588133": "**array-like**",
        "https://doi.org/10.1109/ICDE.2005.1": None,
        "https://doi.org/10.1145/5666.5673": None,
    }
    for doi, term in links.items():
        assert doi in body, doi
        assert doi in references, doi
        if term:
            assert f"[{term}]({doi})" in body, term


def test_the_page_has_a_squared_plus_favicon() -> None:
    """The tab icon is an inline SVG of the chosen emoji, no image file needed."""
    page = _page()

    assert '<link rel="icon"' in page
    assert "⊞" in page
    assert "data:image/svg+xml" in page
