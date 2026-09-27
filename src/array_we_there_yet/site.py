"""Build the single-page HTML report.

The page holds the whole report: the summary, interactive plots, the real-world
example, the detailed results, and the method. It comes from the same code and
data as every other output, so the numbers agree everywhere.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, StrictUndefined
from markdown_it import MarkdownIt
from markdown_it.token import Token

from array_we_there_yet.plots import (
    catalog,
    explorer_spec,
    facet_specs,
    profile_spec,
    real_world_spec,
    row_scaling_spec,
)
from array_we_there_yet.report import (
    FIGURE_IDS,
    render_results_section,
    render_setup_section,
)

PAGE_TITLE = "Array We There Yet?"
PLOTLY_URL = "https://cdn.plot.ly/plotly-basic-2.35.2.min.js"
PLOTLY_INTEGRITY = (
    "sha384-wQ3lfCxuvLfhHGiSdHF5+e3NZ1zNwEMd8/nII+K7VKChF9llO/OwOcn/aVRkB7az"
)
REPO_URL = "https://github.com/d33bs/array-we-there-yet"
# Octicon "mark-github", used at 16x16 in the header and footer links.
GITHUB_ICON = (
    '<svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" '
    'focusable="false"><path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 '
    "3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01."
    "37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53."
    "63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1."
    "78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 "
    "0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 "
    "2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1."
    "87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15"
    '.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8Z"></path></svg>'
)
ASSET_DIR = Path(__file__).parent / "site"
CONTENT_DIR = Path(__file__).parent / "content"
# The order of the sections that come from static text. The setup section, which
# holds the environment and the backends, goes between Methodology and these.
BEFORE_SETUP = ["layouts", "operations", "methodology"]
AFTER_SETUP = ["write_settings", "limitations", "references", "terminology"]


def _markdown() -> MarkdownIt:
    """Return a CommonMark parser with tables and raw HTML blocks."""
    return MarkdownIt("commonmark", {"html": True}).enable("table")


FIGURE_HEADING = re.compile(r"Figure (\d+)\.")


def _slug(text: str) -> str:
    """Return a lowercase, dash-separated anchor name."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def _heading_text(inline: Token) -> str:
    """Return the plain text of a heading token."""
    return "".join(
        child.content for child in inline.children or [] if child.type == "text"
    )


def render_markdown(text: str, *, used: set[str]) -> tuple[str, list[dict[str, Any]]]:
    """Convert Markdown to HTML, give each heading an id, and list the headings.

    ``used`` holds the ids that earlier calls made, so that ids stay unique.
    """
    parser = _markdown()
    tokens = parser.parse(text)
    headings: list[dict[str, Any]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue
        title = _heading_text(tokens[index + 1])
        figure = FIGURE_HEADING.match(title)
        base = f"figure-{figure.group(1)}" if figure else _slug(title) or "section"
        anchor, count = base, 2
        while anchor in used:
            anchor = f"{base}-{count}"
            count += 1
        used.add(anchor)
        token.attrSet("id", anchor)
        headings.append({"level": int(token.tag[1]), "title": title, "id": anchor})
    return parser.renderer.render(tokens, parser.options, {}), headings


def static_section(name: str) -> str:
    """Return the Markdown of one static section."""
    return (CONTENT_DIR / f"{name}.md").read_text(encoding="utf-8")


def report_data(
    *,
    summary: pd.DataFrame,
    sweep: pd.DataFrame | None,
    scaling: pd.DataFrame | None,
) -> dict[str, Any]:
    """Return the data that the page embeds and the script draws."""
    return {
        **catalog(summary),
        "facets": facet_specs(summary),
        "rowScaling": row_scaling_spec(sweep),
        "profiles": profile_spec(summary),
        "realWorld": real_world_spec(summary, scaling),
        "explorer": explorer_spec(summary),
    }


def _embed(data: dict[str, Any]) -> str:
    """Return JSON that is safe inside a script tag."""
    return (
        json.dumps(data, allow_nan=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )


def build_page(  # noqa: PLR0913
    *,
    summary: pd.DataFrame,
    environment: dict[str, Any] | None = None,
    encodings: pd.DataFrame | None = None,
    floor: pd.DataFrame | None = None,
    sweep: pd.DataFrame | None = None,
    scaling: pd.DataFrame | None = None,
) -> str:
    """Return the report as one HTML document."""
    data = report_data(summary=summary, sweep=sweep, scaling=scaling)
    figure_ids = [figure for figure in FIGURE_IDS if _has_figure(figure, data)]
    sections = [
        render_results_section(
            summary=summary,
            figure_ids=figure_ids,
            encodings=encodings,
            floor=floor,
            sweep=sweep,
            scaling=scaling,
        ),
        *(static_section(name) for name in BEFORE_SETUP),
        "\n".join(render_setup_section(summary, environment)),
        *(static_section(name) for name in AFTER_SETUP),
    ]
    used: set[str] = set()
    body = []
    for markdown in sections:
        html, _ = render_markdown(markdown, used=used)
        body.append(html)
    intro, _ = render_markdown(static_section("intro"), used=used)
    template = Environment(autoescape=False, undefined=StrictUndefined).from_string(
        (ASSET_DIR / "template.html").read_text(encoding="utf-8")
    )
    return template.render(
        title=PAGE_TITLE,
        intro=intro,
        body="\n".join(body),
        data=_embed(data),
        css=(ASSET_DIR / "report.css").read_text(encoding="utf-8"),
        script=(ASSET_DIR / "report.js").read_text(encoding="utf-8"),
        plotly_url=PLOTLY_URL,
        plotly_integrity=PLOTLY_INTEGRITY,
        repo_url=REPO_URL,
        github_icon=GITHUB_ICON,
    )


def _has_figure(figure: str, data: dict[str, Any]) -> bool:
    """Return whether the data holds what a figure needs."""
    facets = data["facets"]
    return {
        "combined": "combined" in facets,
        "backend_wide": "backend_wide" in facets,
        "wide_layouts": "wide_layouts" in facets,
        "explorer": bool(data["explorer"]["rows"]),
        "real_world": data["realWorld"] is not None,
        "row_scaling": data["rowScaling"] is not None,
        "profiles": data["profiles"] is not None,
    }[figure]


def write_site(  # noqa: PLR0913
    *,
    summary: pd.DataFrame,
    environment: dict[str, Any] | None = None,
    encodings: pd.DataFrame | None = None,
    floor: pd.DataFrame | None = None,
    sweep: pd.DataFrame | None = None,
    scaling: pd.DataFrame | None = None,
    output_dir: Path = Path("site"),
) -> Path:
    """Write the page to ``index.html`` in the output folder."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "index.html"
    path.write_text(
        build_page(
            summary=summary,
            environment=environment,
            encodings=encodings,
            floor=floor,
            sweep=sweep,
            scaling=scaling,
        ),
        encoding="utf-8",
    )
    (output_dir / ".nojekyll").touch()
    return path
