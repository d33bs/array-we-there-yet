"""Tests for the links and the References section of the README."""

import re

from array_we_there_yet.report import BACKEND_PACKAGES
from array_we_there_yet.site import AFTER_SETUP, static_section

URL = re.compile(r"\]\((https?://[^)\s]+)\)")


def _references() -> str:
    return static_section("references")


def test_every_backend_package_links_to_its_documentation() -> None:
    """A reader can go from each backend name to the project behind it."""
    for backend, package in BACKEND_PACKAGES.items():
        assert URL.search(package), backend
        assert "http://" not in package


def test_references_come_after_limitations_and_before_terminology() -> None:
    """The section sits at the end of the report, before the terms."""
    assert AFTER_SETUP.index("limitations") < AFTER_SETUP.index("references")
    assert AFTER_SETUP.index("references") < AFTER_SETUP.index("terminology")


def test_references_cover_the_formats_streaming_compression_and_method() -> None:
    """Each claim that needs a source has one."""
    text = _references()

    for expected in [
        "https://parquet.apache.org/docs/file-format/",
        "https://parquet.apache.org/docs/file-format/data-pages/encodings/",
        "https://arrow.apache.org/docs/format/Columnar.html",
        "https://zarr-specs.readthedocs.io/",
        "https://docs.tiledb.com/",
        "https://duckdb.org/docs/sql/data_types/array",
        "https://docs.vortex.dev/",
        "https://lance.org/",
        "https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html",
        "https://www.rfc-editor.org/rfc/rfc8878",
        "https://www.blosc.org/",
        "https://www.rfc-editor.org/rfc/rfc9110.html#name-range-requests",
        "https://doi.org/10.1145/5666.5673",
    ]:
        assert expected in text


def test_every_reference_is_a_secure_link_with_a_description() -> None:
    """Each bullet links over https and says what the source supports."""
    bullets = [line for line in _references().splitlines() if line.startswith("- ")]

    assert bullets
    for bullet in bullets:
        assert len(URL.findall(bullet)) == 1, bullet
        assert ": " in bullet or "). " in bullet, bullet
    assert "http://" not in _references()


def test_the_geometric_mean_is_cited() -> None:
    """The averaging method points to the paper that recommends it."""
    assert "Fleming" in _references()
    assert "Communications of the ACM" in _references()
