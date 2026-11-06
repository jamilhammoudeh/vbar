"""The README badges are SVG files in this repository, each held to the project metadata."""

import re
import tomllib

from test_docs import ROOT

from vbar import __version__

README = (ROOT / "README.md").read_text(encoding="utf-8")
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
HOSTED = "https://raw.githubusercontent.com/jamilhammoudeh/vbar/main/"
IMAGES = dict(re.findall(r"\[!\[([^\]]*)\]\((\S+?)\)\]", README))


def badge(name):
    """(label, message) of docs/badges/<name>.svg, checked against its visible text."""
    svg = (ROOT / "docs" / "badges" / f"{name}.svg").read_text(encoding="utf-8")
    label, message = re.search(r"<title>(.+?): (.+?)</title>", svg).groups()
    assert set(re.findall(r"<text[^>]*>([^<]*)</text>", svg)) == {label, message}
    return label, message


def test_every_hosted_badge_is_its_own_file_in_the_repository():
    hosted = {alt: url.removeprefix(HOSTED) for alt, url in IMAGES.items() if HOSTED in url}
    assert hosted == {
        "PyPI": "docs/badges/pypi.svg",
        "Python": "docs/badges/python.svg",
        "License": "docs/badges/license.svg",
        "DOI": "docs/badges/doi.svg",
        "Docs": "docs/badges/docs.svg",
    }
    for path in hosted.values():
        assert (ROOT / path).is_file(), path


def test_no_badge_comes_from_a_badge_service():
    # GitHub's image proxy times out on badge services and shows a broken image.
    assert not [url for url in IMAGES.values() if "shields.io" in url]


def test_pypi_badge_shows_the_package_version():
    assert badge("pypi") == ("pypi", f"v{__version__}")


def test_python_badge_lists_every_supported_minor_version():
    minors = [
        c.rsplit(" :: ", 1)[1]
        for c in PROJECT["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    ]
    assert badge("python") == ("python", " | ".join(minors))


def test_license_badge_names_the_project_license():
    assert badge("license") == ("license", PROJECT["license"])


def test_doi_badge_shows_the_doi_it_links_to():
    _, doi = badge("doi")
    assert f"(https://doi.org/{doi})" in README
