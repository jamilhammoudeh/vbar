"""The README badges are static images, so each one is held to the project metadata here."""

import re
import tomllib
from urllib.parse import unquote

from test_docs import ROOT

from vbar import __version__

README = (ROOT / "README.md").read_text(encoding="utf-8")
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def badge(label):
    """Message of the static badge with this label, URL escapes undone."""
    match = re.search(rf"https://img\.shields\.io/badge/{label}-(.+?)-[0-9a-z]+\)", README)
    assert match, f"no static {label} badge"
    return unquote(match.group(1))


def test_pypi_badge_shows_the_package_version():
    assert badge("pypi") == f"v{__version__}"


def test_python_badge_lists_every_supported_minor_version():
    minors = [
        c.rsplit(" :: ", 1)[1]
        for c in PROJECT["classifiers"]
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", c)
    ]
    assert badge("python") == " | ".join(minors)


def test_license_badge_names_the_project_license():
    assert badge("license") == PROJECT["license"]


def test_no_badge_asks_a_service_for_live_data():
    # GitHub's image proxy shows a broken image when a badge service answers slowly, which the
    # live endpoints (pypi/, github/) do at times; static badges come back at once.
    images = re.findall(r"\[!\[[^\]]*\]\((\S+?)\)\]", README)
    assert images
    for url in images:
        if url.startswith("https://img.shields.io/"):
            assert url.startswith("https://img.shields.io/badge/"), url
