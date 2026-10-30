import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
COLAB_LINK = (
    "https://colab.research.google.com/github/jamilhammoudeh/vbar/blob/main/"
    "examples/quickstart.ipynb"
)
BADGE_IMAGE = "https://colab.research.google.com/assets/colab-badge.svg"

# The badge form Colab documents: an image wrapped in a link, both in Markdown.
BADGE = re.compile(r"\[!\[[^\]]*\]\(([^)\s]+)\)\]\(([^)\s]+)\)")


@pytest.mark.parametrize("page", ["README.md", "docs/evaluation.md"])
def test_page_links_to_the_notebook_in_colab(page):
    assert COLAB_LINK in (ROOT / page).read_text(encoding="utf-8")


def test_colab_link_points_at_a_file_in_the_repository():
    path = COLAB_LINK.split("/blob/main/", 1)[1]
    assert (ROOT / path).is_file(), f"{path} is missing"


def test_notebook_badge_is_the_colab_badge_linking_to_the_notebook():
    path = ROOT / "examples" / "quickstart.ipynb"
    assert path.is_file(), f"{path} is missing"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    first = next(cell for cell in notebook["cells"] if cell["cell_type"] == "markdown")
    source = first["source"] if isinstance(first["source"], str) else "".join(first["source"])
    found = BADGE.findall(source)
    assert found == [(BADGE_IMAGE, COLAB_LINK)]
