import re
from pathlib import Path

import vbar

ROOT = Path(__file__).parent.parent


def test_citation_version_matches_package():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert re.search(r"^version: (\S+)$", citation, re.M).group(1) == vbar.__version__
