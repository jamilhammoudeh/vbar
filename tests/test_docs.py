import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def python_blocks(page: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S)


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_python_example_in_the_docs_runs(page, tmp_path, monkeypatch):
    # Blocks on one page share a namespace, as a reader running them in order would.
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for block in python_blocks(page):
        exec(compile(block, str(page), "exec"), namespace)
