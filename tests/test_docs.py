import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]

# Modules a page may import that work only with an optional extra, mapped to the module the extra
# installs. vbar.viz imports matplotlib only when called, so it is listed by name; a page that
# imports vbar.agents.policy runs without onnx and is not listed.
EXTRA_MODULES = {
    "matplotlib": "matplotlib",
    "vbar.viz": "matplotlib",
    "onnx": "onnx",
    "onnxruntime": "onnxruntime",
}


def python_blocks(page: Path) -> list[str]:
    return re.findall(r"```python\n(.*?)```", page.read_text(encoding="utf-8"), re.S)


def extras_needed(blocks: list[str]) -> set[str]:
    imported = set()
    for block in blocks:
        for node in ast.walk(ast.parse(block)):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module)
    return {
        extra
        for name in imported
        for module, extra in EXTRA_MODULES.items()
        if name == module or name.startswith(module + ".")
    }


@pytest.mark.parametrize("page", PAGES, ids=lambda page: page.name)
def test_every_python_example_in_the_docs_runs(page, tmp_path, monkeypatch):
    blocks = python_blocks(page)
    for extra in sorted(extras_needed(blocks)):
        pytest.importorskip(extra)
    # Blocks on one page share a namespace, as a reader running them in order would.
    monkeypatch.chdir(tmp_path)
    namespace: dict = {}
    for block in blocks:
        exec(compile(block, str(page), "exec"), namespace)


def test_only_pages_that_import_an_extra_are_skipped_without_it():
    needs = {page.name: extras_needed(python_blocks(page)) for page in PAGES}
    assert needs["viz.md"] == {"matplotlib"}
    assert needs["policy-export.md"] == set()
    assert needs["README.md"] == set()
