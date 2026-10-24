import re

import pytest
from test_docs import ROOT, python_blocks

from vbar.agents.sign_check import sign_check

README = ROOT / "README.md"


def test_readme_controller_passes_the_sign_check():
    section = README.read_text(encoding="utf-8").split("## Writing a controller")[1]
    (block,) = re.findall(r"```python\n(.*?)```", section.split("\n### ")[0], re.S)
    namespace: dict = {}
    exec(compile(block, str(README), "exec"), namespace)
    (agent_class,) = (v for v in namespace.values() if isinstance(v, type) and hasattr(v, "act"))
    check = sign_check(agent_class())
    assert check.passed, check.failures


# A reader copies one block, not the page, so these pages' blocks must run on their own.
@pytest.mark.parametrize("page", [README, ROOT / "docs" / "metrics.md"], ids=lambda p: p.name)
def test_every_python_example_runs_on_its_own(page, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for block in python_blocks(page):
        exec(compile(block, str(page), "exec"), {})
