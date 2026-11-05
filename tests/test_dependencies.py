"""The core install stays at three packages; the language-model example's loader is an extra."""

import re
import tomllib
from pathlib import Path

PYPROJECT = tomllib.loads(
    (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
)
EXTRAS = PYPROJECT["project"]["optional-dependencies"]


def test_core_dependencies_are_numpy_scipy_and_gymnasium():
    names = [
        re.match(r"[A-Za-z0-9_.-]+", spec).group(0) for spec in PYPROJECT["project"]["dependencies"]
    ]
    assert sorted(names) == ["gymnasium", "numpy", "scipy"]


def test_llm_extra_installs_python_dotenv():
    assert EXTRAS["llm"] == ["python-dotenv>=1.0"]


def test_dev_extra_installs_python_dotenv_for_the_tests():
    assert "python-dotenv>=1.0" in EXTRAS["dev"]
