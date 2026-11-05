"""The settings template holds only blank names, and a real .env stays out of version control."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / ".env.example"


def ignored_by_git(name):
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    # --no-index judges the patterns alone, as a file that does not exist yet would be.
    run = subprocess.run([git, "check-ignore", "--no-index", "-q", name], cwd=ROOT)
    if run.returncode not in (0, 1):
        pytest.skip("not a git checkout")
    return run.returncode == 0


def template_lines():
    return TEMPLATE.read_text(encoding="utf-8").splitlines()


def test_template_holds_only_comments_blank_lines_and_empty_assignments():
    other = [
        line
        for line in template_lines()
        if line.strip() and not line.startswith("#") and not re.fullmatch(r"[A-Z][A-Z0-9_]*=", line)
    ]
    assert other == []


def test_template_names_the_key_and_the_model():
    names = [line[:-1] for line in template_lines() if re.fullmatch(r"[A-Z][A-Z0-9_]*=", line)]
    assert sorted(names) == ["VBAR_LLM_API_KEY", "VBAR_LLM_MODEL"]


def test_gitignore_lists_the_env_file():
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in lines]


def test_env_file_is_ignored():
    assert ignored_by_git(".env")


def test_env_template_is_not_ignored():
    assert not ignored_by_git(".env.example")
