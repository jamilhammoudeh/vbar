"""The workflows give the repository token the least it needs."""

import re
from pathlib import Path

WORKFLOWS = Path(__file__).parents[1] / ".github" / "workflows"
# A block of indented scopes, or an inline value such as `write-all` or `{ contents: write }`.
JOB_LEVEL = re.compile(r"^( +)permissions:[ \t]*(\S[^\n]*)?\n((?:\1 .+\n)*)", re.MULTILINE)
JOB_NAME = re.compile(r"^  ([\w-]+):\n", re.MULTILINE)


def workflows():
    paths = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    assert paths, "no workflow found"
    return {path.name: path.read_text(encoding="utf-8") for path in paths}


def test_every_workflow_defaults_the_token_to_reading_the_contents():
    for name, text in workflows().items():
        head = text.split("\njobs:")[0]
        assert re.search(r"^permissions:\n  contents: read\n(?! )", head, re.MULTILINE), name


def test_only_the_publish_job_asks_for_more_and_only_for_the_trusted_publishing_token():
    granted = {}
    for name, text in workflows().items():
        for found in JOB_LEVEL.finditer(text):
            job = [m.group(1) for m in JOB_NAME.finditer(text[: found.start()])][-1]
            inline = [found.group(2)] if found.group(2) else []
            granted[(name, job)] = inline + [line.strip() for line in found.group(3).splitlines()]
    assert granted == {("release.yml", "publish"): ["id-token: write"]}
