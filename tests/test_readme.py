"""The README opens with the install and one episode, and names each optional extra."""

import ast
import re
import tomllib

from test_docs import ROOT

README = (ROOT / "README.md").read_text(encoding="utf-8")
EXTRAS = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "optional-dependencies"
]


def headings():
    """(level, title) of every Markdown heading outside code fences."""
    found, fenced = [], False
    for line in README.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        elif not fenced and (match := re.match(r"(#{1,6}) (.+)", line)):
            found.append((len(match.group(1)), match.group(2).strip()))
    return found


def section(title):
    """The text under a `## ` heading, up to the next one, code fences included."""
    body, inside, fenced = [], False, False
    for line in README.splitlines():
        if line.startswith("```"):
            fenced = not fenced
        if not fenced and line.startswith("## "):
            inside = line[3:].strip() == title
            continue
        if inside:
            body.append(line)
    return "\n".join(body)


def test_first_section_is_quick_start():
    second_level = [title for level, title in headings() if level == 2]
    assert second_level[0] == "Quick start"


def test_there_is_one_quick_start_heading_and_no_quickstart():
    titles = [title for _, title in headings() if re.fullmatch(r"quick\s*start", title, re.I)]
    assert titles == ["Quick start"]


def test_quick_start_gives_the_install_command():
    assert re.search(r"pip install vbar(?![\w\[])", section("Quick start"))


def test_quick_start_has_one_python_block_that_flies_one_episode_and_prints_it():
    blocks = re.findall(r"```python\n(.*?)```", section("Quick start"), re.S)
    assert len(blocks) == 1
    calls = [
        node.func.id
        for node in ast.walk(ast.parse(blocks[0]))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ]
    assert calls.count("run_episode") == 1
    assert "print" in calls


def test_llm_controllers_section_names_the_key_the_example_and_the_extra():
    body = section("LLM controllers")
    for text in ("VBAR_LLM_API_KEY", "examples/llm_controller.py", "vbar[llm]"):
        assert text in body, text


def test_readme_names_the_install_extra_of_every_optional_feature():
    missing = [extra for extra in EXTRAS if extra != "dev" and f"vbar[{extra}]" not in README]
    assert missing == []
