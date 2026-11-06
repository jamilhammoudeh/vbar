"""The README opens with the install and one episode, names each extra and marks what PyPI lacks."""

import ast
import re
import tomllib

from test_docs import ROOT

from vbar import __version__

README = (ROOT / "README.md").read_text(encoding="utf-8")
EXTRAS = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
    "optional-dependencies"
]
CHANGELOG = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
UNRELEASED = CHANGELOG.split("## [Unreleased]", 1)[1].split("\n## [", 1)[0]
DEVELOPMENT = "pip install git+https://github.com/jamilhammoudeh/vbar"
COMMAND_UNRELEASED = "\n- The `vbar` command" in UNRELEASED
PAGES = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md"))]


def release_notes(text):
    """Versions named by the notes "not in the X.Y.Z release", line breaks allowed."""
    return set(re.findall(r"not\s+in\s+the\s+(\d+\.\d+\.\d+)\s+release", text))


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


def test_command_line_section_installs_from_the_repository_until_the_command_is_released():
    # Following `pip install vbar` with `vbar setup` fails on a release without the command.
    title = next(t for _, t in headings() if t.startswith("Try it without writing code"))
    body = section(title)
    assert (DEVELOPMENT in body) == COMMAND_UNRELEASED
    assert bool(re.search(r"^pip install vbar$", body, re.M)) != COMMAND_UNRELEASED
    assert ("(next release)" in title) == COMMAND_UNRELEASED


def test_pages_of_the_command_say_the_release_lacks_it_until_it_is_released():
    for name in ("command-line.md", "setup.md", "evaluation.md", "llm-agents.md"):
        text = (ROOT / "docs" / name).read_text(encoding="utf-8")
        assert (release_notes(text) == {__version__}) == COMMAND_UNRELEASED, name


def test_llm_extra_row_says_the_release_lacks_it_until_it_is_released():
    row = next(line for line in README.splitlines() if line.startswith("| `llm` |"))
    assert bool(release_notes(row)) == ("\n- The `llm` extra" in UNRELEASED)


def test_notes_on_what_the_release_lacks_name_the_current_version():
    # The next release's version bump fails this until the release removes its notes.
    for page in PAGES:
        text = page.read_text(encoding="utf-8")
        assert release_notes(text) <= {__version__}, page.name
        if re.search(r"next\s+release", text):
            assert __version__ in release_notes(text), page.name
