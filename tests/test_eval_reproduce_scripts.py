"""Script entries of a reproduction manifest: figures and tables made outside the suite runner."""

import json
import logging
import sys
import tomllib
from pathlib import Path

import pytest

from vbar.eval.__main__ import main, reproduce

ROOT = Path(__file__).resolve().parent.parent

WRITE_OUT = "from pathlib import Path\nPath('out.txt').write_text('done')\n"
COUNT_AND_WRITE = (
    "from pathlib import Path\n"
    "with open('count.txt', 'a') as f:\n"
    "    f.write('x\\n')\n"
    "Path('out.txt').write_text('done')\n"
)


def _entry(**fields):
    # JSON strings and arrays are valid TOML values.
    lines = ["[[script]]"] + [f"{key} = {json.dumps(value)}" for key, value in fields.items()]
    return "\n".join(lines) + "\n"


def _setup(tmp_path, entries, scripts=None):
    for name, text in (scripts or {"make.py": WRITE_OUT}).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(text, encoding="utf-8")
    manifest = tmp_path / "reproduce.toml"
    manifest.write_text("\n".join(entries), encoding="utf-8")
    return manifest


def _make(**extra):
    return _entry(name="make", script="make.py", outputs=["out.txt"], **extra)


def _resolved(paths):
    return {Path(p).resolve() for p in paths}


def test_a_script_runs_and_its_outputs_are_returned(tmp_path):
    manifest = _setup(tmp_path, [_make()])
    written = reproduce(manifest)
    assert (tmp_path / "out.txt").read_text() == "done"
    assert _resolved(written) == {(tmp_path / "out.txt").resolve()}


def test_a_script_runs_in_the_manifest_folder_with_its_arguments(tmp_path):
    script = (
        "import sys\nfrom pathlib import Path\nPath('out.txt').write_text(' '.join(sys.argv[1:]))\n"
    )
    manifest = _setup(
        tmp_path,
        [_entry(name="a", script="sub/make.py", args=["--n", "3"], outputs=["out.txt"])],
        scripts={"sub/make.py": script},
    )
    reproduce(manifest)
    assert (tmp_path / "out.txt").read_text() == "--n 3"


def test_standard_output_is_written_to_the_stdout_file_and_counts_as_an_output(tmp_path):
    manifest = _setup(
        tmp_path,
        [_entry(name="t", script="make.py", outputs=[], stdout="table.md")],
        scripts={"make.py": "print('| a | b |')\n"},
    )
    written = reproduce(manifest)
    assert "| a | b |" in (tmp_path / "table.md").read_text()
    assert _resolved(written) == {(tmp_path / "table.md").resolve()}


def test_a_missing_stdout_file_makes_the_script_run_again(tmp_path):
    manifest = _setup(
        tmp_path,
        [_entry(name="t", script="make.py", outputs=["out.txt"], stdout="table.md")],
        scripts={"make.py": WRITE_OUT + "print('row')\n"},
    )
    reproduce(manifest)
    (tmp_path / "table.md").unlink()
    reproduce(manifest)
    assert "row" in (tmp_path / "table.md").read_text()


def test_a_script_whose_outputs_exist_is_not_run_again(tmp_path):
    manifest = _setup(tmp_path, [_make()], scripts={"make.py": COUNT_AND_WRITE})
    reproduce(manifest)
    again = reproduce(manifest)
    assert (tmp_path / "count.txt").read_text().count("x") == 1
    assert again == []


def test_force_runs_a_script_whose_outputs_exist(tmp_path):
    manifest = _setup(tmp_path, [_make()], scripts={"make.py": COUNT_AND_WRITE})
    reproduce(manifest)
    written = reproduce(manifest, force=True)
    assert (tmp_path / "count.txt").read_text().count("x") == 2
    assert _resolved(written) == {(tmp_path / "out.txt").resolve()}


def test_a_missing_module_names_the_script_and_the_install_command(tmp_path):
    manifest = _setup(tmp_path, [_make(requires=["vbar_no_such_module"], extra="plot")])
    with pytest.raises(ImportError) as info:
        reproduce(manifest)
    assert 'pip install "vbar[plot]"' in str(info.value)
    assert "make" in str(info.value)
    assert not (tmp_path / "out.txt").exists()


def test_skip_missing_skips_with_a_warning_and_later_scripts_still_run(tmp_path, caplog):
    manifest = _setup(
        tmp_path,
        [
            _entry(
                name="needs_extra",
                script="make.py",
                outputs=["out.txt"],
                requires=["vbar_no_such_module"],
                extra="plot",
            ),
            _entry(name="later", script="later.py", outputs=["later.txt"]),
        ],
        scripts={
            "make.py": WRITE_OUT,
            "later.py": "from pathlib import Path\nPath('later.txt').write_text('ok')\n",
        },
    )
    with caplog.at_level(logging.WARNING):
        written = reproduce(manifest, skip_missing=True)
    assert not (tmp_path / "out.txt").exists()
    assert (tmp_path / "later.txt").read_text() == "ok"
    assert _resolved(written) == {(tmp_path / "later.txt").resolve()}
    assert any(
        r.levelno == logging.WARNING and "needs_extra" in r.getMessage() for r in caplog.records
    )


def test_a_failing_script_raises_with_its_name_and_status_and_leaves_no_stdout_file(tmp_path):
    manifest = _setup(
        tmp_path,
        [_entry(name="broken", script="make.py", outputs=[], stdout="table.md")],
        scripts={"make.py": "import sys\nprint('partial')\nsys.exit(3)\n"},
    )
    with pytest.raises(RuntimeError) as info:
        reproduce(manifest)
    assert "broken" in str(info.value) and "3" in str(info.value)
    assert not (tmp_path / "table.md").exists()


def test_an_output_missing_after_a_successful_run_is_named(tmp_path):
    manifest = _setup(
        tmp_path,
        [_entry(name="lazy", script="make.py", outputs=["out.txt", "fig.png"])],
    )
    with pytest.raises(RuntimeError, match="fig.png"):
        reproduce(manifest)


@pytest.mark.parametrize("field", ["script", "outputs", "stdout"])
@pytest.mark.parametrize("kind", ["parent", "absolute"])
def test_paths_that_escape_the_manifest_folder_are_rejected_before_any_run(tmp_path, field, kind):
    folder = tmp_path / "project"
    folder.mkdir()
    bad = "../elsewhere.txt" if kind == "parent" else (tmp_path / "elsewhere.txt").as_posix()
    fields = {"name": "bad", "script": "make.py", "outputs": ["out.txt"]}
    fields[field] = [bad] if field == "outputs" else bad
    manifest = _setup(folder, [_entry(**fields)])
    with pytest.raises(ValueError):
        reproduce(manifest)
    assert not (folder / "out.txt").exists() and not (tmp_path / "elsewhere.txt").exists()


def test_an_unknown_key_is_rejected_by_name(tmp_path):
    manifest = _setup(tmp_path, [_make(colour="red")])
    with pytest.raises(ValueError, match="colour"):
        reproduce(manifest)


@pytest.mark.parametrize("missing", ["name", "script", "outputs"])
def test_a_missing_required_key_is_rejected_by_name(tmp_path, missing):
    fields = {"name": "make", "script": "make.py", "outputs": ["out.txt"]}
    del fields[missing]
    manifest = _setup(tmp_path, [_entry(**fields)])
    with pytest.raises(ValueError, match=missing):
        reproduce(manifest)


def test_the_whole_manifest_is_validated_before_the_first_script_runs(tmp_path):
    manifest = _setup(
        tmp_path,
        [_make(), _entry(name="invalid", script="make.py", outputs=["out2.txt"], colour="red")],
    )
    with pytest.raises(ValueError, match="colour"):
        reproduce(manifest)
    assert not (tmp_path / "out.txt").exists()


def test_the_command_line_passes_force_through(tmp_path, capsys):
    manifest = _setup(tmp_path, [_make()], scripts={"make.py": COUNT_AND_WRITE})
    assert main(["reproduce", str(manifest)]) == 0
    assert main(["reproduce", str(manifest)]) == 0
    assert (tmp_path / "count.txt").read_text().count("x") == 1
    assert main(["reproduce", str(manifest), "--force"]) == 0
    assert (tmp_path / "count.txt").read_text().count("x") == 2
    assert "out.txt" in capsys.readouterr().out


def test_the_command_line_passes_skip_missing_through(tmp_path):
    manifest = _setup(tmp_path, [_make(requires=["vbar_no_such_module"], extra="plot")])
    with pytest.raises(ImportError):
        main(["reproduce", str(manifest)])
    assert main(["reproduce", str(manifest), "--skip-missing"]) == 0
    assert not (tmp_path / "out.txt").exists()


# The repository's own manifest.


@pytest.fixture(scope="module")
def repo_spec():
    return tomllib.loads((ROOT / "reproduce.toml").read_text(encoding="utf-8"))


def _by_script(spec, path):
    matches = [s for s in spec.get("script", []) if s["script"] == path]
    assert len(matches) == 1, f"expected one [[script]] for {path}"
    return matches[0]


def test_the_readme_figure_is_a_script_entry(repo_spec):
    entry = _by_script(repo_spec, "examples/readme_figures.py")
    assert set(entry["outputs"]) == {"docs/images/pd_approach.png", "docs/images/pd_approach.gif"}
    assert entry["requires"] == ["matplotlib"] and entry["extra"] == "plot"


def test_the_agreement_table_is_a_script_entry(repo_spec):
    entry = _by_script(repo_spec, "benchmarks/agreement_table.py")
    assert entry["stdout"] == "outputs/agreement-table.md"
    assert entry["requires"] == ["Basilisk"] and entry["extra"] == "basilisk"


def test_the_coverage_table_is_a_script_entry(repo_spec):
    entry = _by_script(repo_spec, "benchmarks/coverage_table.py")
    assert entry["stdout"] == "outputs/coverage-table.md"


def test_every_listed_script_exists_in_the_repository(repo_spec):
    scripts = repo_spec.get("script", [])
    assert scripts
    for entry in scripts:
        assert (ROOT / entry["script"]).is_file(), entry["script"]


def test_every_extra_named_by_a_script_is_declared_in_pyproject(repo_spec):
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = set(pyproject["project"]["optional-dependencies"])
    named = {entry["extra"] for entry in repo_spec.get("script", []) if "extra" in entry}
    assert named and named <= declared


def test_no_output_path_is_produced_twice(repo_spec):
    scripts = repo_spec.get("script", [])
    assert scripts
    paths = []
    for table in repo_spec.get("table", []):
        paths += [table["results"], table["markdown"]]
    for entry in scripts:
        paths += entry["outputs"] + ([entry["stdout"]] if "stdout" in entry else [])
    assert len(paths) == len(set(paths))


def test_the_evaluation_page_documents_script_entries():
    assert "[[script]]" in (ROOT / "docs" / "evaluation.md").read_text(encoding="utf-8")


def _valid_first_then(tmp_path, tail):
    first = _entry(name="first", script="make.py", outputs=["out.txt"])
    manifest = _setup(tmp_path, [first, tail], scripts={"make.py": COUNT_AND_WRITE})
    return manifest


def _assert_rejected_without_running(tmp_path, tail, match=None):
    manifest = _valid_first_then(tmp_path, tail)
    with pytest.raises(ValueError, match=match):
        reproduce(manifest)
    assert not (tmp_path / "count.txt").exists()


def _second(**fields):
    base = {"name": "second", "script": "make.py", "outputs": ["second.txt"]}
    return _entry(**{**base, **fields})


@pytest.mark.parametrize(
    "fields",
    [
        {"args": ["--n", 3]},
        {"args": "--n 3"},
        {"requires": "matplotlib"},
        {"outputs": "second.txt"},
    ],
    ids=["args-item-not-text", "args-text", "requires-text", "outputs-text"],
)
def test_fields_of_the_wrong_type_are_rejected_before_any_run(tmp_path, fields):
    _assert_rejected_without_running(tmp_path, _second(**fields))


@pytest.mark.parametrize("bad", ["", "."])
@pytest.mark.parametrize("field", ["script", "outputs", "stdout"])
def test_empty_and_folder_paths_are_rejected_before_any_run(tmp_path, field, bad):
    value = [bad] if field == "outputs" else bad
    _assert_rejected_without_running(tmp_path, _second(**{field: value}))


def test_a_path_listed_twice_in_one_entry_is_rejected_before_any_run(tmp_path):
    tail = _second(outputs=["second.txt", "second.txt"])
    _assert_rejected_without_running(tmp_path, tail, match="second.txt")


def test_an_output_that_is_also_the_stdout_file_is_rejected_before_any_run(tmp_path):
    tail = _second(stdout="second.txt")
    _assert_rejected_without_running(tmp_path, tail, match="second.txt")


def test_a_path_produced_by_two_entries_is_rejected_before_any_run(tmp_path):
    _assert_rejected_without_running(tmp_path, _second(outputs=["out.txt"]), match="out.txt")


def test_an_unknown_top_level_key_is_rejected_by_name_before_any_run(tmp_path):
    tail = '[[scripts]]\nname = "typo"\n'
    _assert_rejected_without_running(tmp_path, tail, match="scripts")


@pytest.mark.skipif(sys.platform != "win32", reason="drive-relative paths exist on Windows")
@pytest.mark.parametrize("field", ["script", "outputs", "stdout"])
def test_drive_relative_paths_are_rejected_before_any_run(tmp_path, field):
    value = ["C:foo.txt"] if field == "outputs" else "C:foo.txt"
    _assert_rejected_without_running(tmp_path, _second(**{field: value}))


@pytest.mark.parametrize(
    "head",
    ['script = "make.py"\n', "table = 3\n", '[script]\nname = "make"\n'],
    ids=["script-text", "table-number", "script-single-table"],
)
def test_a_top_level_entry_that_is_not_an_array_of_tables_is_rejected_before_any_run(
    tmp_path, head
):
    # A [script] key cannot share a file with [[script]] entries, so only the table case
    # carries a valid entry that must not run.
    valid = _entry(name="first", script="make.py", outputs=["out.txt"])
    text = head + valid if head.startswith("table") else head
    manifest = _setup(tmp_path, [text], scripts={"make.py": COUNT_AND_WRITE})
    with pytest.raises(ValueError):
        reproduce(manifest)
    assert not (tmp_path / "count.txt").exists()


def test_a_dotted_module_whose_parent_is_missing_counts_as_missing(tmp_path):
    manifest = _setup(tmp_path, [_make(requires=["nopkg_zz.sub"], extra="plot")])
    with pytest.raises(ImportError) as info:
        reproduce(manifest)
    assert 'pip install "vbar[plot]"' in str(info.value)
    assert not (tmp_path / "out.txt").exists()


def test_skip_missing_skips_a_dotted_module_whose_parent_is_missing(tmp_path, caplog):
    manifest = _setup(tmp_path, [_make(requires=["nopkg_zz.sub"], extra="plot")])
    with caplog.at_level(logging.WARNING):
        written = reproduce(manifest, skip_missing=True)
    assert written == [] and not (tmp_path / "out.txt").exists()
    assert any(r.levelno == logging.WARNING for r in caplog.records)
