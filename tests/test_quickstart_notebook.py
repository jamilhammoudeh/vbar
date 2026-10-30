import ast
import json
import socket
from pathlib import Path

import pytest

from vbar.scenarios import SUITES

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "examples" / "quickstart.ipynb"
SUITE_ID = "quickstart-v1"
INSTALL_LINE = "%pip install -q vbar"
LAYOUT = ["markdown", "code", "markdown", "code", "markdown", "code", "markdown"]


def load_notebook() -> dict:
    assert NOTEBOOK.is_file(), f"{NOTEBOOK} is missing"
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def source_of(cell: dict) -> str:
    # nbformat stores a cell's source either as one string or as a list of lines.
    source = cell["source"]
    return source if isinstance(source, str) else "".join(source)


def code_sources() -> list[str]:
    return [source_of(cell) for cell in load_notebook()["cells"] if cell["cell_type"] == "code"]


def magic_lines(source: str) -> list[str]:
    return [line for line in source.splitlines() if line.lstrip().startswith(("%", "!"))]


def test_notebook_is_nbformat_4_json():
    notebook = load_notebook()
    assert notebook["nbformat"] == 4
    assert isinstance(notebook["cells"], list)


def test_cells_follow_the_documented_order():
    # The headless run and the install-cell checks pick cells by position.
    assert [cell["cell_type"] for cell in load_notebook()["cells"]] == LAYOUT


def test_notebook_is_committed_without_outputs():
    for number, cell in enumerate(load_notebook()["cells"], start=1):
        if cell["cell_type"] != "code":
            continue
        assert cell["outputs"] == [], f"cell {number} holds outputs"
        assert cell["execution_count"] is None, f"cell {number} holds an execution count"


def test_first_code_cell_installs_the_unpinned_released_package():
    # Colab starts without the package; a pin would stop the notebook following the release.
    first = code_sources()[0]
    assert first.splitlines() == [INSTALL_LINE]


def test_only_the_install_cell_uses_a_magic():
    cells = code_sources()
    assert len(magic_lines(cells[0])) == 1
    for number, source in enumerate(cells[1:], start=2):
        assert magic_lines(source) == [], f"code cell {number} uses a magic"


def test_every_code_cell_after_the_install_cell_compiles():
    for number, source in enumerate(code_sources()[1:], start=2):
        ast.parse(source, filename=f"<quickstart cell {number}>")


@pytest.fixture
def fresh_suite_registry(monkeypatch):
    # setitem records whether the identifier was present, so teardown restores either state;
    # deleting it here lets the notebook register the suite from scratch on every run.
    monkeypatch.setitem(SUITES, SUITE_ID, None)
    del SUITES[SUITE_ID]


@pytest.fixture
def no_remote_connections(monkeypatch):
    # Colab needs the network only for the install cell, which this test skips. Loopback stays
    # open because the interpreter's own socket pairs use it.
    connect = socket.socket.connect

    def local_only(sock, address, *args):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"remote connection attempted: {address!r}")
        return connect(sock, address, *args)

    monkeypatch.setattr(socket.socket, "connect", local_only)


def test_quickstart_runs_headless_and_a_second_run_flies_nothing(
    tmp_path, monkeypatch, capsys, fresh_suite_registry, no_remote_connections
):
    monkeypatch.chdir(tmp_path)
    cells = code_sources()[1:]
    namespace: dict = {}
    for number, source in enumerate(cells, start=2):
        exec(compile(source, f"<quickstart cell {number}>", "exec"), namespace)

    assert namespace["metrics"]["success"] is True
    printed = capsys.readouterr().out.splitlines()
    assert any(line.startswith("success: True") for line in printed)
    # The README quotes about 620 s of flight for this start state.
    assert "time_to_complete: 620.0" in printed
    for key in ("delta_v", "terminal_pos_error", "min_margin"):
        assert any(line.startswith(f"{key}: ") for line in printed), key

    table = (tmp_path / "quickstart.md").read_text(encoding="utf-8")
    assert table.startswith("# quickstart-v1: pd")
    assert "2 runs" in table
    assert any(line.startswith("| approach-rbar |") for line in table.splitlines())
    assert table.strip() in "\n".join(printed)

    results = tmp_path / "quickstart.jsonl"
    flown = results.read_text(encoding="utf-8")
    assert len(flown.splitlines()) == 2

    # The closing cell says running the table cell again flies nothing: both runs are on file.
    (table_cell,) = [source for source in cells if "reproduce" in source]
    exec(compile(table_cell, "<quickstart table cell again>", "exec"), namespace)
    assert results.read_text(encoding="utf-8") == flown
