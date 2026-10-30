"""Without the replay extra, the export names the command that installs it."""

import re
import runpy
import sys
from pathlib import Path

import pytest
from metric_cases import episode

from vbar.agents import PDAgent
from vbar.replay import export_rrd

INSTALL = re.escape('pip install "vbar[replay]"')
DEMO = Path(__file__).resolve().parents[1] / "examples" / "replay_demo.py"


def test_export_without_rerun_names_the_install_command(monkeypatch, tmp_path):
    ep = episode(PDAgent(), start_state=(0.0, 400.0, 0.0, 0.0, 0.0, 0.0), max_steps=3)
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match=f"needs rerun: {INSTALL}"):
        export_rrd(ep, tmp_path / "x.rrd")
    assert not (tmp_path / "x.rrd").exists()


def test_the_missing_extra_is_reported_before_the_arguments_are_checked(monkeypatch, tmp_path):
    ep = episode(PDAgent(), start_state=(0.0, 400.0, 0.0, 0.0, 0.0, 0.0), max_steps=3)
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match=INSTALL):
        export_rrd(ep, tmp_path / "x.rrd", arrow_m=-1.0)


def test_importing_the_module_does_not_need_rerun(monkeypatch):
    monkeypatch.setitem(sys.modules, "rerun", None)
    monkeypatch.delitem(sys.modules, "vbar.replay")
    import vbar.replay  # noqa: F401


def test_demo_without_rerun_names_the_install_command(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "rerun", None)
    with pytest.raises(ImportError, match=f"needs rerun: {INSTALL}"):
        runpy.run_path(str(DEMO))
    assert not list(tmp_path.rglob("*.rrd"))
