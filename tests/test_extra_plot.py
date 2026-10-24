"""Without the plot extra, plotting and the demo name the command that installs it."""

import re
import runpy
import sys
from pathlib import Path

import pytest

from vbar import ApproachEnv
from vbar.viz import plot_trajectories

INSTALL = re.escape('pip install "vbar[plot]"')
DEMO = Path(__file__).resolve().parents[1] / "examples" / "v0_demo.py"


def test_plot_without_matplotlib_names_the_install_command(monkeypatch):
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ModuleNotFoundError, match=f"needs matplotlib.*: {INSTALL}"):
        plot_trajectories([], ApproachEnv())


def test_demo_without_matplotlib_names_the_install_command(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    with pytest.raises(ModuleNotFoundError, match=f"needs matplotlib: {INSTALL}"):
        runpy.run_path(str(DEMO))
