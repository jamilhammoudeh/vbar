"""Files the installed package needs beyond its Python modules."""

import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import pytest

from vbar import flightdata

ROOT = Path(__file__).resolve().parents[1]
EXCERPT = "vbar/flightdata/gracefo_2019-12-03.csv"


def test_the_installed_excerpt_holds_sixteen_hours_of_both_states_and_the_range():
    excerpt = flightdata.gracefo_excerpt()
    assert excerpt["t"].shape == excerpt["range"].shape == (481,)
    assert excerpt["target"].shape == excerpt["chaser"].shape == (481, 6)
    # 2019-12-03 00:00:00 GPS in GPS seconds past 2000-01-01 12:00:00 GPS.
    assert excerpt["t"][0] == 628603200.0
    np.testing.assert_array_equal(np.diff(excerpt["t"]), 120.0)


def test_the_sdist_and_the_wheel_built_from_it_carry_the_excerpt(tmp_path):
    if not (ROOT / "pyproject.toml").is_file():
        pytest.skip("needs the source tree")
    pytest.importorskip("build")
    # Without --sdist or --wheel, build makes the sdist and then the wheel from that sdist.
    subprocess.run(
        [sys.executable, "-m", "build", "--outdir", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
    )
    (sdist,) = tmp_path.glob("vbar-*.tar.gz")
    (wheel,) = tmp_path.glob("vbar-*.whl")
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
    assert any(name.endswith("/src/" + EXCERPT) for name in names)
    with zipfile.ZipFile(wheel) as archive:
        assert EXCERPT in archive.namelist()
