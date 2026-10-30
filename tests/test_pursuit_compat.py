import importlib
import json
import subprocess
import sys
import textwrap
from types import ModuleType

import pytest

from vbar.games import SCENARIOS, AgentEnvRunner, BaseAgent, PursuitEvasionEnv, compat


class Coast(BaseAgent):
    def get_action(self, observation):
        return [0.0, 0.0, 0.0, 1.0]


@pytest.fixture
def alias(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    compat.install()
    yield
    compat.uninstall()


def test_kspdg_module_paths_resolve_to_the_pe1_classes(alias):
    from kspdg.agent_api.base_agent import KSPDGBaseAgent
    from kspdg.agent_api.runner import AgentEnvRunner as Runner
    from kspdg.base_envs import KSPDGBaseEnv
    from kspdg.pe1.pe1_base import DEFAULT_CAPTURE_DIST, PursuitEvadeGroup1Env

    assert KSPDGBaseAgent is BaseAgent and Runner is AgentEnvRunner
    assert KSPDGBaseEnv is PursuitEvadeGroup1Env is PursuitEvasionEnv
    assert DEFAULT_CAPTURE_DIST == 5.0
    kspdg = importlib.import_module("kspdg")
    for key, cls in SCENARIOS.items():
        _, evader, _ = key.split("-")
        module = importlib.import_module(f"kspdg.pe1.{evader}_envs")
        assert getattr(module, cls.__name__) is cls
        assert getattr(kspdg, cls.__name__.removesuffix("_Env") + "_V1") is cls


@pytest.mark.parametrize(
    ("module", "name", "reason"),
    [
        ("kspdg.lbg1.lg1_envs", "LBG1_LG1_I1_Env", "lady-bandit-guard"),
        ("kspdg.lbg1.lg0_envs", "LBG1_LG0_I2_Env", "lady-bandit-guard"),
        ("kspdg.lbg1.lbg1_base", "LadyBanditGuardGroup1Env", "lady-bandit-guard"),
        ("kspdg.sb1.e1_envs", "SB1_E1_I5_Env", "sun blocking"),
        ("kspdg.sb1.sb1_base", "SunBlockingGroup1Env", "sun blocking"),
        ("kspdg", "LBG1_LG3_I1_V1", "obfuscated"),
        ("kspdg", "LBG1_LG7_I2_V1", "obfuscated"),
        ("kspdg", "SB1_E1_I1_V1", "sun blocking"),
        ("kspdg.pe1.e3_envs", "PE1_E3_I20220516_Env", "2022-05-16"),
    ],
)
def test_environments_vbar_lacks_import_and_refuse_to_build(alias, module, name, reason):
    cls = getattr(importlib.import_module(module), name)
    with pytest.raises(NotImplementedError, match=reason):
        cls()
    with pytest.raises(NotImplementedError, match=cls.__name__):
        AgentEnvRunner(Coast(), cls, None).run()


def test_other_kspdg_names_fail_with_a_clear_error(alias):
    with pytest.raises(ModuleNotFoundError, match="kspdg.utils is not in V-BAR"):
        importlib.import_module("kspdg.utils")
    with pytest.raises(AttributeError, match="kspdg.LBG1_LG9_I1_V1 is not in V-BAR's KSPDG names"):
        importlib.import_module("kspdg").LBG1_LG9_I1_V1  # noqa: B018  the access is the test
    with pytest.raises(ImportError, match="LBG1_LG9_I1_V1"):
        exec("from kspdg import LBG1_LG9_I1_V1", {})


def test_install_refuses_when_kspdg_itself_is_imported(monkeypatch):
    monkeypatch.setitem(sys.modules, "kspdg", ModuleType("kspdg"))
    with pytest.raises(RuntimeError, match="already imported"):
        compat.install()


def test_install_twice_adds_one_finder_and_uninstall_removes_it():
    compat.install()
    compat.install()
    try:
        finders = [f for f in sys.meta_path if type(f).__name__ == "_Finder"]
        assert len(finders) == 1
        importlib.import_module("kspdg.pe1")
    finally:
        compat.uninstall()
    assert not [f for f in sys.meta_path if type(f).__name__ == "_Finder"]
    assert not any(name == "kspdg" or name.startswith("kspdg.") for name in sys.modules)


AGENT = """
import json
import sys

from kspdg.agent_api.base_agent import KSPDGBaseAgent
from kspdg.agent_api.runner import AgentEnvRunner
from kspdg.pe1.e1_envs import PE1_E1_I3_Env


class NaivePursuitAgent(KSPDGBaseAgent):
    def get_action(self, observation):
        return {"burn_vec": [1.0, 0, 0, 1.0], "vec_type": 0, "ref_frame": 0}


if __name__ == "__main__":
    runner = AgentEnvRunner(
        agent=NaivePursuitAgent(),
        env_cls=PE1_E1_I3_Env,
        env_kwargs={"episode_timeout": 20.0},
        runner_timeout=100,
        debug=False,
    )
    results = runner.run()["agent_env_results"]
    with open(sys.argv[1], "w") as out:
        json.dump({"results": results, "path": sys.path[0], "argv": sys.argv}, out)
"""


def test_a_kspdg_agent_script_runs_as_written(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.setattr(sys, "argv", list(sys.argv))
    script = tmp_path / "naive_agent.py"
    script.write_text(textwrap.dedent(AGENT))
    out = tmp_path / "results.json"
    try:
        compat.main([str(script), str(out)])
    finally:
        compat.uninstall()
    record = json.loads(out.read_text())
    # Run as `python naive_agent.py results.json` would run it.
    assert record["argv"] == [str(script), str(out)] and record["path"] == str(tmp_path)
    results = record["results"]
    assert results["is_episode_done"] and results["steps"] == 20
    assert results["pursuer_fuel_usage"] == pytest.approx(20 * 8000.0 / (240.0 * 9.80665))


def test_main_without_a_script_prints_its_usage():
    with pytest.raises(SystemExit, match="usage"):
        compat.main([])


RUN_JOBS = """
import json
import sys

from kspdg.pe1.e1_envs import PE1_E1_I3_Env

from vbar.eval import Job, run_jobs
from vbar.games import PursuitTask

if __name__ == "__main__":
    assert PE1_E1_I3_Env.scenario == "pe1-e1-i3"
    jobs = [Job("pe1-e1-i3", "validation", 0, 2**100, 2**101)]
    agent = "vbar.games.pursuers:VelocityMatchingPursuer"
    (result,) = run_jobs(jobs, agent, task=PursuitTask(), workers=1)
    with open(sys.argv[1], "w") as out:
        json.dump(result["metrics"], out)
"""


def test_the_module_entry_point_runs_a_script_in_a_fresh_process(tmp_path):
    script = tmp_path / "jobs.py"
    script.write_text(textwrap.dedent(RUN_JOBS))
    out = tmp_path / "metrics.json"
    completed = subprocess.run(
        [sys.executable, "-m", "vbar.games.compat", str(script), str(out)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    assert json.loads(out.read_text())["success"] is True
