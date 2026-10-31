"""Published KSPDG agents run in V-BAR with only their imports changed.

The agent files of github.com/ARCLab-MIT/kspdg (MIT licence) are downloaded when the tests run,
from a pinned commit, and checked against their SHA-256 before they are run; the repository
stores none of them. Without network access these tests skip. KSPDG's own import names come from
`vbar.games.compat`; the one other import the deterministic agent needs, its poliastro-based
`closest_approach`, is replaced by V-BAR's.
"""

import hashlib
import json
import subprocess
import sys
import textwrap
import urllib.error
import urllib.request

import pytest

from vbar.games import AgentEnvRunner, PE1_E1_I3_Env, PE1_E3_I3_Env, compat
from vbar.games.pursuers import ClosestApproachAgent

COMMIT = "a45e5c6dbae7707fb6d0a0d5860cd78e22978f55"
FILES = {
    "deterministic_agent.py": (
        "arclab_mit/agents/extended_obs_agent/deterministic_agent.py",
        "cd9938ba63ebe13b95927b3a787da9fda9b3e6d9a1a0d8a38d3bd64d7d93e321",
    ),
    "no_action_agent.py": (
        "arclab_mit/agents/no_action_agent.py",
        "d68c2f648abd597080a251ca9c3d5e7ef64a0b581cc84682efadb81bb8c934a5",
    ),
}
# The import changes, each an exact line of the original file and its replacement.
IMPORTS = {
    "deterministic_agent.py": {
        "from arclab_mit.agents.extended_obs_agent.simulate import closest_approach": (
            "from vbar.games.pursuers import closest_approach"
        ),
    },
    "no_action_agent.py": {},
}


@pytest.fixture(scope="module")
def sources():
    texts = {}
    for name, (path, sha256) in FILES.items():
        url = f"https://raw.githubusercontent.com/ARCLab-MIT/kspdg/{COMMIT}/{path}"
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                raw = response.read()
        except (urllib.error.URLError, TimeoutError) as error:
            pytest.skip(f"cannot download {url}: {error}")
        assert hashlib.sha256(raw).hexdigest() == sha256, f"{path} changed at {COMMIT}"
        texts[name] = raw.decode("utf-8")
    return texts


def changed(text, name):
    lines = text.splitlines()
    for old, new in IMPORTS[name].items():
        assert lines.count(old) == 1
        lines[lines.index(old)] = new
    return "\n".join(lines) + "\n"


@pytest.fixture
def agents(sources, tmp_path, monkeypatch):
    """The changed files written out, KSPDG's names installed, and a loader for a module."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    paths = {}
    for name, text in sources.items():
        paths[name] = tmp_path / name
        paths[name].write_text(changed(text, name), encoding="utf-8")
    compat.install()

    def load(name):
        import importlib.util

        spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), paths[name])
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    yield paths, load
    compat.uninstall()


def same_episode(one, other):
    """The same steps and outcome, and every number equal to 1e-9 relative.

    Both agents do the same arithmetic in the same order, but numpy may round a small reduction
    differently with the arrays' memory alignment, and an episode of many steps can carry a
    last-bit difference up to about 1e-12.
    """
    assert one.keys() == other.keys()
    assert one["steps"] == other["steps"] and one["is_episode_done"] == other["is_episode_done"]
    for key in one.keys() - {"wall_s", "steps", "is_episode_done"}:
        assert one[key] == pytest.approx(other[key], rel=1e-9, abs=1e-12), key


def test_only_the_import_lines_change(sources):
    for name, text in sources.items():
        before, after = text.splitlines(), changed(text, name).splitlines()
        edits = [(a, b) for a, b in zip(before, after, strict=True) if a != b]
        assert len(before) == len(after) and len(edits) == len(IMPORTS[name])
        assert all(a.startswith(("from ", "import ")) for a, _ in edits)


def test_the_no_action_agent_coasts_through_the_e3_episode(agents):
    _, load = agents
    module = load("no_action_agent.py")
    results = AgentEnvRunner(module.BaseAgent(), PE1_E3_I3_Env, None, 240).run()
    info = results["agent_env_results"]
    # [0, 0, 0, 1] coasts 1 s at a time; the I3 vessels drift no nearer than 2618 m.
    assert info["steps"] == 240 and info["pursuer_fuel_usage"] == 0.0
    assert info["closest_approach"] == pytest.approx(2617.99, abs=0.01)


def test_the_deterministic_agent_flies_as_its_reimplementation(agents, capsys):
    _, load = agents
    module = load("deterministic_agent.py")
    kwargs = {"episode_timeout": 60.0}
    theirs = AgentEnvRunner(module.DeterministicAgent(), PE1_E1_I3_Env, kwargs).run()
    ours = AgentEnvRunner(ClosestApproachAgent(), PE1_E1_I3_Env, kwargs).run()
    theirs, ours = theirs["agent_env_results"], ours["agent_env_results"]
    same_episode(theirs, ours)
    assert theirs["steps"] == 12
    # The original prints each predicted approach time and state.
    assert capsys.readouterr().out.count("\n") >= 12


RUN = """
import json
import sys

from vbar.games import compat
from vbar.games import runner

done = {}
fly = runner.fly


def recording(*args, **kwargs):
    done.update(fly(*args, **kwargs))
    return done


script, results = sys.argv[1:3]
runner.fly = recording
try:
    compat.main([script])
finally:
    with open(results, "w") as out:
        json.dump({key: value for key, value in done.items() if key != "wall_s"}, out)
"""


@pytest.mark.slow
def test_the_deterministic_agent_file_runs_as_its_own_main(agents, tmp_path):
    """Its `__main__` flies the full 240 s E1 I3 episode, as `python -m vbar.games.compat`."""
    paths, _ = agents
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(RUN), encoding="utf-8")
    out = tmp_path / "results.json"
    completed = subprocess.run(
        [sys.executable, str(driver), str(paths["deterministic_agent.py"]), str(out)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert completed.returncode == 0, completed.stderr
    info = json.loads(out.read_text())
    reference = AgentEnvRunner(ClosestApproachAgent(), PE1_E1_I3_Env, None).run()
    expected = {k: v for k, v in reference["agent_env_results"].items() if k != "wall_s"}
    same_episode(info, expected)
    # It captures the passive evader.
    assert info["closest_approach"] == pytest.approx(5.0, abs=1e-6)
