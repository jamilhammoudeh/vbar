"""KSPDG's import names for V-BAR's pe1, so a KSPDG agent script runs with its imports unchanged.

`install()` makes `import kspdg`, `from kspdg.agent_api.base_agent import KSPDGBaseAgent`,
`from kspdg.pe1.e3_envs import PE1_E3_I3_Env` and KSPDG's other public module paths resolve, in
this process, to V-BAR's pe1 classes. The environments V-BAR does not rebuild (lady-bandit-guard
lbg1, sun blocking sb1, the obfuscated LG3 to LG7 and the 2022-05-16 pe1 start) still import, so a
script that only names them at the top keeps working, and raise `NotImplementedError` with the
reason when they are built. Any other `kspdg` module raises `ModuleNotFoundError`.

`python -m vbar.games.compat script.py [args]` installs the names and runs the script as
`__main__`, so a KSPDG agent file runs as it is (an agent that talks to the game through kRPC
still needs the game).

Module paths and names follow KSPDG v0.15.0 (github.com/mit-ll/spacegym-kspdg, commit 6251ca3,
`src/kspdg/__init__.py` and its subpackages); nothing of KSPDG is imported or copied.
"""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import importlib.util
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

from .envs import SCENARIOS
from .pursuit import CAPTURE_DIST, EPISODE_TIMEOUT, PursuitEvasionEnv
from .runner import AgentEnvRunner, BaseAgent

_LBG1 = "lady-bandit-guard (lbg1) is not rebuilt in V-BAR; only pursuit-evasion (pe1) is"
_HIDDEN = "LG3 to LG7 use KSPDG's obfuscated guard bots, which V-BAR does not reproduce"
_SB1 = "sun blocking (sb1) is not rebuilt in V-BAR; only pursuit-evasion (pe1) is"


def _missing(name: str, reason: str) -> type:
    """A class standing in for a KSPDG environment V-BAR lacks; building it raises."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(f"{name}: {reason}")

    return type(name, (), {"__init__": __init__, "__doc__": f"Not in V-BAR: {reason}."})


def _pe1(evader: int) -> dict[str, Any]:
    return {
        cls.__name__: cls for key, cls in SCENARIOS.items() if key.startswith(f"pe1-e{evader}-")
    }


def _layout() -> dict[str, dict[str, Any]]:
    """Every module path of the alias and the names it holds."""
    lbg1 = {
        f"lg{g}_envs": {
            f"LBG1_LG{g}_I{i}_Env": _missing(f"LBG1_LG{g}_I{i}_Env", _LBG1) for i in (1, 2)
        }
        for g in (0, 1, 2)
    }
    sb1 = {f"SB1_E1_I{i}_Env": _missing(f"SB1_E1_I{i}_Env", _SB1) for i in range(1, 6)}
    pe1 = {f"e{e}_envs": _pe1(e) for e in (1, 2, 3, 4)}
    pe1["e3_envs"].update(
        PE1_E3_I20220516_Env=_missing(
            "PE1_E3_I20220516_Env", "the 2022-05-16 start is not rebuilt in V-BAR"
        ),
        DEFAULT_EVADE_DIST=200.0,
        DEFAULT_EVADE_THROTTLE=1.0,
    )
    top: dict[str, Any] = {}
    for module in pe1.values():
        for name, cls in module.items():
            if name.endswith("_Env"):
                top[name.removesuffix("_Env") + "_V1"] = cls
    for module in lbg1.values():
        for name, cls in module.items():
            top[name.removesuffix("_Env") + "_V1"] = cls
    for name, cls in sb1.items():
        top[name.removesuffix("_Env") + "_V1"] = cls
    for g in range(3, 8):
        for i in (1, 2):
            top[f"LBG1_LG{g}_I{i}_V1"] = _missing(f"LBG1_LG{g}_I{i}_V1", _HIDDEN)
    layout = {
        "kspdg": top,
        "kspdg.agent_api": {},
        "kspdg.agent_api.base_agent": {"KSPDGBaseAgent": BaseAgent},
        "kspdg.agent_api.runner": {"AgentEnvRunner": AgentEnvRunner},
        "kspdg.base_envs": {"KSPDGBaseEnv": PursuitEvasionEnv, "Group1BaseEnv": PursuitEvasionEnv},
        "kspdg.pe1": {},
        "kspdg.pe1.pe1_base": {
            "PursuitEvadeGroup1Env": PursuitEvasionEnv,
            "DEFAULT_EPISODE_TIMEOUT": EPISODE_TIMEOUT,
            "DEFAULT_CAPTURE_DIST": CAPTURE_DIST,
        },
        "kspdg.lbg1": {},
        "kspdg.lbg1.lbg1_base": {
            "LadyBanditGuardGroup1Env": _missing("LadyBanditGuardGroup1Env", _LBG1)
        },
        "kspdg.sb1": {},
        "kspdg.sb1.sb1_base": {"SunBlockingGroup1Env": _missing("SunBlockingGroup1Env", _SB1)},
        "kspdg.sb1.e1_envs": sb1,
    }
    layout.update({f"kspdg.pe1.{name}": names for name, names in pe1.items()})
    layout.update({f"kspdg.lbg1.{name}": names for name, names in lbg1.items()})
    return layout


_LAYOUT = _layout()
_PACKAGES = {name.rpartition(".")[0] for name in _LAYOUT if "." in name}


class _Loader(importlib.abc.Loader):
    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        name = module.__name__
        module.__dict__.update(_LAYOUT[name])
        module.__dict__["__vbar_alias__"] = True

        def __getattr__(attribute: str) -> Any:
            raise AttributeError(
                f"{name}.{attribute} is not in V-BAR's KSPDG names (vbar.games.compat)"
            )

        module.__dict__["__getattr__"] = __getattr__


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname != "kspdg" and not fullname.startswith("kspdg."):
            return None
        if fullname not in _LAYOUT:
            raise ModuleNotFoundError(
                f"{fullname} is not in V-BAR's KSPDG names (vbar.games.compat)", name=fullname
            )
        return importlib.util.spec_from_loader(
            fullname, _Loader(), is_package=fullname in _PACKAGES
        )


def install() -> None:
    """Make KSPDG's module paths import V-BAR's pe1 in this process.

    Raises
    ------
    RuntimeError
        If a real `kspdg` package is already imported.
    """
    existing = sys.modules.get("kspdg")
    if existing is not None and not getattr(existing, "__vbar_alias__", False):
        raise RuntimeError("the kspdg package itself is already imported in this process")
    if not any(isinstance(finder, _Finder) for finder in sys.meta_path):
        sys.meta_path.insert(0, _Finder())


def uninstall() -> None:
    """Undo `install`: remove the finder and forget the alias modules already imported."""
    sys.meta_path[:] = [finder for finder in sys.meta_path if not isinstance(finder, _Finder)]
    for name in list(sys.modules):
        if (name == "kspdg" or name.startswith("kspdg.")) and getattr(
            sys.modules[name], "__vbar_alias__", False
        ):
            del sys.modules[name]


def main(argv: Sequence[str] | None = None) -> None:
    """Run a KSPDG agent script as `__main__` with KSPDG's names installed.

    Parameters
    ----------
    argv : sequence of str, optional
        The script's path and its arguments; `sys.argv[1:]` by default.
    """
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise SystemExit("usage: python -m vbar.games.compat script.py [args]")
    script = Path(args[0]).resolve()
    if not script.is_file():
        raise SystemExit(f"no such script: {args[0]}")
    install()
    sys.argv = [str(script), *args[1:]]
    # As `python script.py` would, so the script's own sibling imports resolve.
    sys.path.insert(0, str(script.parent))
    runpy.run_path(str(script), run_name="__main__")


__all__ = ["install", "main", "uninstall"]

if __name__ == "__main__":
    # Run as `-m`, this file is a second copy of the module; use the imported one, whose
    # finder `install` and `uninstall` recognise.
    from vbar.games.compat import main as _main

    _main()
