"""Cheap tasks and agents for the harness tests, importable by worker processes."""

from dataclasses import dataclass

from vbar.agents import PDAgent
from vbar.eval import SpaceTask
from vbar.scenarios import make


@dataclass(frozen=True)
class ShortTask(SpaceTask):
    """The catalogue scenarios cut to a few steps, so a test flies whole suites in seconds."""

    name: str = "space-short"
    steps: int = 4

    def make(self, scenario):
        return make(scenario, max_steps=self.steps)


class Crashes(PDAgent):
    def act(self, obs, info):
        raise RuntimeError("controller fault")
