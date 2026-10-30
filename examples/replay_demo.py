"""Replay of the PD baseline on the approach scenario, written for the Rerun viewer.

The target lies between the start and the hold point, so the controller flies through the
keep-out zone: the file shows the violation as an event and in the metric card. Needs the replay
extra (pip install "vbar[replay]"). Open the result with: rerun outputs/pd_approach.rrd
"""

from pathlib import Path

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.metrics import evaluate, record_episode
from vbar.replay import export_rrd

out = Path("outputs")
out.mkdir(exist_ok=True)

env = ApproachEnv(ApproachConfig(start_state=(0, 400, 0, 0, 0, 0)))
episode = record_episode(env, PDAgent(), seed=1, agent_seed=2)
card = evaluate(episode)
path = export_rrd(episode, out / "pd_approach.rrd", card=card)

print(
    f"{len(episode.records)} steps, {episode.duration:.0f} s, "
    f"keep-out margin {card['min_margin']['keep_out']:.1f} m, failure label {card['failure_label']}"
)
print(f"wrote {path}; open it with: rerun {path}")
