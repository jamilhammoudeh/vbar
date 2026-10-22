"""Evidence across the classical baselines: coast predictions against continuous-time truth,
tracking a nominal trajectory against re-targeting, a smoke run of every baseline in every action
mode it supports, and the sign-flip mutant on each feedback law.

Every number is for a 500 km circular orbit of the WGS 84 Earth on exact CW motion with a 10 s
step; reference numbers come from an independent computation flown in the same environment.
"""

import math

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from vbar import ApproachConfig, ApproachEnv
from vbar.agents.base import agent_generator
from vbar.agents.cw_targeting import TwoImpulseAgent
from vbar.agents.elements import (
    ElementAgent,
    FehseSequenceAgent,
    NominalTrackingAgent,
    radial_transfer,
)
from vbar.agents.glideslope import GlideslopeAgent
from vbar.agents.lambert import LambertAgent
from vbar.agents.lqg import LQGAgent
from vbar.agents.lqr import LQRAgent
from vbar.agents.lyapunov import CWTrackingAgent, LyapunovAgent
from vbar.agents.mutants import Mutant
from vbar.agents.shuttle import ElevationTimedAgent, PilotAgent
from vbar.agents.sign_check import DIRECTIONS, sign_check
from vbar.metrics import MetricSettings, passive_metrics, record_episode
from vbar.spacecraft import GaussianNav

KEEP_OUT = 50.0  # m, the approach scenario's keep-out radius
HOLD = np.array([0.0, -100.0, 0.0])


def hop_to_hold(view, state):
    # Fehse's radial transfer along the V-bar from the observed start to the hold point.
    return radial_transfer(view.n, view.goal["hold_point"][1] - state[1])


def flown(agent, seed, **config):
    env = ApproachEnv(ApproachConfig(**config))
    return record_episode(env, agent, seed=seed, agent_seed=seed + 1000)


def cw_positions(n, x0, t):
    """Closed-form CW positions (Hill frame, m) at times `t` (s) from the state `x0`."""
    x, y, z, vx, vy, vz = x0
    s, c, nt = np.sin(n * t), np.cos(n * t), n * t
    return np.stack(
        [
            (4.0 - 3.0 * c) * x + s / n * vx + 2.0 * (1.0 - c) / n * vy,
            6.0 * (s - nt) * x + y - 2.0 * (1.0 - c) / n * vx + (4.0 * s - 3.0 * nt) / n * vy,
            c * z + s / n * vz,
        ],
        axis=-1,
    )


def coast_minimum(n, x0, coast_s, h=0.5):
    """Keep-out margin minimum over a free CW coast (m): 0.5 s samples refined by Brent's method.

    Independent of the package's propagator and of its sample-and-refine rule.
    """
    t = np.linspace(0.0, coast_s, math.ceil(coast_s / h) + 1)
    rho = np.linalg.norm(cw_positions(n, x0, t), axis=1)
    k = int(np.argmin(rho))
    found = minimize_scalar(
        lambda tau: float(np.linalg.norm(cw_positions(n, x0, np.array(tau)))),
        bounds=(t[max(k - 1, 0)], t[min(k + 1, len(t) - 1)]),
        method="bounded",
        options={"xatol": 1e-6},
    )
    return min(float(rho[k]), float(found.fun)) - KEEP_OUT


def committed_coasts(ep):
    """(predicted margin, its bound, true continuous minimum) of each coast the agent logged."""
    found = []
    for entry, record in zip(ep.agent_log, ep.records, strict=True):
        if entry.get("predicted_margin") is None:
            continue
        # In "impulse_coast" the coast is the record's second segment; in the continuous modes
        # it starts where the burn step ends.
        segment = record.segment
        x0 = segment.starts[1] if len(segment.segments) > 1 else record.true_state
        truth = coast_minimum(ep.view.n, np.asarray(x0, dtype=float), entry["coast_s"])
        found.append((entry["predicted_margin"], entry["predicted_margin_bound"], truth))
    return found


BEHIND = {
    "start_pos_range": ((-30.0, 30.0), (-400.0, -250.0), (-20.0, 20.0)),
    "start_vel_range": ((0.0, 0.0),) * 3,
}
ON_VBAR = {
    "start_pos_range": ((0.0, 0.0), (-700.0, -400.0), (0.0, 0.0)),
    "start_vel_range": ((0.0, 0.0),) * 3,
}

# Seeds per agent from an independent 30-seed sweep. On the default random starts seed 0 is
# predicted clear and seed 2 predicts a violation for the two-impulse and Lambert plans, so both
# outcomes are flown; the glideslope, the radial hop, the Fehse sequence and the pilot are
# predicted clear on every seed of their starts. One or two seeds keep each case between 0.2 and
# 1 s on a loaded machine (the Fehse sequence is allowed 900 steps and captures in about 400; the
# pilot, allowed 200 steps of which each impulse step commits a 60 s read, ends in about 30).
PREDICTIVE = {
    "two-impulse-open-loop": (
        lambda: TwoImpulseAgent(mode="open_loop"),
        {"action_mode": "impulse_coast", "max_steps": 100},
        (0, 2),
    ),
    "two-impulse-track": (
        lambda: TwoImpulseAgent(mode="track"),
        {"action_mode": "accel", "max_steps": 100},
        (0, 2),
    ),
    "glideslope-open-loop": (
        lambda: GlideslopeAgent(mode="open_loop"),
        {"action_mode": "impulse_coast", **BEHIND},
        (0, 1),
    ),
    "glideslope-retarget": (
        lambda: GlideslopeAgent(mode="retarget"),
        {"action_mode": "impulse_coast", **BEHIND},
        (0, 1),
    ),
    "radial-hop": (
        lambda: ElementAgent(hop_to_hold),
        {"action_mode": "impulse_coast", **ON_VBAR},
        (0, 1),
    ),
    "fehse-sequence": (
        lambda: FehseSequenceAgent(),
        {"action_mode": "accel", "max_steps": 900, **ON_VBAR},
        (0,),
    ),
    "pilot": (
        lambda: PilotAgent(),
        {"action_mode": "impulse_coast", "max_steps": 200},
        (0,),
    ),
    "lambert-open-loop": (
        lambda: LambertAgent(mode="open_loop"),
        {"action_mode": "impulse_coast", "max_steps": 100},
        (0, 2),
    ),
}


@pytest.mark.parametrize("key", PREDICTIVE)
def test_a_plan_predicted_clear_of_the_keep_out_zone_stays_clear_in_continuous_time(key):
    make, config, seeds = PREDICTIVE[key]
    clear = 0
    for seed in seeds:
        ep = flown(make(), seed, **config)
        coasts = committed_coasts(ep)
        assert coasts
        for predicted, bound, truth in coasts:
            # The bound is certified on the continuous coast. Where the minimum is a sample it
            # can exceed the truth by rounding (1.9e-13 m in a 30-seed sweep), hence 1e-9 m.
            assert bound <= truth + 1e-9
            # The second-order bound skips refinement once every interval is within the 1 mm
            # keep-out resolution of the best sample, so the prediction is good to 1 mm (at most
            # 2.9e-4 m off over 30 seeds of the two-impulse and Lambert coasts).
            assert abs(predicted - truth) < 1e-3
        if min(bound for _, bound, _ in coasts) >= 0.0:
            clear += 1
            assert min(r.segment.minima["keep_out"].margin for r in ep.records) >= 0.0
    assert clear >= 1


SLACK = {"two-impulse-open-loop": 2.3e-4, "radial-hop": 1e-9}


@pytest.mark.parametrize("key", SLACK)
def test_the_certified_bound_is_no_looser_than_the_reference_sweep_found(key):
    # Truth minus the certified bound, over 30 seeds: 0.05 to 0.19 mm on the 850 to 880 s
    # two-impulse coasts (an independent computation), and within 1.9e-13 m of zero on the
    # radial hop, whose minimum is a sample; the two-impulse limit adds 20 %, the radial hop's
    # allows rounding.
    # A looser bound would turn safe plans into predicted violations.
    make, config, seeds = PREDICTIVE[key]
    ep = flown(make(), seeds[0], **config)
    slack = [truth - bound for _, bound, truth in committed_coasts(ep)]
    assert slack
    assert max(slack) <= SLACK[key]


def s4_and_s8(mode, seed):
    """Passive safety margin S4 and CAM margin S8 (m) of the two-impulse agent in `mode`."""
    ep = flown(
        TwoImpulseAgent(mode=mode),
        seed,
        start_state=(0.0, -300.0, 0.0, 0.0, 0.0, 0.0),
        action_mode="accel",
        max_steps=100,
        sensor=GaussianNav(1.0, 1e-3),
    )
    # No sub-step burn points: they feed only the stuck-thruster margin, and S4 and S8 come out
    # the same (checked on seeds 0 to 4 of both modes) at a fraction of the cost.
    scored = passive_metrics(ep, MetricSettings(partial_burn_samples=0))
    return scored["passive_safety_margin"], scored["cam_margin"]


def test_tracking_the_nominal_keeps_passive_safety_that_re_targeting_loses():
    # Fehse, Automated Rendezvous and Docking of Spacecraft, section 4.4.1: re-targeting leaves
    # the deviations from the nominal uncontrolled, so the passive-safety analysis of the nominal
    # no longer holds. On the standard approach from 300 m behind with GaussianNav(1 m, 1 mm/s),
    # an independent computation over 40 seeds gives median S4 of 30.6 m tracked and -36.3 m
    # re-targeted, tracking at least as good on 34 seeds. The claim holds for S4 and not for S8
    # in this scenario: median S8 is 51.33 m re-targeted and 50.57 m tracked over the same 40
    # seeds, since re-targeting ends 0.8 m further out (51.42 against 50.57 m without noise).
    # Seeds 0 to 2 give median S4 of 22.4 m tracked and -47.1 m re-targeted, and median S8 of
    # 50.34 m tracked and 51.69 m re-targeted. The six episodes with their S4 and S8 drifts take
    # about 2.8 s together: three seeds per mode are the fewest with a median, and one seed would
    # be a single draw from a distribution where re-targeting wins S4 on 6 of 40 seeds.
    found = {mode: [s4_and_s8(mode, seed) for seed in range(3)] for mode in ("track", "retarget")}
    s4 = {mode: np.median([m[0] for m in rows]) for mode, rows in found.items()}
    s8 = {mode: np.median([m[1] for m in rows]) for mode, rows in found.items()}
    assert s4["track"] > 0.0 > s4["retarget"]
    assert s4["track"] - s4["retarget"] > 50.0
    assert s8["retarget"] > s8["track"]


CONTINUOUS = ("normalized", "accel", "thrust")
ALL_MODES = (*CONTINUOUS, "impulse_coast")
BASELINES = {
    "two-impulse-retarget": (lambda: TwoImpulseAgent(mode="retarget"), ALL_MODES),
    "two-impulse-open-loop": (lambda: TwoImpulseAgent(mode="open_loop"), ALL_MODES),
    "two-impulse-track": (lambda: TwoImpulseAgent(mode="track"), CONTINUOUS),
    "glideslope-open-loop": (lambda: GlideslopeAgent(mode="open_loop"), ALL_MODES),
    "glideslope-retarget": (lambda: GlideslopeAgent(mode="retarget"), ALL_MODES),
    "radial-hop": (lambda: ElementAgent(hop_to_hold), ALL_MODES),
    "fehse-sequence": (lambda: FehseSequenceAgent(), CONTINUOUS),
    "nominal-tracking": (
        lambda: NominalTrackingAgent(TwoImpulseAgent(mode="open_loop")),
        CONTINUOUS,
    ),
    "elevation-timed": (lambda: ElevationTimedAgent(trigger=math.radians(42.0)), ALL_MODES),
    "pilot": (lambda: PilotAgent(), ALL_MODES),
    "lambert-retarget": (lambda: LambertAgent(mode="retarget"), ALL_MODES),
    "lambert-open-loop": (lambda: LambertAgent(mode="open_loop"), ALL_MODES),
    "lqr": (lambda: LQRAgent(), CONTINUOUS),
    "lqg": (lambda: LQGAgent(), CONTINUOUS),
    "cw-tracking": (lambda: CWTrackingAgent(), CONTINUOUS),
    "lyapunov": (lambda: LyapunovAgent(), CONTINUOUS),
}


class Spy:
    """Passes everything to the agent and records the info keys it was handed."""

    def __init__(self, agent):
        self.agent, self.name, self.info_keys = agent, agent.name, set()

    def reset(self, view, rng=None):
        self.agent.reset(view, rng)

    def act(self, obs, info):
        self.info_keys |= set(info)
        return self.agent.act(obs, info)


@pytest.mark.parametrize(
    ("key", "mode"), [(key, mode) for key, (_, modes) in BASELINES.items() for mode in modes]
)
def test_every_baseline_resets_on_the_public_view_and_acts_on_the_time_alone(key, mode):
    make, _ = BASELINES[key]
    # The Fehse sequence refuses an episode shorter than its plan.
    max_steps = 900 if key == "fehse-sequence" else 5
    env = ApproachEnv(
        ApproachConfig(
            action_mode=mode, start_state=(0.0, -600.0, 0.0, 0.0, 0.0, 0.0), max_steps=max_steps
        )
    )
    obs, _ = env.reset(seed=0)
    spy = Spy(make())
    spy.reset(env.agent_view(), agent_generator(1))
    action = np.asarray(spy.act(obs, env.agent_info()), dtype=float)
    assert action.shape == ((4,) if mode == "impulse_coast" else (3,))
    assert np.all(np.isfinite(action))
    env.step(action)
    assert spy.info_keys == {"time_s"}


FEEDBACK = {
    "lqr": LQRAgent,
    "lqg": LQGAgent,
    "cw-tracking": CWTrackingAgent,
    "two-impulse-retarget": lambda: TwoImpulseAgent(mode="retarget"),
    "glideslope-retarget": lambda: GlideslopeAgent(mode="retarget"),
}


@pytest.mark.parametrize("key", FEEDBACK)
def test_the_sign_flip_mutant_of_each_baseline_fails_the_sign_check_on_every_axis(key):
    # Each baseline passes from rest on the V-bar in the default continuous mode, so negating
    # its command must fail all six directions.
    make = FEEDBACK[key]
    assert sign_check(make(), ApproachConfig()).passed
    mutant = Mutant(make(), "sign_flip", 1.0)
    assert mutant.name.startswith(make().name)
    check = sign_check(mutant, ApproachConfig())
    assert set(check.failures) == {label for label, _, _ in DIRECTIONS}
