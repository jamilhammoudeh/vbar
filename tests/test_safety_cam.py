"""CAM-recoverable barrier and the backup filter that enforces it.

The interface these tests assume:

- `CamBarrier(*, radius, n, dt, accel, steps, grid=5.0, t_cap=None, samples=20)` in
  `vbar.safety.barriers` takes keyword arguments only and keeps each as an attribute of the same
  name. `accel` is the collision avoidance manoeuvre a_c, m/s^2, Hill frame, held for `steps`
  whole steps of `dt` s. Construction raises ValueError when `radius`, `n` or `dt` is not
  positive, `steps` is not a positive integer or `accel` is not three finite numbers.
- `value(x)` for one state of shape (6,) is h = max over j = 0..steps of h^j, where
  h^0 = PassiveBarrier(radius=radius, n=n, grid=grid, t_cap=t_cap, terminal=True).value and
  h^j(x) = min(held_segment_bound(x, a_c, dt, samples, radius, n), h^(j-1)(Phi x + Gamma a_c)),
  with (Phi, Gamma) = cw_discrete(n, dt). `contains(x)` is h(x) >= 0.
- `backup(x, actuator)` returns zero thrust where h^0(x) >= 0 and a_c elsewhere.
- `BackupFilter(barrier="cam")` is named "cbf-cam". Its reset declares one barrier, a CamBarrier
  with the view's keep-out radius, mean motion and dt, a grid that divides dt, a_c of magnitude
  a_max along the direction `vbar.metrics.passive.cam_impulses` gives for the view's goal, and
  steps = ceil(0.3 m/s / (a_max dt)). The certificate is "nominal" with horizon inf. Reset raises
  ValueError when a navigation or execution error set is declared.
"""

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from vbar.agents.base import action_scale
from vbar.constraints import KeepOut
from vbar.dynamics import cw_discrete
from vbar.safety import FilteredAgent
from vbar.safety.barriers import CamBarrier, PassiveBarrier
from vbar.safety.cbf import BACKUP_BARRIERS, BackupFilter
from vbar.safety.sampled import held_segment_bound
from vbar.safety.tightening import BoxError
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.solvers.qp import ActuatorSet
from vbar.spacecraft import PerfectNav

R = 50.0
A_MAX = 0.02
DT = 10.0
N = ApproachEnv().n
PHI, GAMMA = cw_discrete(N, DT)
BOX = ActuatorSet(a_max=A_MAX, layout="box")
# The approach-vbar hold point is on -V-bar, so its CAM is along -y; ceil(0.3 / (0.02 * 10)) = 2.
CAM = np.array([0.0, -A_MAX, 0.0])
CAM_STEPS = 2
# A violation is a certified in-step bound deeper than the keep-out's resolution.
DEPTH = KeepOut.resolution
T_CAP = 20.0 * math.pi / N

# Closing on the keep-out at 0.3 m/s from 10 m: free drift enters it, the two-step CAM stops
# 2.25 m short of it and leaves.
CLOSING = np.array([0.0, -60.0, 0.0, 0.0, 0.3, 0.0])
REST_FAR = np.array([0.0, 400.0, 0.0, 0.0, 0.0, 0.0])
# Found by search against the definition: free drift grazes the keep-out by 16.5 mm within the
# first step; the two-step CAM clears it by 64 mm, and the CAM measured from the drift states
# instead of the burn states admits it with the wrong successor.
GRAZE = np.array([21.908749, -44.927831, 3.078069, 0.201502, 0.147051, 0.296639])
# Found by search: the first CAM step dips 0.15 m into the keep-out but ends on a safe drift.
DIP = np.array([-42.774595, -18.717885, 18.073731, 0.074537, -0.022097, -0.010586])
# x_c = 0.5 m and vy = -1.5 n x_c: the drift closes along V-bar at 0.75 n m/s and enters the
# keep-out 2 s after the ten-orbit cap; a CAM toward the target from here enters it at once.
SLOW = np.array(
    [0.5, math.sqrt(R * R - 0.25) + 1.5 * N * 0.5 * (T_CAP + 2.0), 0.0, 0.0, -0.75 * N, 0.0]
)

# Exact samples of a held step, for the in-step check.
TAU = np.linspace(0.0, DT, 201)
PHIS, GAMMAS = cw_discrete(N, TAU)


def cam_barrier(**kw):
    args = {"radius": R, "n": N, "dt": DT, "accel": tuple(CAM), "steps": CAM_STEPS} | kw
    return CamBarrier(**args)


class Defined:
    """The barrier of the contract built from its parts; each flag off is one mutant of it."""

    def __init__(self, *, gamma=True, held=True, terminal=True):
        self.drift = PassiveBarrier(radius=R, n=N, grid=5.0, terminal=terminal)
        self.push = GAMMA @ CAM if gamma else np.zeros(6)
        self.held = held

    def levels(self, x):
        levels = [float(self.drift.value(x))]
        burn, state = math.inf, np.asarray(x, dtype=float)
        for _ in range(CAM_STEPS):
            if self.held:
                burn = min(burn, held_segment_bound(state, CAM, DT, 20, R, N))
            state = PHI @ state + self.push
            levels.append(min(burn, float(self.drift.value(state))))
        return levels

    def value(self, x):
        return max(self.levels(x))

    def backup(self, x, actuator):
        return np.zeros(3) if self.drift.value(x) >= 0.0 else CAM.copy()


def step_failures(barrier, states, count=None):
    """Check the one-step invariance of a CAM barrier on the members among `states`.

    For each state x with h(x) >= 0, up to `count` of them, the backup step x+ = Phi x + Gamma
    beta(x) must have h(x+) >= 0 and the exact held step must stay outside the keep-out. Returns
    the number of members checked, how many of them took the CAM, and the members that failed.
    """
    members, cams, failed = 0, 0, []
    for x in states:
        if count is not None and members == count:
            break
        if barrier.value(x) < 0.0:
            continue
        members += 1
        u = np.asarray(barrier.backup(x, BOX), dtype=float)
        cams += bool(np.any(u))
        x1 = PHI @ x + GAMMA @ u
        held = PHIS @ x + GAMMAS @ u
        # The barrier bounds the held step from below, so its exact samples clear the keep-out
        # up to rounding.
        clear = np.linalg.norm(held[:, :3], axis=1).min() >= R - 1e-9
        if barrier.value(x1) < 0.0 or not clear:
            failed.append(x)
    return members, cams, failed


# Construction.


def test_cam_barrier_takes_keyword_arguments_only():
    with pytest.raises(TypeError):
        CamBarrier(R, N, DT, tuple(CAM), CAM_STEPS)


@pytest.mark.parametrize(
    "kw",
    [
        {"steps": 0},
        {"steps": 1.5},
        {"accel": (0.0, math.nan, 0.0)},
        {"accel": (0.0, -A_MAX)},
        {"dt": 0.0},
        {"radius": -1.0},
        {"n": 0.0},
    ],
)
def test_cam_barrier_refuses_malformed_parameters(kw):
    with pytest.raises(ValueError):
        cam_barrier(**kw)


def test_cam_barrier_keeps_its_parameters():
    barrier = cam_barrier()
    assert barrier.radius == R and barrier.n == N and barrier.dt == DT
    assert barrier.steps == CAM_STEPS and barrier.grid == 5.0 and barrier.samples == 20
    assert np.array_equal(np.asarray(barrier.accel, dtype=float), CAM)


# Known answers: a 1 ms burn of 300 m/s^2 is the 0.3 m/s impulse of the book cases to within
# 0.15 mm of position.

N_400 = ApproachEnv(ApproachConfig(altitude_m=400e3)).n
BURN = 1e-3
# A keep-out around every start, so the drift margin is negative and the backup is the CAM.
K_RADIUS = 150.0


def cam_arc(barrier, x0):
    """Hill state t s after the start: the held backup for one burn, then free drift."""
    phi, gamma = cw_discrete(barrier.n, BURN)
    x1 = phi @ x0 + gamma @ barrier.backup(x0, ActuatorSet(a_max=300.0, layout="box"))
    return lambda t: cw_discrete(barrier.n, t - BURN)[0] @ x1


def test_minus_vbar_cam_passes_396_m_under_the_target_and_ends_4978_m_ahead():
    # Fehse, Automated Rendezvous and Docking of Spacecraft, sec. 4.5, Fig. 4.16; review K25
    # with the corrected crossing at 400 km, r = 6778.137 km (ERRATA.md, Fehse part 1 entry):
    # x = -396.220 m at y = 0, y(T) = +4978.3 m one orbit after the burn.
    barrier = cam_barrier(radius=K_RADIUS, n=N_400, dt=BURN, accel=(0.0, -300.0, 0.0), steps=1)
    arc = cam_arc(barrier, np.array([0.0, -20.0, 0.0, 0.0, 0.0, 0.0]))
    period = 2.0 * math.pi / N_400
    crossing = brentq(lambda t: arc(t)[1], BURN, 0.5 * period, xtol=1e-9)
    assert arc(crossing)[0] == pytest.approx(-396.220, abs=0.01)
    assert arc(period)[1] == pytest.approx(4978.3, abs=0.1)


@pytest.mark.parametrize(("d", "y_end"), [(20.0, 753.98), (50.0, 1884.96), (100.0, 3769.91)])
def test_radial_cam_from_rbar_drifts_12_pi_d_ahead_in_one_orbit(d, y_end):
    # Fehse sec. 4.5, Figs. 4.17 to 4.18; review K23: x0 = -d, dvx = -0.3 m/s gives
    # y(T) = 12 pi d whatever the impulse.
    barrier = cam_barrier(radius=K_RADIUS, n=N_400, dt=BURN, accel=(-300.0, 0.0, 0.0), steps=1)
    arc = cam_arc(barrier, np.array([-d, 0.0, 0.0, 0.0, 0.0, 0.0]))
    assert arc(2.0 * math.pi / N_400)[1] == pytest.approx(y_end, abs=0.05)


# The value and the backup.


def test_value_is_the_largest_cam_then_drift_margin_of_the_definition():
    barrier, defined = cam_barrier(), Defined()
    for x in (CLOSING, REST_FAR, GRAZE, DIP, SLOW):
        assert barrier.value(x) == pytest.approx(defined.value(x), abs=1e-9)
        assert barrier.contains(x) == (barrier.value(x) >= 0.0)


def test_a_state_only_the_cam_saves_is_in_the_set_and_its_backup_is_the_cam():
    barrier = cam_barrier()
    assert PassiveBarrier(radius=R, n=N, grid=5.0, terminal=True).value(CLOSING) < 0.0
    assert barrier.contains(CLOSING)
    assert np.array_equal(np.asarray(barrier.backup(CLOSING, BOX), dtype=float), CAM)


def test_backup_is_zero_thrust_where_free_drift_is_safe():
    beta = np.asarray(cam_barrier().backup(REST_FAR, BOX), dtype=float)
    assert np.array_equal(beta, np.zeros(3))


def test_the_cam_set_contains_the_free_drift_set():
    barrier = cam_barrier()
    drift = PassiveBarrier(radius=R, n=N, grid=5.0, terminal=True)
    for x in (CLOSING, REST_FAR, GRAZE, DIP, SLOW):
        assert barrier.value(x) >= float(drift.value(x))


# One-step invariance on 200 seeded members: 8 tests of 25, so each stays short.

HOLD_SIDE = ((-30.0, -150.0, -20.0, -0.1, 0.0, -0.1), (30.0, -55.0, 20.0, 0.1, 0.6, 0.1))
NEAR_FAST = ((-80.0, -150.0, -30.0, -0.3, -0.3, -0.3), (80.0, 150.0, 30.0, 0.3, 0.3, 0.3))
CHUNKS = 8
PER_CHUNK = 25


def chunk_states(k):
    """Seeded draws for chunk k: closing on the keep-out from the hold side, or near it."""
    lo, hi = HOLD_SIDE if k % 2 == 0 else NEAR_FAST
    return np.random.default_rng([1142, k]).uniform(lo, hi, (150, 6))


@pytest.mark.parametrize("k", range(CHUNKS))
def test_backup_step_keeps_members_in_the_set_and_out_of_the_keep_out(k):
    members, cams, failed = step_failures(cam_barrier(), chunk_states(k), PER_CHUNK)
    assert members == PER_CHUNK
    assert failed == []
    if k % 2 == 0:
        # About half the hold-side members need the CAM; at least a fifth keeps the check on it.
        assert cams >= 5


def test_the_definition_passes_the_invariance_check_on_the_planted_states():
    members, cams, failed = step_failures(Defined(), (CLOSING, REST_FAR, GRAZE))
    assert (members, cams, failed) == (3, 2, [])


@pytest.mark.parametrize(
    ("mutant", "planted"),
    [
        ({"gamma": False}, GRAZE),
        ({"held": False}, DIP),
        ({"terminal": False}, SLOW),
    ],
    ids=["successor-without-gamma", "burn-bound-dropped", "drift-margin-not-terminal"],
)
def test_the_invariance_check_kills_each_mutant_of_the_definition(mutant, planted):
    members, _, failed = step_failures(Defined(**mutant), (planted,))
    assert members == 1 and len(failed) == 1


# The backup filter with the cam barrier.


def approach(dt=DT, steps=1, start=REST_FAR, **cfg):
    start = tuple(float(s) for s in start)
    return ApproachEnv(ApproachConfig(dt=dt, max_steps=steps, start_state=start, **cfg))


def ready(filt, dt=DT, **cfg):
    filt.reset(approach(dt, **cfg).agent_view())
    return filt


def declared(dt=DT, **cfg):
    barriers = ready(BackupFilter(barrier="cam"), dt, **cfg).declaration.barriers
    assert len(barriers) == 1 and isinstance(barriers[0], CamBarrier)
    return barriers[0]


def test_cam_is_a_backup_barrier_named_cbf_cam():
    assert "cam" in BACKUP_BARRIERS
    assert BackupFilter(barrier="cam").name == "cbf-cam"


def test_cam_filter_declares_the_vbar_cam_and_a_nominal_certificate():
    filt = ready(BackupFilter(barrier="cam"))
    d = filt.declaration
    assert d.certificate == "nominal" and d.counts_toward_certification is True
    barrier = declared()
    assert barrier.radius == R and barrier.n == N and barrier.dt == DT
    assert barrier.steps == CAM_STEPS
    assert np.asarray(barrier.accel, dtype=float) == pytest.approx(CAM, abs=1e-15)
    assert DT / barrier.grid == round(DT / barrier.grid)


@pytest.mark.parametrize(("dt", "steps"), [(1.0, 15), (5.0, 3), (10.0, 2), (20.0, 1)])
def test_cam_lasts_the_fewest_whole_steps_giving_0_3_m_s(dt, steps):
    # ceil(0.3 / (0.02 dt)): the 0.3 m/s CAM size of the spec, an assumption tied to S8.
    barrier = declared(dt)
    assert barrier.steps == steps and barrier.dt == dt
    assert dt / barrier.grid == round(dt / barrier.grid)


def test_cam_under_a_ball_limit_is_along_minus_vbar_at_full_thrust():
    barrier = declared(thrust_limit="ball")
    assert np.asarray(barrier.accel, dtype=float) == pytest.approx(CAM, abs=1e-15)


def test_cam_for_an_rbar_hold_point_is_radially_away_with_a_fifth_retrograde():
    # cam_impulses (metrics/passive.py): radially away with a retrograde part of one fifth,
    # Fehse secs. 4.5 and 5.7, scaled to the thrust limit.
    barrier = declared(hold_point=(-100.0, 0.0, 0.0))
    away = A_MAX * np.array([-5.0, -1.0, 0.0]) / math.sqrt(26.0)
    assert np.asarray(barrier.accel, dtype=float) == pytest.approx(away, rel=1e-12)
    assert barrier.steps == CAM_STEPS


@pytest.mark.parametrize(
    "kw", [{"execution": 2e-4}, {"navigation": BoxError(r_pos=1.0, r_vel=0.01)}]
)
def test_cam_filter_makes_no_claim_under_error_sets(kw):
    with pytest.raises(ValueError):
        ready(BackupFilter(barrier="cam", **kw))


def test_a_state_only_the_cam_saves_is_filtered_with_a_certificate_not_as_recovery():
    view = approach().agent_view()
    filt = BackupFilter(barrier="cam")
    filt.reset(view)
    assert filt.contains(CLOSING)
    inward = -CLOSING[:3] / np.linalg.norm(CLOSING[:3]) * action_scale(view) / A_MAX
    result = filt.filter(CLOSING.copy(), inward)
    assert result.status == "feasible" and result.recovering is False
    assert result.certificate == "nominal" and result.certificate_horizon == math.inf
    u = np.asarray(result.action, dtype=float) * A_MAX / action_scale(view)
    assert cam_barrier().value(PHI @ CLOSING + GAMMA @ u) >= 0.0
    held = PHIS @ CLOSING + GAMMAS @ u
    assert np.linalg.norm(held[:, :3], axis=1).min() >= R - 1e-9


def test_the_backup_returned_inside_the_set_is_certified():
    filt = ready(BackupFilter(barrier="cam", iterations=0))
    result = filt.filter(CLOSING.copy(), np.array([0.0, 1.0, 0.0]))
    assert result.fallback_level == 2
    assert result.status == "feasible" and result.certificate == "nominal"
    assert np.asarray(result.action, dtype=float) == pytest.approx(CAM / A_MAX, abs=1e-12)


# Closed loop on approach-vbar, KAT-8 setup (Ames et al. review, ames_cbf_survey.md): PD
# nominal, dt 10 s, box 0.02 m/s^2. benchmarks run the check on 40 starts; the test runs one.


class HoldPD:
    """PD to the scenario's hold point, clipped to the box."""

    name = "hold-pd"

    def reset(self, view, rng=None):
        self.hold = np.asarray(view.goal["hold_point"], dtype=float)
        self.scale = action_scale(view) / view.max_accel
        self.a_max = view.max_accel

    def act(self, obs, info):
        u = np.clip(-2e-4 * (obs[:3] - self.hold) - 0.03 * obs[3:6], -self.a_max, self.a_max)
        return u * self.scale


def cam_starts(count):
    """The first `count` approach-vbar starts, by seed, inside the cam filter's set."""
    filt = ready(BackupFilter(barrier="cam"))
    env = ApproachEnv()
    starts, seed = [], 0
    while len(starts) < count:
        _, info = env.reset(seed=seed)
        if filt.contains(info["true_state"]):
            starts.append(info["true_state"])
        seed += 1
    return starts


def check_cam_filter_keeps_approach_starts_out(count, steps=360):
    """Fly the PD nominal through the cam filter from `count` approach-vbar starts.

    Returns, for each start, the lowest certified keep-out bound of the run (m) and the
    statuses the filter reported.
    """
    flights = []
    for x0 in cam_starts(count):
        env = approach(DT, steps, x0, sensor=PerfectNav())
        pilot = FilteredAgent(HoldPD(), BackupFilter(barrier="cam"))
        obs, _ = env.reset(seed=0)
        pilot.reset(env.agent_view(), np.random.default_rng(0))
        worst, statuses = math.inf, set()
        for _ in range(steps):
            obs, _, terminated, truncated, _ = env.step(pilot.act(obs, env.agent_info()))
            worst = min(worst, env.record.segment.minima["keep_out"].lower_bound)
            statuses.add(pilot.last_result.status)
            if terminated or truncated:
                break
        flights.append((worst, statuses))
    return flights


def test_cam_filter_keeps_an_approach_vbar_start_out_of_the_keep_out():
    for worst, statuses in check_cam_filter_keeps_approach_starts_out(1):
        assert worst >= -DEPTH
        assert statuses == {"feasible"}
