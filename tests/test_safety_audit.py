import collections
import functools
import math
from types import SimpleNamespace

import numpy as np
import pytest

from vbar.agents.base import action_scale
from vbar.constraints import KeepOut, SegmentMinimum
from vbar.safety import FilterResult, InfeasibilityProof
from vbar.safety import audit as audit_module
from vbar.safety.audit import FilterAudit, FlaggedDecision, RunAudit, audit_filter
from vbar.safety.cbf import BackupFilter, CbfFilter, SoftCbfFilter
from vbar.scenarios import base as base_module
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.spacecraft import ExecutionError, GaussianNav

# A fixed 128-bit root seed, so the drawn starts and seeds are the same on every run.
SEED = 0xC3A5C85C97CB3127B4A3E1A53C1F2D0B
# Hill states (m, m/s) from just outside the 50 m keep-out to 150 m, where the filters act.
NEAR = ((-80.0, 52.0, -30.0, -0.1, -0.1, -0.1), (80.0, 150.0, 30.0, 0.1, 0.1, 0.1))
STARTS = 20
# Each decision near the boundary costs up to 0.15 s, so the test flights are short.
STEPS = 3


class Reckless:
    """Full command straight at the target."""

    name = "reckless"

    def reset(self, view, rng=None):
        self.scale = action_scale(view)

    def act(self, obs, info):
        return -obs[:3] / np.linalg.norm(obs[:3]) * self.scale


class PassThrough(CbfFilter):
    """Declares the safe set of `CbfFilter` and passes every command through."""

    def filter(self, obs, action):
        return FilterResult(np.array(action, dtype=float), intervened=False)


FILTERS = {
    # The pole rule allows 0.06 1/s at dt 10 s; v_cap 0.1 m/s keeps the safe set nonempty.
    "cbf-limit": lambda: CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1),
    "cbf": lambda: CbfFilter(),
    "passive": lambda: BackupFilter(barrier="passive"),
    "braking": lambda: BackupFilter(barrier="braking"),
}
NOMINALS = {"pd": lambda: None, "reckless": Reckless}


@functools.cache
def report(filter_name, nominal_name):
    filt = FILTERS[filter_name]()
    found = audit_filter(
        filt,
        ApproachEnv(),
        nominal=NOMINALS[nominal_name](),
        starts=STARTS,
        box=NEAR,
        seed=SEED,
        max_steps=STEPS,
    )
    return filt, found


@pytest.mark.parametrize("nominal_name", NOMINALS)
@pytest.mark.parametrize("filter_name", FILTERS)
def test_certified_filters_have_no_violation_or_uncertified_step(filter_name, nominal_name):
    _, found = report(filter_name, nominal_name)
    assert len(found.runs) == STARTS
    assert found.steps == STARTS * STEPS
    assert found.violations == 0
    assert found.uncertified == 0
    assert found.infeasible == 0
    assert found.passed
    assert all(math.isfinite(found.worst_bound["keep_out"]) for _ in found.runs)
    assert found.worst_bound["keep_out"] >= -KeepOut.resolution


@pytest.mark.parametrize("filter_name", FILTERS)
def test_starts_are_drawn_inside_the_declared_safe_set(filter_name):
    filt, found = report(filter_name, "pd")
    lo, hi = (np.array(corner) for corner in NEAR)
    for run in found.runs:
        assert filt.contains(run.start)
        assert np.all(lo <= run.start) and np.all(run.start <= hi)
    assert found.drawn == STARTS + found.rejected
    assert found.mode == "inside"
    assert all(run.reentry_s is None for run in found.runs)


def test_starts_follow_the_root_seed():
    _, first = report("cbf", "pd")
    _, again = report("cbf", "reckless")
    np.testing.assert_array_equal(
        np.array([r.start for r in first.runs]), np.array([r.start for r in again.runs])
    )
    assert [r.seed for r in first.runs] == [r.seed for r in again.runs]


def test_seeds_are_128_bit_and_agent_seeds_differ():
    _, found = report("cbf", "pd")
    assert found.seed == SEED
    seeds = [r.seed for r in found.runs] + [r.agent_seed for r in found.runs]
    assert all(0 <= s < 1 << 128 for s in seeds)
    # Of 40 uniform 128-bit draws, the largest has fewer than 124 bits with probability 2^-160.
    assert max(s.bit_length() for s in seeds) >= 124
    assert len(set(seeds)) == len(seeds)
    assert all(r.agent_seed != r.seed for r in found.runs)


def test_a_root_seed_is_drawn_when_none_is_given():
    found = audit_filter(CbfFilter(), ApproachEnv(), starts=1, max_steps=1)
    # A uniform 128-bit draw has fewer than 65 bits with probability 2^-63.
    assert found.seed.bit_length() > 64
    assert found.runs[0].seed != found.runs[0].agent_seed


@pytest.mark.parametrize("seed", [-1, 1 << 128, True])
def test_a_root_seed_outside_128_bits_is_refused(seed):
    with pytest.raises(ValueError, match="seed"):
        audit_filter(CbfFilter(), ApproachEnv(), starts=1, max_steps=1, seed=seed)


def test_a_filter_that_passes_the_nominal_through_is_caught():
    found = audit_filter(
        PassThrough(), ApproachEnv(), nominal=Reckless(), starts=5, box=NEAR, seed=SEED,
        max_steps=15,
    )  # fmt: skip
    assert found.violations > 0
    assert not found.passed
    assert found.worst_bound["keep_out"] < -KeepOut.resolution


def coarse_braking():
    # Two samples per step leave the certified bound up to 0.4 m below the true minimum while
    # the braking filter rides the boundary at full thrust.
    env = ApproachEnv(ApproachConfig(violation_samples=2))
    box = ((-1.0, 50.2, -1.0, 0.1, -0.01, -0.01), (1.0, 50.6, 1.0, 0.2, 0.0, 0.01))
    return audit_filter(
        BackupFilter(barrier="braking"), env, nominal=Reckless(), starts=3, box=box, seed=SEED,
        max_steps=15,
    )  # fmt: skip


def test_a_coarse_segment_is_split_until_it_certifies():
    found = coarse_braking()
    refinements = [r for run in found.runs for r in run.refinements]
    factors = collections.Counter(r.factor for r in refinements)
    # Measured: 33 steps certify at 4 times the samples and 3 at 16 times.
    assert found.refined == len(refinements) > 0
    assert set(factors) <= set(audit_module.REFINEMENTS)
    assert factors[16] > 0
    assert all(r.outcome == "certified" for r in refinements)
    for r in refinements:
        assert r.bound < -KeepOut.resolution <= r.margin
        assert r.refined_bound >= -KeepOut.resolution > r.bound
    assert found.uncertified == found.violations == 0
    assert found.passed


def test_a_split_that_runs_out_leaves_the_step_uncertified(monkeypatch):
    monkeypatch.setattr(audit_module, "REFINEMENTS", (4,))
    found = coarse_braking()
    refinements = [r for run in found.runs for r in run.refinements]
    assert any(r.outcome == "uncertified" for r in refinements)
    assert found.uncertified > 0
    assert found.violations == 0
    assert not found.passed


def test_the_refly_at_the_environment_samples_reproduces_its_bound():
    env = ApproachEnv(ApproachConfig(start_state=(0.0, 55.0, 0.0, 0.05, -0.05, 0.0)))
    env.reset(seed=0)
    keep_out = env.constraints[0]
    for command in ([0.0, -1.0, 0.0], [0.3, 0.2, -0.5], [0.0, 0.0, 0.0]):
        env.step(np.array(command))
        found = env.record.segment.minima["keep_out"]
        bound, margin = audit_module._refly(keep_out, env.record, env.n, 10.0, 20, 1)
        # The closed form row by row and the stacked matrices differ only by rounding.
        assert bound == pytest.approx(found.lower_bound, rel=1e-9, abs=1e-9)
        assert margin == pytest.approx(found.margin, rel=1e-9, abs=1e-9)


def test_the_audit_refuses_a_soft_filter():
    soft = SoftCbfFilter(alpha=1.0, poles=(0.05, 0.05), v_cap=0.1, gamma=0.02)
    with pytest.raises(ValueError, match="certification"):
        audit_filter(soft, ApproachEnv(), starts=1, max_steps=1)


@pytest.mark.parametrize("dynamics", ["twobody", "twobody_j2"])
def test_the_audit_refuses_dynamics_other_than_cw(dynamics):
    env = ApproachEnv(ApproachConfig(dynamics=dynamics))
    with pytest.raises(ValueError, match="cw"):
        audit_filter(CbfFilter(model_error=1e-6), env, starts=1, max_steps=1)


def test_the_audit_refuses_imperfect_navigation_and_execution_error():
    noisy = ApproachEnv(ApproachConfig(sensor=GaussianNav()))
    with pytest.raises(ValueError, match="navigation"):
        audit_filter(CbfFilter(), noisy, starts=1, max_steps=1)
    pushed = ApproachEnv(ApproachConfig(execution=ExecutionError(noise_sigma=1e-4)))
    with pytest.raises(ValueError, match="execution"):
        audit_filter(CbfFilter(), pushed, starts=1, max_steps=1)


def test_the_audit_refuses_a_filter_without_a_declared_set():
    with pytest.raises(TypeError, match="safe set"):
        audit_filter(SimpleNamespace(name="bare"), ApproachEnv(), starts=1, max_steps=1)


def test_a_box_too_small_for_the_asked_side_is_refused():
    # Every state of this box lies inside the keep-out, so none is in the safe set.
    box = ((-1.0, -1.0, -1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="0 of 2 starts inside"):
        audit_filter(CbfFilter(), ApproachEnv(), starts=2, box=box, max_steps=1)


@pytest.mark.parametrize("bad", [{"starts": 0}, {"max_steps": 0}, {"workers": 0}, {"mode": "on"}])
def test_out_of_range_parameters_are_refused(bad):
    with pytest.raises(ValueError):
        audit_filter(CbfFilter(), ApproachEnv(), **{"starts": 1, "max_steps": 1, **bad})


@functools.cache
def outside():
    # Closing at 0.3 to 0.5 m/s from 55 to 70 m: above the speed cap, and the barrier row
    # exceeds the box on some steps.
    box = ((-5.0, 55.0, -5.0, -0.05, -0.5, -0.05), (5.0, 70.0, 5.0, 0.05, -0.3, 0.05))
    filt = CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1)
    found = audit_filter(
        filt, ApproachEnv(), mode="outside", starts=5, box=box, seed=SEED, max_steps=20
    )
    return filt, found


def test_starts_outside_the_set_report_the_time_to_re_enter():
    filt, found = outside()
    assert found.mode == "outside"
    for run in found.runs:
        assert not filt.contains(run.start)
        assert 0.0 < run.reentry_s < math.inf
        # Re-entry is judged at decision times, whole steps of 10 s.
        assert run.reentry_s % 10.0 == 0.0
    assert found.reentered == len(found.runs)
    assert found.longest_reentry_s == max(r.reentry_s for r in found.runs)


def test_infeasible_steps_outside_the_set_are_reported_with_their_proof_check():
    _, found = outside()
    flagged = [d for run in found.runs for d in run.flagged]
    assert found.infeasible > 0
    for d in flagged:
        assert d.status in ("infeasible_certified", "solver_failed")
        assert d.recovering is True and d.inside is False
        assert d.conflicting
    infeasible = [d for d in flagged if d.status == "infeasible_certified"]
    verified = sum(d.proof_verified is True for d in infeasible)
    assert found.proofless + found.proofs_rejected + verified == found.infeasible
    assert found.passed == (
        found.solver_failed == 0 and found.proofless == 0 and found.proofs_rejected == 0
    )


class Proof:
    def __init__(self, holds):
        self.holds = holds

    def verify(self):
        return self.holds


def flagged_from(results):
    """`_flagged` on a stand-in episode whose decisions return `results`."""
    states = [np.full(6, float(k)) for k in range(len(results) + 1)]
    records = [
        SimpleNamespace(t=10.0 * (k + 1), true_state=states[k + 1], filter_result=r)
        for k, r in enumerate(results)
    ]
    episode = SimpleNamespace(start=states[0], records=records)
    return audit_module._flagged(episode, lambda x: x[0] >= 2.0)


def test_each_proof_is_checked_and_a_missing_proof_is_recorded_as_none():
    def result(status, **extra):
        return SimpleNamespace(status=status, conflicting=("keep_out",), recovering=True, **extra)

    flagged = flagged_from(
        [
            result("infeasible_certified", proof=Proof(True)),
            result("feasible"),
            result("infeasible_certified", proof=Proof(False)),
            result("infeasible_certified"),
            result("solver_failed"),
        ]
    )
    assert [d.step for d in flagged] == [0, 2, 3, 4]
    assert [d.t for d in flagged] == [0.0, 20.0, 30.0, 40.0]
    assert [d.proof_verified for d in flagged] == [True, False, None, None]
    assert [d.inside for d in flagged] == [False, True, True, True]


def outside_audit(*decisions):
    run = RunAudit(
        start=np.zeros(6), seed=1, agent_seed=2, steps=len(decisions), violations=0,
        uncertified=0, refinements=(), flagged=decisions, worst_bound={}, reentry_s=10.0,
    )  # fmt: skip
    return FilterAudit(
        filter_name="f", nominal_name="pd", mode="outside", declaration=None, seed=3,
        box=((0.0,) * 6, (1.0,) * 6), drawn=1, rejected=0, runs=(run,),
    )  # fmt: skip


def decision(status, verified):
    return FlaggedDecision(0, 0.0, status, ("keep_out",), True, False, verified)


def test_outside_the_set_only_verified_proofs_and_no_solver_failure_pass():
    assert outside_audit().passed
    assert outside_audit(decision("infeasible_certified", True)).passed
    assert not outside_audit(decision("infeasible_certified", None)).passed
    assert not outside_audit(decision("infeasible_certified", False)).passed
    assert not outside_audit(decision("solver_failed", None)).passed
    found = outside_audit(
        decision("infeasible_certified", None), decision("infeasible_certified", False)
    )
    assert (found.proofless, found.proofs_rejected, found.infeasible) == (1, 1, 2)


def test_parallel_runs_give_the_same_report():
    kwargs = {"nominal": Reckless(), "starts": 2, "box": NEAR, "seed": SEED, "max_steps": 2}
    serial = audit_filter(CbfFilter(), ApproachEnv(), **kwargs)
    parallel = audit_filter(CbfFilter(), ApproachEnv(), workers=2, **kwargs)
    for a, b in zip(serial.runs, parallel.runs, strict=True):
        np.testing.assert_array_equal(a.start, b.start)
        assert (a.seed, a.agent_seed, a.steps) == (b.seed, b.agent_seed, b.steps)
        assert dict(a.worst_bound) == dict(b.worst_bound)


class FailsOnce(CbfFilter):
    """`CbfFilter` with one "solver_failed" decision, the second of every episode."""

    def reset(self, view):
        super().reset(view)
        self.k = 0

    def filter(self, obs, action):
        result = super().filter(obs, action)
        self.k += 1
        if self.k == 2:
            return FilterResult(result.action, result.intervened, status="solver_failed")
        return result


def test_one_solver_failure_inside_the_set_fails_the_audit_and_is_counted():
    found = audit_filter(
        FailsOnce(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1), ApproachEnv(), starts=2, box=NEAR,
        seed=SEED, max_steps=3,
    )  # fmt: skip
    assert [run.solver_failed for run in found.runs] == [1, 1]
    assert found.solver_failed == 2
    assert not found.passed


def tiny_audit(env, seed=SEED):
    return audit_filter(CbfFilter(), env, starts=1, box=NEAR, seed=seed, max_steps=1)


def test_the_audit_accepts_no_delay_no_minimum_impulse_and_the_accel_drive():
    env = ApproachEnv(ApproachConfig(delay_s=0.0, min_impulse=0.0, drive="accel"))
    assert len(tiny_audit(env).runs) == 1


@pytest.mark.parametrize(
    "config, message",
    [
        ({"delay_s": 1e-9}, "delay"),
        ({"min_impulse": 1e-12}, "impulse"),
        # The thrust drive needs a propellant load below the mass.
        ({"drive": "thrust", "propellant_kg": 100.0}, "drive"),
    ],
)
def test_the_audit_refuses_any_delay_minimum_impulse_or_thrust_drive(config, message):
    # Each changes the delivered command without the filter knowing, so its claim is void.
    with pytest.raises(ValueError, match=message):
        tiny_audit(ApproachEnv(ApproachConfig(**config)))


def test_the_root_seed_starts_at_two_to_the_64():
    # Below 2**64 a seed can be found by trying candidates.
    with pytest.raises(ValueError, match="seed"):
        tiny_audit(ApproachEnv(), seed=(1 << 64) - 1)
    assert tiny_audit(ApproachEnv(), seed=1 << 64).seed == 1 << 64


def test_a_step_with_no_refinement_left_stays_uncertified(monkeypatch):
    # With no finer sampling to try, a step whose bound misses is reported, not dropped.
    monkeypatch.setattr(audit_module, "REFINEMENTS", ())
    found = coarse_braking()
    assert found.uncertified > 0
    assert not found.passed


class CertifiesOnce(CbfFilter):
    """`CbfFilter` that reports its second decision of every episode as proven infeasible."""

    def reset(self, view):
        super().reset(view)
        self.k = 0

    def filter(self, obs, action):
        result = super().filter(obs, action)
        self.k += 1
        if self.k != 2:
            return result
        proof = InfeasibilityProof(
            normals=np.array([[1.0, 0.0, 0.0]]),
            bounds=np.array([0.03]),
            multipliers=np.array([1.0]),
            layout="box",
            a_max=0.02,
            names=("keep_out",),
        )
        return FilterResult(
            result.action, True, status="infeasible_certified", conflicting=("keep_out",),
            proof=proof,
        )  # fmt: skip


def test_one_certified_infeasible_step_inside_the_set_fails_the_audit():
    # Inside the declared set the filter claims every step feasible; a proof is no excuse there.
    found = audit_filter(
        CertifiesOnce(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1), ApproachEnv(), starts=2,
        box=NEAR, seed=SEED, max_steps=3,
    )  # fmt: skip
    assert found.infeasible == 2
    assert found.violations == found.uncertified == found.solver_failed == 0
    assert all(d.proof_verified for run in found.runs for d in run.flagged)
    assert not found.passed


def test_a_violation_the_environment_missed_is_found_by_the_re_fly(monkeypatch):
    # The environment's refinement is made to miss every dip below the resolution: it reports
    # the margin at zero with an uncertified bound, as a minimiser caught on a shallower dip
    # would. Only the audit's own re-fly can then find the violation.
    real = base_module.segment_minimum

    def missed(*args, **kwargs):
        found = real(*args, **kwargs)
        if found.margin >= -KeepOut.resolution:
            return found
        return SegmentMinimum(found.t, 0.0, min(found.lower_bound, -1.0))

    monkeypatch.setattr(base_module, "segment_minimum", missed)
    found = audit_filter(
        PassThrough(), ApproachEnv(), nominal=Reckless(), starts=5, box=NEAR, seed=SEED,
        max_steps=15,
    )  # fmt: skip
    refinements = [r for run in found.runs for r in run.refinements]
    assert any(r.outcome == "violation" for r in refinements)
    assert all(r.margin >= -KeepOut.resolution for r in refinements)
    assert found.violations > 0
    assert not found.passed


def test_the_re_fly_at_factor_one_flies_the_environments_own_samples(monkeypatch):
    env = ApproachEnv(ApproachConfig(start_state=(0.0, 55.0, 0.0, 0.05, -0.05, 0.0)))
    env.reset(seed=0)
    env.step(np.array([0.3, -1.0, 0.2]))
    trace = env.record.segment
    flown = []
    real = audit_module.cw_states

    def recording(n, tau, starts, accels):
        states = real(n, tau, starts, accels)
        if len(tau) > 1:
            flown.append((np.array(tau), states))
        return states

    monkeypatch.setattr(audit_module, "cw_states", recording)
    audit_module._refly(env.constraints[0], env.record, env.n, 10.0, 20, 1)
    assert len(flown) == len(trace.segments)
    pairs = zip(trace.segments, flown, strict=True)
    times = np.concatenate([t0 + tau for (t0, _, _), (tau, _) in pairs])
    states = np.concatenate([s for _, s in flown])
    assert times.shape == trace.times.shape
    np.testing.assert_allclose(times, trace.times, rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(states, trace.states, rtol=1e-12, atol=1e-12)


def falsify(*args, **kwargs):
    # Imported here so the rest of this file runs while the falsifier is being built.
    from vbar.safety.audit import falsify_filter

    return falsify_filter(*args, **kwargs)


# Hill states (m, m/s) up to 1 m outside the 50 m keep-out: full thrust toward the target
# crosses it within one 10 s step, so one-step episodes show a violation.
CLOSE = ((-8.0, 50.2, -1.0, -0.01, -0.01, -0.01), (8.0, 51.0, 1.0, 0.01, 0.01, 0.01))
# A slab 0.32 m wide in the start's radial position, 2% of the CLOSE box's 16 m.
SLAB = (1.0, 1.32)


class PassesSlab:
    """From a start in the slab, passes every command through."""

    def reset(self, view):
        super().reset(view)
        self.open = None

    def filter(self, obs, action):
        if self.open is None:
            self.open = SLAB[0] <= float(obs[0]) <= SLAB[1]
        if self.open:
            return FilterResult(np.array(action, dtype=float), intervened=False)
        return super().filter(obs, action)


class SlabCbf(PassesSlab, CbfFilter):
    """`CbfFilter` with a slab of starts it passes through."""


class SlabBraking(PassesSlab, BackupFilter):
    """The braking `BackupFilter` with a slab of starts it passes through."""


PLANTED = {
    "cbf": lambda: SlabCbf(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1),
    # Three bisection steps instead of twelve still end the line search on an accepted
    # command, and halve the cost of a run.
    "braking": lambda: SlabBraking(barrier="braking", iterations=3),
}


@functools.cache
def planted_search(filter_name):
    filt = PLANTED[filter_name]()
    found = falsify(
        filt, ApproachEnv(), nominal=Reckless(), box=CLOSE, budget=200, seed=SEED, max_steps=1
    )
    return filt, found


@pytest.mark.parametrize("filter_name", PLANTED)
def test_the_falsifier_finds_the_planted_slab_and_reports_the_start_and_trajectory(filter_name):
    filt, found = planted_search(filter_name)
    assert found.verdict == "violation"
    assert found.evaluations <= 200
    start = np.asarray(found.start)
    assert SLAB[0] <= start[0] <= SLAB[1]
    assert filt.contains(start)
    states = np.asarray(found.states)
    assert states.ndim == 2 and states.shape[1] == 6
    np.testing.assert_allclose(states[0], start, rtol=0, atol=1e-12)
    # The reported run is the violating one, and the score is the certified bound over the
    # resolution, so a violation scores below -1.
    assert found.run.violations > 0
    assert found.score < -1.0


def test_draws_outside_the_declared_set_are_counted_and_never_flown():
    flown = []

    class Recorded(ApproachEnv):
        def __init__(self, config=None):
            super().__init__(config)
            if self.cfg.start_state is not None:
                flown.append(np.array(self.cfg.start_state))

    filt = CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1)
    found = falsify(filt, Recorded(), box=NEAR, budget=20, population=10, seed=SEED, max_steps=1)
    assert found.rejected > 0
    # Every run of the search, then the re-fly of the worst start.
    assert len(flown) == found.evaluations + 1
    assert all(filt.contains(x) for x in flown)
    assert found.evaluations > 0 and found.seed == SEED


@functools.cache
def clean_search(seed=SEED):
    found = falsify(
        CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1), ApproachEnv(), nominal=Reckless(),
        box=NEAR, budget=30, seed=seed, max_steps=3,
    )  # fmt: skip
    return found


def test_a_short_search_on_a_certified_filter_finds_nothing_and_claims_nothing():
    found = clean_search()
    # "none_found" is the best verdict there is: a search never shows a filter safe.
    assert found.verdict == "none_found"
    assert found.score >= -1.0
    assert math.isfinite(found.score)
    assert not any("safe" in name for name in vars(found))


def test_a_short_search_on_a_correct_braking_filter_finds_nothing_and_claims_nothing():
    found = falsify(
        BackupFilter(barrier="braking"), ApproachEnv(), nominal=Reckless(), box=CLOSE, budget=30,
        seed=SEED, max_steps=3,
    )  # fmt: skip
    assert found.verdict == "none_found"
    assert found.run.violations == found.run.uncertified == 0
    assert found.score >= -1.0
    assert math.isfinite(found.score)
    assert not any("safe" in name for name in vars(found))


class RecordedBraking(BackupFilter):
    """The braking `BackupFilter`, keeping every result it returns."""

    def __init__(self):
        super().__init__(barrier="braking")
        self.results = []

    def filter(self, obs, action):
        result = super().filter(obs, action)
        self.results.append(result)
        return result


def test_braking_filter_from_inside_starts_never_recovers():
    # Every accepted step ends in the braking set and the backup keeps it, so a start drawn
    # inside never leaves it under the inward nominal.
    filt = RecordedBraking()
    found = audit_filter(
        filt, ApproachEnv(), nominal=Reckless(), starts=5, box=NEAR, seed=SEED, max_steps=30
    )
    assert len(filt.results) == found.steps > 0
    assert [r.recovering for r in filt.results].count(True) == 0
    assert all(r.certificate == "nominal" for r in filt.results)
    assert found.uncovered == 0 and found.passed


def tiny_search(filt, env=None, box=NEAR, max_steps=2):
    """Two runs of a one-chain search, then the re-fly of the worse start."""
    env = ApproachEnv() if env is None else env
    return falsify(
        filt, env, nominal=Reckless(), box=box, budget=2, chains=1, seed=SEED,
        max_steps=max_steps,
    )  # fmt: skip


def test_a_run_with_no_refinement_left_is_uncertified_although_it_scores_below_minus_one(
    monkeypatch,
):
    monkeypatch.setattr(audit_module, "REFINEMENTS", ())
    # The coarse braking runs of the audit tests: their certified bounds miss the resolution
    # with no violation, so the score alone would call the start a violation.
    env = ApproachEnv(ApproachConfig(violation_samples=2))
    box = ((-1.0, 50.2, -1.0, 0.1, -0.01, -0.01), (1.0, 50.6, 1.0, 0.2, 0.0, 0.01))
    found = tiny_search(BackupFilter(barrier="braking"), env, box, max_steps=15)
    assert found.score < -1.0
    assert found.run.violations == 0 and found.run.uncertified > 0
    assert found.verdict == "uncertified"


def test_a_certified_infeasible_decision_gives_the_verdict_infeasible():
    found = tiny_search(CertifiesOnce(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1))
    assert found.score >= -1.0
    assert found.run.infeasible == 1
    assert found.verdict == "infeasible"


def test_a_failed_solve_gives_the_verdict_solver_failed():
    found = tiny_search(FailsOnce(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1))
    assert found.score >= -1.0
    assert found.run.solver_failed == 1
    assert found.verdict == "solver_failed"


def test_the_same_root_seed_gives_the_same_search():
    again = falsify(
        CbfFilter(poles=(0.06, 0.06), gamma=0.06, v_cap=0.1), ApproachEnv(), nominal=Reckless(),
        box=NEAR, budget=30, seed=SEED, max_steps=3,
    )  # fmt: skip
    first = clean_search()
    np.testing.assert_array_equal(np.asarray(again.start), np.asarray(first.start))
    assert again.score == first.score and again.evaluations == first.evaluations


@pytest.mark.parametrize("seed", [(1 << 64) - 1, 3])
def test_the_falsifier_refuses_a_small_root_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        falsify(CbfFilter(), ApproachEnv(), box=NEAR, budget=10, seed=seed, max_steps=1)


def test_the_falsifier_refuses_what_the_audit_refuses():
    soft = SoftCbfFilter(alpha=1.0, poles=(0.05, 0.05), v_cap=0.1, gamma=0.02)
    with pytest.raises(ValueError, match="certification"):
        falsify(soft, ApproachEnv(), box=NEAR, budget=10, seed=SEED, max_steps=1)
    two_body = ApproachEnv(ApproachConfig(dynamics="twobody"))
    with pytest.raises(ValueError, match="cw"):
        falsify(CbfFilter(model_error=1e-6), two_body, box=NEAR, budget=10, seed=SEED,
                max_steps=1)  # fmt: skip
    noisy = ApproachEnv(ApproachConfig(sensor=GaussianNav()))
    with pytest.raises(ValueError, match="navigation"):
        falsify(CbfFilter(), noisy, box=NEAR, budget=10, seed=SEED, max_steps=1)
    pushed = ApproachEnv(ApproachConfig(execution=ExecutionError(noise_sigma=1e-4)))
    with pytest.raises(ValueError, match="execution"):
        falsify(CbfFilter(), pushed, box=NEAR, budget=10, seed=SEED, max_steps=1)
