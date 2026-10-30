import dataclasses
import math
import time

import numpy as np
import pytest
from metric_cases import episode

from vbar.agents import PDAgent
from vbar.agents.base import action_scale
from vbar.metrics import agent_metrics
from vbar.metrics.filter import filter_metrics
from vbar.safety import (
    FilteredAgent,
    FilterResult,
    InfeasibilityProof,
    PassThroughFilter,
    run_filter,
)
from vbar.safety.cbf import BackupFilter, CbfFilter, SoftCbfFilter
from vbar.safety.corridor import CorridorCamFilter
from vbar.safety.predictive import PredictiveFilter
from vbar.safety.rta import (
    AfrlSetup,
    ExplicitOptimizationFilter,
    ExplicitSwitchingFilter,
    ImplicitOptimizationFilter,
    ImplicitSwitchingFilter,
    reproduce,
)
from vbar.scenarios import make
from vbar.scenarios.approach import ApproachConfig, ApproachEnv
from vbar.spacecraft import PerfectNav

OBS, COMMAND = np.zeros(6), np.zeros(3)
# Sleeps stay short so each test runs well under a second; only lower bounds are asserted on them.
PREP_SLEEP_S = 0.05
FEEDBACK_SLEEP_S = 0.02


class PreparedFilter:
    """Sleeps in `prepare` and in `filter`, logs each call with its clock, and claims `claims`."""

    name = "prepared"

    def __init__(self, prep_sleep_s=0.0, feedback_sleep_s=0.0, **claims):
        self.prep_sleep_s, self.feedback_sleep_s, self.claims = (
            prep_sleep_s,
            feedback_sleep_s,
            claims,
        )
        self.calls = []

    def reset(self, view):
        self.calls = []

    def prepare(self):
        start = time.perf_counter()
        if self.prep_sleep_s:
            time.sleep(self.prep_sleep_s)
        self.calls.append(("prepare", start, time.perf_counter()))

    def filter(self, obs, action):
        start = time.perf_counter()
        if self.feedback_sleep_s:
            time.sleep(self.feedback_sleep_s)
        self.calls.append(("filter", start, time.perf_counter()))
        return FilterResult(action, False, **self.claims)


class UnpreparedFilter:
    """Has no `prepare`; sleeps in `filter` and claims `claims`."""

    name = "unprepared"

    def __init__(self, feedback_sleep_s=0.0, **claims):
        self.feedback_sleep_s, self.claims = feedback_sleep_s, claims

    def reset(self, view):
        pass

    def filter(self, obs, action):
        if self.feedback_sleep_s:
            time.sleep(self.feedback_sleep_s)
        return FilterResult(action, False, **self.claims)


class FailingPrepareFilter(PreparedFilter):
    """Raises from `prepare`."""

    def prepare(self):
        self.calls.append(("prepare", time.perf_counter(), time.perf_counter()))
        raise RuntimeError("warm start lost")


def test_filter_result_has_prep_and_feedback_fields_defaulting_to_none():
    result = FilterResult(COMMAND, False)
    assert result.prep_s is None and result.feedback_s is None


def test_prepare_that_sleeps_50_ms_and_reports_zero_is_recorded_at_50_ms_or_more():
    result, _ = run_filter(PreparedFilter(prep_sleep_s=PREP_SLEEP_S, prep_s=0.0), OBS, COMMAND)
    assert result.prep_s >= PREP_SLEEP_S


def test_feedback_time_is_measured_around_filter_and_excludes_prepare():
    f = PreparedFilter(prep_sleep_s=PREP_SLEEP_S, feedback_sleep_s=FEEDBACK_SLEEP_S)
    before = time.perf_counter()
    result, _ = run_filter(f, OBS, COMMAND)
    after = time.perf_counter()
    (_, prep_start, prep_end), (_, filt_start, filt_end) = f.calls
    assert result.feedback_s >= filt_end - filt_start >= FEEDBACK_SLEEP_S
    assert result.prep_s >= prep_end - prep_start >= PREP_SLEEP_S
    # The clocks bracket each phase from outside, so these bounds hold under any load.
    assert result.feedback_s <= after - prep_end
    assert result.prep_s <= filt_start - before


def test_compute_time_is_the_sum_of_prep_and_feedback_times():
    f = PreparedFilter(prep_sleep_s=PREP_SLEEP_S, feedback_sleep_s=FEEDBACK_SLEEP_S)
    result, _ = run_filter(f, OBS, COMMAND)
    assert result.compute_s == pytest.approx(result.prep_s + result.feedback_s, rel=1e-12)
    assert result.compute_s >= PREP_SLEEP_S + FEEDBACK_SLEEP_S


def test_filter_without_prepare_records_no_prep_time_and_compute_equal_to_feedback():
    result, _ = run_filter(UnpreparedFilter(feedback_sleep_s=FEEDBACK_SLEEP_S), OBS, COMMAND)
    assert result.prep_s is None
    assert result.feedback_s >= FEEDBACK_SLEEP_S
    assert result.compute_s == result.feedback_s


def test_prepared_filter_claims_are_overwritten_by_the_measured_times():
    f = PreparedFilter(
        prep_sleep_s=PREP_SLEEP_S,
        feedback_sleep_s=FEEDBACK_SLEEP_S,
        prep_s=0.0,
        feedback_s=0.0,
        compute_s=0.0,
    )
    result, _ = run_filter(f, OBS, COMMAND)
    assert result.prep_s >= PREP_SLEEP_S
    assert result.feedback_s >= FEEDBACK_SLEEP_S
    assert result.compute_s >= PREP_SLEEP_S + FEEDBACK_SLEEP_S


def test_prepared_filter_cannot_inflate_its_times():
    f = PreparedFilter(prep_s=100.0, feedback_s=100.0, compute_s=100.0)
    before = time.perf_counter()
    result, _ = run_filter(f, OBS, COMMAND)
    wall = time.perf_counter() - before
    assert result.prep_s <= wall and result.feedback_s <= wall and result.compute_s <= wall


def test_unprepared_filter_claimed_prep_time_is_replaced_by_none():
    f = UnpreparedFilter(prep_s=5.0, feedback_s=0.0, compute_s=0.0)
    result, _ = run_filter(f, OBS, COMMAND)
    assert result.prep_s is None
    assert result.feedback_s > 0.0
    assert result.compute_s == result.feedback_s


def test_prepare_is_called_once_per_decision_before_filter():
    f = PreparedFilter()
    for _ in range(3):
        run_filter(f, OBS, COMMAND)
    assert [name for name, _, _ in f.calls] == ["prepare", "filter"] * 3
    starts = [start for _, start, _ in f.calls]
    assert starts == sorted(starts)


def test_filtered_agent_prepares_once_per_decision_and_records_the_split():
    f = PreparedFilter(prep_sleep_s=0.005, prep_s=0.0)
    agent = FilteredAgent(PDAgent(), f)
    episode(agent, start_state=(0.0, -400.0, 30.0, 0.0, 0.0, 0.0), max_steps=5)
    names = [name for name, _, _ in f.calls]
    assert agent.decisions > 0
    assert names == ["prepare", "filter"] * agent.decisions
    assert agent.last_result.prep_s >= 0.005
    assert agent.last_result.compute_s == pytest.approx(
        agent.last_result.prep_s + agent.last_result.feedback_s, rel=1e-12
    )


def test_exception_in_prepare_propagates_and_filter_is_not_called():
    f = FailingPrepareFilter()
    with pytest.raises(RuntimeError, match="warm start lost"):
        run_filter(f, OBS, COMMAND)
    assert [name for name, _, _ in f.calls] == ["prepare"]


class ZeroingResult(FilterResult):
    """Zeroes its own times whenever it is built, `dataclasses.replace` included."""

    def __post_init__(self):
        super().__post_init__()
        for field in ("prep_s", "feedback_s", "compute_s"):
            object.__setattr__(self, field, 0.0)


class ZeroingResultFilter(PreparedFilter):
    """Sleeps in both phases and returns a `ZeroingResult`."""

    def filter(self, obs, action):
        super().filter(obs, action)
        return ZeroingResult(action, False)


def test_result_subclass_cannot_zero_the_measured_times():
    f = ZeroingResultFilter(prep_sleep_s=PREP_SLEEP_S, feedback_sleep_s=FEEDBACK_SLEEP_S)
    try:
        result, elapsed = run_filter(f, OBS, COMMAND)
    except TypeError:
        # Rejecting the subclass outright also keeps a false time out of the record.
        return
    assert result.prep_s >= PREP_SLEEP_S
    assert result.feedback_s >= FEEDBACK_SLEEP_S
    assert result.compute_s >= PREP_SLEEP_S + FEEDBACK_SLEEP_S
    assert elapsed >= PREP_SLEEP_S + FEEDBACK_SLEEP_S


class SlowPrepareLookupFilter(UnpreparedFilter):
    """Spends `PREP_SLEEP_S` resolving its `prepare` attribute, then returns a no-op."""

    @property
    def prepare(self):
        time.sleep(PREP_SLEEP_S)
        return lambda: None


class SlowMissingPrepareFilter(UnpreparedFilter):
    """Spends `PREP_SLEEP_S` in `__getattr__` before reporting it has no `prepare`."""

    def __getattr__(self, name):
        if name == "prepare":
            time.sleep(PREP_SLEEP_S)
        raise AttributeError(name)


@pytest.mark.parametrize("make", [SlowPrepareLookupFilter, SlowMissingPrepareFilter])
def test_time_spent_looking_up_prepare_is_counted(make):
    result, elapsed = run_filter(make(), OBS, COMMAND)
    assert result.compute_s >= PREP_SLEEP_S
    assert elapsed >= PREP_SLEEP_S


def test_returned_wall_time_is_compute_time_with_prepare_included():
    f = PreparedFilter(prep_sleep_s=PREP_SLEEP_S, feedback_sleep_s=FEEDBACK_SLEEP_S)
    result, elapsed = run_filter(f, OBS, COMMAND)
    assert elapsed == result.compute_s
    assert elapsed >= PREP_SLEEP_S + FEEDBACK_SLEEP_S


CONVERSION_SLEEP_S = 0.02


class SlowToConvert:
    """An action that sleeps when numpy converts it, the work a filter could defer to that point."""

    def __init__(self, action):
        self.action = np.array(action, dtype=float)

    def __array__(self, dtype=None, copy=None):
        time.sleep(CONVERSION_SLEEP_S)
        return self.action.astype(dtype or float)


class ReturnsAction:
    """A filter whose result carries `wrap(action)` as its action."""

    name = "returns-action"

    def __init__(self, wrap):
        self.wrap = wrap

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(self.wrap(action), False)


class ZeroAgent:
    name = "zero"

    def reset(self, view, rng=None):
        pass

    def act(self, obs, info):
        return np.zeros(3)


def test_time_spent_converting_the_action_is_counted_in_the_filter_time():
    result, elapsed = run_filter(ReturnsAction(SlowToConvert), OBS, COMMAND)
    assert result.feedback_s >= CONVERSION_SLEEP_S
    assert result.compute_s >= CONVERSION_SLEEP_S
    assert elapsed >= CONVERSION_SLEEP_S


def test_filter_that_defers_its_work_to_the_action_conversion_is_charged_in_f6():
    agent = FilteredAgent(PDAgent(), ReturnsAction(SlowToConvert))
    ep = episode(agent, start_state=(0.0, -400.0, 30.0, 0.0, 0.0, 0.0), max_steps=3)
    assert agent.last_filter_s >= CONVERSION_SLEEP_S
    assert filter_metrics(ep)["filter_compute_median"] >= CONVERSION_SLEEP_S


class SlowNumber:
    """Not a bool, int or float; sleeps however Python converts it to one."""

    def __bool__(self):
        time.sleep(CONVERSION_SLEEP_S)
        return False

    def __int__(self):
        time.sleep(CONVERSION_SLEEP_S)
        return 0

    def __index__(self):
        time.sleep(CONVERSION_SLEEP_S)
        return 0

    def __float__(self):
        time.sleep(CONVERSION_SLEEP_S)
        return 0.0


class SlowStatus(str):
    """A status that sleeps when compared, as the check of a result against the statuses does."""

    def __eq__(self, other):
        time.sleep(CONVERSION_SLEEP_S)
        return str.__eq__(self, other)

    def __ne__(self, other):
        time.sleep(CONVERSION_SLEEP_S)
        return str.__ne__(self, other)

    __hash__ = str.__hash__


class SelfStr(SlowStatus):
    """A slow str whose `str()` is itself, so converting it with `str` changes nothing."""

    def __str__(self):
        return self


class QuietStr(str):
    """A str subclass with no behavior of its own: only its type is wrong."""


class SlowFloat(float):
    """A float that sleeps when converted or compared."""

    def __float__(self):
        time.sleep(CONVERSION_SLEEP_S)
        return float.__float__(self)

    def __eq__(self, other):
        time.sleep(CONVERSION_SLEEP_S)
        return float.__eq__(self, other)

    def __gt__(self, other):
        time.sleep(CONVERSION_SLEEP_S)
        return float.__gt__(self, other)

    __hash__ = float.__hash__


class SlowInt(int):
    """An int that sleeps when converted or compared."""

    def __int__(self):
        time.sleep(CONVERSION_SLEEP_S)
        return int.__int__(self)

    def __index__(self):
        time.sleep(CONVERSION_SLEEP_S)
        return int.__index__(self)

    def __eq__(self, other):
        time.sleep(CONVERSION_SLEEP_S)
        return int.__eq__(self, other)

    __hash__ = int.__hash__


class SlowTuple(tuple):
    """A tuple that sleeps when iterated."""

    def __iter__(self):
        time.sleep(CONVERSION_SLEEP_S)
        return tuple.__iter__(self)


class ReturnsResult:
    """A filter whose result is `make(action)`."""

    name = "returns-result"

    def __init__(self, make):
        self.make = make

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return self.make(action)


class WritesField:
    """Builds an honest result, then writes `value` into `field` past the checks of FilterResult."""

    name = "writes-field"

    def __init__(self, field, value, **honest):
        self.field, self.value, self.honest = field, value, honest

    def reset(self, view):
        pass

    def filter(self, obs, action):
        out = FilterResult(action, False, **self.honest)
        object.__setattr__(out, self.field, self.value)
        return out


# A certificate and its horizon are given together, so the honest half stays in place.
PAIRED = {"certificate": "nominal", "certificate_horizon": 1.0}

# Every value is the wrong type for its field, whether or not it would pass the value checks of
# FilterResult; the slow ones show the work a consumer would otherwise run after the clock.
DISHONEST = [
    pytest.param("intervened", SlowNumber(), id="intervened-slow-number"),
    pytest.param("intervened", 1, id="intervened-int"),
    pytest.param("intervened", "False", id="intervened-str"),
    pytest.param("intervened", None, id="intervened-none"),
    pytest.param("status", SlowStatus("feasible"), id="status-slow-eq"),
    pytest.param("status", SelfStr("feasible"), id="status-str-returns-itself"),
    pytest.param("status", QuietStr("feasible"), id="status-str-subclass"),
    pytest.param("status", np.str_("feasible"), id="status-numpy-str"),
    pytest.param("slack", SlowFloat(0.0), id="slack-slow-float"),
    pytest.param("slack", SlowNumber(), id="slack-slow-number"),
    pytest.param("conflicting", SlowTuple(("keep_out",)), id="conflicting-slow-tuple"),
    pytest.param("conflicting", ("keep_out", SelfStr("speed")), id="conflicting-slow-element"),
    pytest.param("conflicting", ["keep_out"], id="conflicting-list"),
    pytest.param("conflicting", "keep_out", id="conflicting-str"),
    pytest.param("conflicting", (n for n in ("keep_out",)), id="conflicting-generator"),
    pytest.param("compute_s", SlowFloat(0.0), id="compute_s-slow-float"),
    pytest.param("compute_s", SlowNumber(), id="compute_s-slow-number"),
    pytest.param("prep_s", SlowFloat(0.0), id="prep_s-slow-float"),
    pytest.param("prep_s", SlowNumber(), id="prep_s-slow-number"),
    pytest.param("feedback_s", SlowFloat(0.0), id="feedback_s-slow-float"),
    pytest.param("feedback_s", SlowNumber(), id="feedback_s-slow-number"),
    pytest.param("active", SlowNumber(), id="active-slow-number"),
    pytest.param("active", 1, id="active-int"),
    pytest.param("recovering", SlowNumber(), id="recovering-slow-number"),
    pytest.param("recovering", 1, id="recovering-int"),
    pytest.param("fallback_level", SlowInt(0), id="fallback_level-slow-int"),
    pytest.param("fallback_level", SlowNumber(), id="fallback_level-slow-number"),
    pytest.param("fallback_level", True, id="fallback_level-bool"),
    pytest.param("certificate", SelfStr("nominal"), id="certificate-str-returns-itself"),
    pytest.param("certificate", QuietStr("nominal"), id="certificate-str-subclass"),
    pytest.param("certificate", np.str_("nominal"), id="certificate-numpy-str"),
    pytest.param("certificate_horizon", SlowFloat(1.0), id="certificate_horizon-slow-float"),
    pytest.param("certificate_horizon", SlowNumber(), id="certificate_horizon-slow-number"),
    pytest.param("iterations", SlowInt(3), id="iterations-slow-int"),
    pytest.param("iterations", SlowNumber(), id="iterations-slow-number"),
    pytest.param("iterations", True, id="iterations-bool"),
]


@pytest.mark.parametrize(("field", "value"), DISHONEST)
def test_run_filter_refuses_a_result_field_that_is_not_a_plain_type(field, value):
    honest = PAIRED if field in PAIRED else {}
    with pytest.raises(TypeError):
        run_filter(WritesField(field, value, **honest), OBS, COMMAND)


def test_every_result_field_but_action_and_proof_has_a_refusal_case():
    refused = {case.values[0] for case in DISHONEST}
    fields = {field.name for field in dataclasses.fields(FilterResult)}
    assert refused == fields - {"action", "proof"}


# Numpy scalars and ints are not the exact type of a float or int field either.
NOT_PLAIN_NUMBERS = [
    pytest.param("intervened", np.int64(0), id="intervened-numpy-int"),
    pytest.param("slack", np.float64(0.0), id="slack-numpy-float"),
    pytest.param("slack", 0, id="slack-int"),
    pytest.param("certificate_horizon", np.float64(1.0), id="certificate_horizon-numpy-float"),
    pytest.param("fallback_level", np.int64(1), id="fallback_level-numpy-int"),
    pytest.param("iterations", np.int64(2), id="iterations-numpy-int"),
]


@pytest.mark.parametrize(("field", "value"), NOT_PLAIN_NUMBERS)
def test_run_filter_refuses_a_numpy_scalar_or_int_where_a_float_or_int_is_expected(field, value):
    honest = PAIRED if field in PAIRED else {}
    with pytest.raises(TypeError):
        run_filter(WritesField(field, value, **honest), OBS, COMMAND)


def test_a_numpy_bool_is_accepted_as_the_intervention_flag():
    filt = ReturnsResult(lambda action: FilterResult(action, np.bool_(True)))
    result, _ = run_filter(filt, OBS, COMMAND)
    assert type(result.intervened) in (bool, np.bool_)
    assert bool(result.intervened) is True


# Built from the constructor, so the checks of FilterResult run in the filter's own call.
BUILT_DISHONEST = {
    "slow-intervened": lambda action: FilterResult(action, SlowNumber()),
    "slow-status": lambda action: FilterResult(action, False, status=SlowStatus("feasible")),
    "status-str-returns-itself": lambda action: FilterResult(
        action, False, status=SelfStr("feasible")
    ),
}


@pytest.mark.parametrize("build", BUILT_DISHONEST.values(), ids=BUILT_DISHONEST)
def test_filtered_agent_refuses_a_result_that_defers_its_work_to_a_field(build):
    agent = FilteredAgent(ZeroAgent(), ReturnsResult(build))
    agent.reset(make("approach-vbar").agent_view())
    with pytest.raises(TypeError):
        agent.act(OBS, {})


UNCONVERTIBLE = [
    pytest.param({"a": 0.0}, id="dict"),
    pytest.param(1j, id="complex-scalar"),
    pytest.param([1j, 0j, 0j], id="complex-list"),
    pytest.param(np.array([1j, 0.0, 0.0]), id="complex-array"),
    pytest.param("abc", id="string"),
    pytest.param(["a", "b", "c"], id="string-list"),
]


@pytest.mark.parametrize("bad", UNCONVERTIBLE)
def test_run_filter_refuses_an_action_that_does_not_convert_to_floats(bad):
    with pytest.raises(ValueError):
        run_filter(ReturnsAction(lambda action: bad), OBS, COMMAND)


def honest_proof():
    return InfeasibilityProof(
        normals=[[1.0, 0.0, 0.0]],
        bounds=[1.0],
        multipliers=[1.0],
        layout="box",
        a_max=0.5,
        names=("x",),
    )


class ForgedProof:
    """Reports `InfeasibilityProof` as its class, so isinstance passes, and verifies slowly."""

    def __init__(self, proof):
        self.__dict__.update(vars(proof))

    @property
    def __class__(self):
        return InfeasibilityProof

    def verify(self):
        time.sleep(CONVERSION_SLEEP_S)
        return True


class LyingProof(InfeasibilityProof):
    """A subclass whose `verify` sleeps and always says yes."""

    def verify(self):
        time.sleep(CONVERSION_SLEEP_S)
        return True


class ReturnsProof:
    """A filter that certifies infeasibility with `proof`."""

    name = "returns-proof"

    def __init__(self, proof):
        self.proof = proof

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(
            action, True, status="infeasible_certified", conflicting=("x",), proof=self.proof
        )


FAKE_PROOFS = {
    "forged-class": lambda: ForgedProof(honest_proof()),
    "overridden-verify": lambda: LyingProof(**vars(honest_proof())),
    "plain-object": object,
}


@pytest.mark.parametrize("fake", FAKE_PROOFS.values(), ids=FAKE_PROOFS)
def test_run_filter_refuses_a_proof_that_is_not_exactly_an_infeasibility_proof(fake):
    with pytest.raises(TypeError):
        run_filter(ReturnsProof(fake()), OBS, COMMAND)


def test_an_honest_proof_is_recorded_as_an_infeasibility_proof_that_verifies():
    result, _ = run_filter(ReturnsProof(honest_proof()), OBS, COMMAND)
    assert type(result.proof) is InfeasibilityProof
    assert result.proof.verify()


class CountedConversion:
    """An array that sleeps and counts each time numpy converts it."""

    def __init__(self, array):
        self.array, self.calls = np.array(array, dtype=float), 0

    def __array__(self, dtype=None, copy=None):
        self.calls += 1
        time.sleep(CONVERSION_SLEEP_S)
        return self.array.astype(dtype or float)


@pytest.mark.parametrize("field", ["normals", "bounds", "multipliers"])
def test_a_proof_array_that_converts_lazily_is_converted_and_charged_in_the_clock(field):
    proof = honest_proof()
    lazy = CountedConversion(getattr(proof, field))
    object.__setattr__(proof, field, lazy)
    result, elapsed = run_filter(ReturnsProof(proof), OBS, COMMAND)
    assert result.compute_s >= CONVERSION_SLEEP_S
    assert elapsed >= CONVERSION_SLEEP_S
    recorded = getattr(result.proof, field)
    assert type(recorded) is np.ndarray and recorded.dtype == np.float64
    calls = lazy.calls
    assert result.proof.verify()
    assert lazy.calls == calls


# Builtin filters, which are honest: the strict checks must pass every one of them.
A_MAX = ApproachConfig().max_accel
# At rest 400 m out on V-bar, and closing at 0.2 m/s from 120 m, where the filters act.
FAR = (0.0, 400.0, 0.0, 0.0, 0.0, 0.0)
CLOSING = (0.0, 120.0, 0.0, 0.0, -0.2, 0.0)

BUILT_IN = {
    "cbf": lambda: CbfFilter(poles=(0.05, 0.05), v_cap=0.1, gamma=0.02),
    "soft-cbf": lambda: SoftCbfFilter(poles=(0.05, 0.05), v_cap=0.1, gamma=0.02),
    "backup-free-drift": lambda: BackupFilter(barrier="passive"),
    "backup-braking": lambda: BackupFilter(barrier="braking", rho_cap=2000.0, v_cap=1.0),
    "predictive": PredictiveFilter,
    "corridor-cam": lambda: CorridorCamFilter(phases=((0.0, math.inf, (0.0, A_MAX, 0.0), 2),)),
    "pass-through": PassThroughFilter,
}

PLAIN_TYPES = {
    "intervened": (bool, np.bool_),
    "status": (str,),
    "slack": (float,),
    "conflicting": (tuple,),
    "compute_s": (float,),
    "prep_s": (float, type(None)),
    "feedback_s": (float,),
    "active": (bool, np.bool_),
    "recovering": (bool, np.bool_, type(None)),
    "fallback_level": (int, type(None)),
    "certificate": (str, type(None)),
    "certificate_horizon": (float, type(None)),
    "iterations": (int, type(None)),
}


class Charging:
    """Full command straight at the target."""

    name = "charging"

    def reset(self, view, rng=None):
        self.scale = action_scale(view) / view.max_accel

    def act(self, obs, info):
        return -obs[:3] / np.linalg.norm(obs[:3]) * self.scale


def fly(filt, start, steps=6):
    """The filter's recorded results around a charging agent over `steps` steps of 10 s."""
    env = ApproachEnv(
        ApproachConfig(dt=10.0, max_steps=steps, start_state=start, sensor=PerfectNav())
    )
    pilot = FilteredAgent(Charging(), filt)
    obs, _ = env.reset(seed=0)
    pilot.reset(env.agent_view(), np.random.default_rng(0))
    results = []
    for _ in range(steps):
        obs, _, terminated, truncated, _ = env.step(pilot.act(obs, env.agent_info()))
        results.append(pilot.last_result)
        if terminated or truncated:
            break
    return results


def assert_plain(result):
    assert type(result.action) is np.ndarray and result.action.dtype == np.float64
    for field, allowed in PLAIN_TYPES.items():
        assert type(getattr(result, field)) in allowed, field
    assert all(type(name) is str for name in result.conflicting)
    if result.proof is not None:
        assert type(result.proof) is InfeasibilityProof
        for array in (result.proof.normals, result.proof.bounds, result.proof.multipliers):
            assert type(array) is np.ndarray and array.dtype == np.float64


def test_the_plain_type_table_names_every_result_field_but_action_and_proof():
    fields = {field.name for field in dataclasses.fields(FilterResult)}
    assert set(PLAIN_TYPES) == fields - {"action", "proof"}


@pytest.mark.parametrize("build", BUILT_IN.values(), ids=BUILT_IN)
def test_every_builtin_filter_passes_the_strict_checks_over_a_short_episode(build):
    for start in (FAR, CLOSING):
        results = fly(build(), start)
        assert results
        for result in results:
            assert_plain(result)


RTA_FILTERS = (
    ExplicitSwitchingFilter,
    ExplicitOptimizationFilter,
    ImplicitSwitchingFilter,
    ImplicitOptimizationFilter,
)


@pytest.mark.parametrize("cls", RTA_FILTERS)
def test_every_rta_filter_passes_the_strict_checks_over_a_short_episode(cls):
    short = dataclasses.replace(AfrlSetup.authors_code(), stop_time=20.0)
    run = reproduce(cls(setup=short), short, np.array([0.0, 1000.0, 0.0, 0.0, -2.3, 0.0]))
    assert run.results
    for result in run.results:
        assert_plain(result)


def certified_cbf():
    filt = CbfFilter(poles=(0.05, 0.05), v_cap=0.1, gamma=0.02)
    filt.reset(ApproachEnv(ApproachConfig(dt=10.0)).agent_view())
    # Closing at 0.5 m/s from 1 m outside the keep-out: the rows are beyond the thrust box.
    return filt, np.array([0.0, 51.0, 0.0, 0.0, -0.5, 0.0]), np.array([0.0, -1.0, 0.0])


def certified_rta(cls):
    setup = AfrlSetup.authors_code()
    filt = cls(setup=setup)
    filt.reset(setup.view())
    # 2.6 m/s above the speed limit, beyond what the thrust box can undo.
    return filt, np.array([1000.0, 0.0, 0.0, 0.0, -5.0, 0.0]), np.zeros(3)


CERTIFYING = {
    "cbf": certified_cbf,
    "explicit-optimization": lambda: certified_rta(ExplicitOptimizationFilter),
    "implicit-optimization": lambda: certified_rta(ImplicitOptimizationFilter),
}


@pytest.mark.parametrize("setup", CERTIFYING.values(), ids=CERTIFYING)
def test_a_builtin_filters_certificate_of_infeasibility_passes_the_strict_checks(setup):
    filt, x, command = setup()
    result, _ = run_filter(filt, x, command)
    assert result.status == "infeasible_certified" and result.proof is not None
    assert_plain(result)
    assert result.proof.verify()


class SharedBuffer:
    """An action whose conversion returns the filter's own array although a copy is requested."""

    def __init__(self, buffer):
        self.buffer = buffer

    def __array__(self, dtype=None, copy=None):
        return self.buffer


def test_recorded_action_is_not_a_buffer_the_filter_keeps_writing_to():
    buffer = np.array([0.001, 0.002, 0.003])
    agent = FilteredAgent(ZeroAgent(), ReturnsAction(lambda action: SharedBuffer(buffer)))
    agent.reset(make("approach-vbar").agent_view())
    flown = agent.act(OBS, {})
    buffer[:] = 9.0
    np.testing.assert_array_equal(agent.last_result.action, [0.001, 0.002, 0.003])
    np.testing.assert_array_equal(flown, [0.001, 0.002, 0.003])


BAD_ACTIONS = [
    pytest.param([math.nan, 0.0, 0.0], id="nan"),
    pytest.param([0.0, math.inf, 0.0], id="inf"),
    pytest.param([0.0, 0.0, -math.inf], id="minus-inf"),
    pytest.param([0.0, 0.0], id="too-short"),
    pytest.param([0.0, 0.0, 0.0, 0.0], id="too-long"),
    pytest.param([[0.0, 0.0, 0.0]], id="two-dimensional"),
    pytest.param(0.0, id="scalar"),
]


@pytest.mark.parametrize("bad", BAD_ACTIONS)
def test_run_filter_refuses_a_non_finite_or_wrong_shape_action(bad):
    with pytest.raises(ValueError):
        run_filter(ReturnsAction(lambda action: np.array(bad)), OBS, COMMAND)


@pytest.mark.parametrize("bad", BAD_ACTIONS)
def test_a_lazily_converted_bad_action_is_refused_too(bad):
    with pytest.raises(ValueError):
        run_filter(ReturnsAction(lambda action: SlowToConvert(bad)), OBS, COMMAND)


@pytest.mark.parametrize("bad", [[math.nan, 0.0, 0.0], [0.0, 0.0]], ids=["nan", "too-short"])
def test_filtered_agent_refuses_a_bad_action_instead_of_flying_it(bad):
    agent = FilteredAgent(ZeroAgent(), ReturnsAction(lambda action: np.array(bad)))
    agent.reset(make("approach-vbar").agent_view())
    with pytest.raises(ValueError):
        agent.act(OBS, {})


FINALIZER_SLEEP_S = 0.02


class SlowFinalizer:
    """Sleeps when freed, which for a returned object nobody keeps is when run_filter lets go."""

    def __del__(self):
        time.sleep(FINALIZER_SLEEP_S)


class FinalizedAction(SlowFinalizer):
    """An action whose wrapper, not the array it converts to, runs the finalizer."""

    def __init__(self, action):
        self.action = np.array(action, dtype=float)

    def __array__(self, dtype=None, copy=None):
        return self.action.astype(dtype or float)


def result_with_finalized_attribute(action):
    out = FilterResult(action, False)
    object.__setattr__(out, "left_over", SlowFinalizer())
    return out


def result_with_finalized_proof_attribute(action):
    proof = honest_proof()
    object.__setattr__(proof, "left_over", SlowFinalizer())
    return FilterResult(
        action, True, status="infeasible_certified", conflicting=("x",), proof=proof
    )


# Each object is built inside `filter` and held only by the returned result, so run_filter alone
# decides when it is freed. An object the filter keeps is outside the clock by design.
UNKEPT_FINALIZERS = {
    "action-wrapper": lambda action: FilterResult(FinalizedAction(action), False),
    "result-attribute": result_with_finalized_attribute,
    "proof-attribute": result_with_finalized_proof_attribute,
}


@pytest.mark.parametrize("build", UNKEPT_FINALIZERS.values(), ids=UNKEPT_FINALIZERS)
def test_finalizer_of_a_returned_object_the_filter_does_not_keep_is_inside_the_clock(build):
    result, elapsed = run_filter(ReturnsResult(build), OBS, COMMAND)
    assert result.compute_s >= FINALIZER_SLEEP_S
    assert elapsed >= FINALIZER_SLEEP_S


@pytest.mark.parametrize("build", UNKEPT_FINALIZERS.values(), ids=UNKEPT_FINALIZERS)
def test_finalizer_time_is_charged_to_f6_and_not_to_the_agents_latency(build):
    agent = FilteredAgent(PDAgent(), ReturnsResult(build))
    ep = episode(agent, start_state=(0.0, -400.0, 30.0, 0.0, 0.0, 0.0), max_steps=5)
    assert filter_metrics(ep)["filter_compute_median"] >= FINALIZER_SLEEP_S
    # The median over the steps, so one scheduling hiccup cannot fail it.
    assert agent_metrics(ep)["latency_median"] < FINALIZER_SLEEP_S


class FinalizedPrepare(SlowFinalizer):
    """A `prepare` callable that does nothing and sleeps when freed."""

    def __call__(self):
        pass


class FreshPrepareFilter(UnpreparedFilter):
    """Hands out a new `FinalizedPrepare` on each lookup and keeps none of them."""

    @property
    def prepare(self):
        return FinalizedPrepare()


def test_finalizer_of_a_prepare_callable_the_filter_does_not_keep_is_inside_the_prep_time():
    result, elapsed = run_filter(FreshPrepareFilter(), OBS, COMMAND)
    assert result.prep_s >= FINALIZER_SLEEP_S
    assert result.compute_s >= FINALIZER_SLEEP_S
    assert elapsed >= FINALIZER_SLEEP_S


def filter_returning_action(bad):
    return ReturnsAction(lambda action: bad)


def filter_returning_proof_array(field, bad):
    def build(action):
        proof = honest_proof()
        object.__setattr__(proof, field, bad)
        return FilterResult(
            action, True, status="infeasible_certified", conflicting=("x",), proof=proof
        )

    return ReturnsResult(build)


# 10**309 is the smallest power of ten that does not fit a float.
OVERFLOWING = {
    "action-huge-int": lambda: filter_returning_action([10**400, 0, 0]),
    "action-huge-negative-int": lambda: filter_returning_action([0, -(10**400), 0]),
    "action-just-over-float-max": lambda: filter_returning_action([0, 0, 10**309]),
    "action-huge-int-object-array": lambda: filter_returning_action(
        np.array([10**400, 0, 0], dtype=object)
    ),
    "proof-bounds-huge-int": lambda: filter_returning_proof_array("bounds", [10**400]),
}


@pytest.mark.parametrize("build", OVERFLOWING.values(), ids=OVERFLOWING)
def test_run_filter_refuses_an_integer_too_large_for_a_float_with_a_value_error(build):
    with pytest.raises(ValueError):
        run_filter(build(), OBS, COMMAND)


TEXT_AS_NUMBERS = {
    "action-list-of-str": lambda: filter_returning_action(["1", "2", "3"]),
    "action-str-array": lambda: filter_returning_action(np.array(["1", "2", "3"])),
    "action-list-of-bytes": lambda: filter_returning_action([b"1", b"2", b"3"]),
    "action-str-among-numbers": lambda: filter_returning_action(["1", 2, 3.0]),
    "action-object-array-with-str": lambda: filter_returning_action(
        np.array([1.0, "2", 3.0], dtype=object)
    ),
    "proof-multipliers-str": lambda: filter_returning_proof_array("multipliers", ["1.0"]),
}


@pytest.mark.parametrize("build", TEXT_AS_NUMBERS.values(), ids=TEXT_AS_NUMBERS)
def test_run_filter_refuses_an_action_or_proof_array_that_carries_numeric_text(build):
    with pytest.raises(ValueError):
        run_filter(build(), OBS, COMMAND)


@pytest.mark.parametrize(
    "good",
    [[1, 0, 0], (0.5, 1, 2), np.array([1, 2, 3], dtype=np.float32), [0, 0, 10**308]],
    ids=["ints", "mixed-tuple", "float32-array", "large-finite-int"],
)
def test_run_filter_still_accepts_numeric_actions_that_are_not_float64_arrays(good):
    result, _ = run_filter(filter_returning_action(good), OBS, COMMAND)
    assert type(result.action) is np.ndarray and result.action.dtype == np.float64
    np.testing.assert_array_equal(result.action, np.asarray(good, dtype=float))
