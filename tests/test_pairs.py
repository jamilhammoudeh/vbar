import numpy as np
import pytest
from metric_cases import ConstantAgent, GuardedApproach

from vbar import ApproachConfig, ApproachEnv
from vbar.agents import PDAgent
from vbar.dynamics.cw import CWDiscretizer
from vbar.metrics import paired_filter_metrics, record_episode
from vbar.metrics.efficiency import delta_v
from vbar.metrics.episode import record_pair
from vbar.runner import run_pairs
from vbar.safety import FilterResult, PassThroughFilter
from vbar.spacecraft import GaussianNav

# Fixed 128-bit seeds, distinct from each other, as scored runs draw them.
SEED = 0x9E3779B97F4A7C15F39CC0605CEDC834
AGENT_SEED = 0x6A09E667F3BCC908B2FB1366EA957D3E
HOLD = (0.0, -100.0, 0.0, 0.0, 0.0, 0.0)


class InterveneFrom:
    """Passes commands through, then halves them from decision `k` on."""

    name = "intervene-from"

    def __init__(self, k):
        self.k = k

    def reset(self, view):
        self.decisions = 0

    def filter(self, obs, action):
        self.decisions += 1
        if self.decisions <= self.k:
            return FilterResult(action, False)
        return FilterResult(0.5 * action, True, active=True)


class PushOut:
    """Replaces every command with a full radial burn."""

    name = "push-out"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(np.array([1.0, 0.0, 0.0]), True, active=True)


class Quiet:
    """A filter of its own class that never changes a command."""

    name = "quiet"

    def reset(self, view):
        pass

    def filter(self, obs, action):
        return FilterResult(np.array(action, dtype=float), False)


def noisy_env(**cfg):
    cfg.setdefault("max_steps", 8)
    cfg.setdefault("start_state", None)
    return ApproachEnv(ApproachConfig(sensor=GaussianNav(2.0, 2e-3), **cfg))


def test_both_runs_start_from_the_same_state_with_the_same_seeds():
    pair = record_pair(noisy_env(), PDAgent, InterveneFrom(2), seed=SEED, agent_seed=AGENT_SEED)
    assert pair.on.seed == pair.off.seed == SEED
    assert pair.on.agent_seed == pair.off.agent_seed == AGENT_SEED
    np.testing.assert_array_equal(pair.on.start, pair.off.start)
    np.testing.assert_array_equal(pair.on.first_obs, pair.off.first_obs)
    assert pair.filter_name == "intervene-from"
    assert pair.first in ("on", "off")


def test_drawn_seeds_are_128_bit_and_distinct():
    pair = record_pair(noisy_env(max_steps=1), PDAgent, Quiet())
    assert pair.on.seed == pair.off.seed and pair.on.agent_seed == pair.off.agent_seed
    assert pair.on.seed != pair.on.agent_seed
    assert pair.on.seed.bit_length() > 64 and pair.on.agent_seed.bit_length() > 64


def test_the_unfiltered_run_equals_a_plain_recorded_episode():
    env = noisy_env()
    pair = record_pair(env, PDAgent, InterveneFrom(2), seed=SEED, agent_seed=AGENT_SEED)
    plain = record_episode(noisy_env(), PDAgent(), seed=SEED, agent_seed=AGENT_SEED)
    assert len(pair.off.records) == len(plain.records)
    for a, b in zip(pair.off.records, plain.records, strict=True):
        np.testing.assert_array_equal(a.true_state, b.true_state)
        np.testing.assert_array_equal(a.obs, b.obs)
        np.testing.assert_array_equal(a.command, b.command)
        np.testing.assert_array_equal(a.filtered, b.filtered)
        assert a.filter_result is None or a.filter_result.intervened is False


def test_a_pass_through_pair_records_two_identical_episodes():
    pair = record_pair(noisy_env(), PDAgent, PassThroughFilter(), seed=SEED, agent_seed=AGENT_SEED)
    for a, b in zip(pair.on.records, pair.off.records, strict=True):
        np.testing.assert_array_equal(a.true_state, b.true_state)
        np.testing.assert_array_equal(a.obs, b.obs)
        np.testing.assert_array_equal(a.filtered, b.filtered)


def test_a_deterministic_agent_commands_alike_until_the_first_intervention():
    k = 3
    pair = record_pair(noisy_env(), PDAgent, InterveneFrom(k), seed=SEED, agent_seed=AGENT_SEED)
    on, off = pair.on.records, pair.off.records
    for i in range(k + 1):
        # The agent's input is the same up to and including the first changed decision.
        np.testing.assert_array_equal(on[i].command, off[i].command)
    assert not np.array_equal(on[k].filtered, off[k].filtered)
    assert not np.array_equal(on[k + 1].obs, off[k + 1].obs)


def test_one_agent_object_for_both_runs_is_refused():
    shared = PDAgent()
    with pytest.raises(ValueError, match="agent"):
        record_pair(noisy_env(), lambda: shared, Quiet(), seed=SEED, agent_seed=AGENT_SEED)


def test_a_training_environment_is_refused():
    env = ApproachEnv(ApproachConfig(max_steps=2, training=True))
    with pytest.raises(ValueError, match="training"):
        record_pair(env, PDAgent, Quiet(), seed=SEED, agent_seed=AGENT_SEED)


def test_an_agent_seed_equal_to_the_episode_seed_is_refused():
    with pytest.raises(ValueError, match="agent_seed"):
        record_pair(noisy_env(), PDAgent, Quiet(), seed=SEED, agent_seed=SEED)


def test_safety_cost_and_filter_dependence_on_two_scripted_pairs():
    # At rest on the hold point a zero command captures at once. Pushed out at full thrust the
    # chaser leaves the capture speed and burns 0.02 m/s^2 x 10 s = 0.2 m/s in each of 3 steps.
    env = ApproachEnv(ApproachConfig(start_state=HOLD, max_steps=3))
    pushed = record_pair(env, lambda: ConstantAgent(np.zeros(3)), PushOut(), seed=SEED,
                         agent_seed=AGENT_SEED)  # fmt: skip
    quiet = record_pair(env, lambda: ConstantAgent(np.zeros(3)), Quiet(), seed=SEED + 1,
                        agent_seed=AGENT_SEED + 1)  # fmt: skip
    assert delta_v(pushed.on) == pytest.approx(0.6, rel=1e-9)
    assert delta_v(pushed.off) == 0.0
    m = paired_filter_metrics([pushed.on, quiet.on], [pushed.off, quiet.off])
    # The interquartile mean of two values is their mean.
    assert m["safety_dv_cost"] == pytest.approx(0.3, rel=1e-9)
    assert m["safety_dv_cost_by_seed"] == pytest.approx({SEED: 0.6, SEED + 1: 0.0})
    assert m["filter_dependence"] == pytest.approx(0.5 - 1.0)


class Recorder:
    """Keeps everything it is given: the view, the info, the observations and its own stream."""

    name = "recorder"
    runs = []

    def reset(self, view, rng=None):
        self.seen = [repr(view), tuple(rng.integers(0, 2**63, size=4)) if rng else None]
        Recorder.runs.append(self)

    def act(self, obs, info):
        self.seen.append((np.array(obs).tobytes(), repr(sorted(info.items()))))
        return np.zeros(3)


def test_the_agent_cannot_tell_the_filtered_run_from_the_unfiltered_one():
    # Before the filter changes a command, everything the agent receives is the same in both
    # runs, and the order of the runs is random, so neither its inputs nor counting its own
    # runs tells it which one is filtered.
    env = noisy_env(max_steps=3)
    orders, guessed_right = [], 0
    for k in range(64):
        Recorder.runs = []
        pair = record_pair(env, Recorder, Quiet(), seed=SEED + k, agent_seed=AGENT_SEED + k)
        first, second = Recorder.runs
        assert first is not second
        assert first.seen == second.seen
        orders.append(pair.first)
        # An agent that assumes its first run is the filtered one.
        guessed_right += pair.first == "on"
    assert set(orders) == {"on", "off"}
    # Binomial(64, 1/2): outside [12, 52] with probability below 1e-6.
    assert 12 <= guessed_right <= 52


class Remembering:
    """Keeps run 1's observations in a class attribute and fits run 2's start from them."""

    name = "remembering"
    seen: list = []
    runs = 0
    fitted = None
    first_obs = None

    def reset(self, view, rng=None):
        self.view, self.k = view, 0
        type(self).runs += 1

    def act(self, obs, info):
        o = np.array(obs[:6], dtype=float)
        cls = type(self)
        if cls.runs == 1:
            cls.seen.append(o)
        elif self.k == 0:
            d = CWDiscretizer(self.view.n)
            phi = np.vstack([d.matrices(i * self.view.dt)[0] for i in range(len(cls.seen))])
            cls.fitted = np.linalg.lstsq(phi, np.concatenate(cls.seen), rcond=None)[0]
            cls.first_obs = o
        self.k += 1
        return np.zeros(3)


def test_state_outside_the_agent_instance_carries_one_run_of_a_pair_into_the_other():
    # The documented limit: both runs share the start and the noise, so a class attribute
    # recovers the second run's start far better than its first observation does.
    Remembering.seen, Remembering.runs = [], 0
    env = ApproachEnv(
        ApproachConfig(sensor=GaussianNav(2.0, 2e-3), max_steps=300, start_state=None)
    )
    pair = record_pair(env, Remembering, Quiet(), seed=SEED, agent_seed=AGENT_SEED)
    start = np.asarray(pair.on.start, dtype=float)[:3]
    fitted_error = np.linalg.norm(Remembering.fitted[:3] - start)
    naive_error = np.linalg.norm(Remembering.first_obs[:3] - start)
    assert fitted_error < naive_error / 3


def test_run_pairs_flies_one_pair_per_seed_and_scores_them():
    seeds = [SEED + k for k in range(3)]
    agent_seeds = [AGENT_SEED + k for k in range(3)]
    pairs, m = run_pairs(noisy_env(max_steps=3), PDAgent, InterveneFrom(1), seeds, agent_seeds)
    assert [p.on.seed for p in pairs] == seeds
    assert [p.on.agent_seed for p in pairs] == agent_seeds
    assert {"safety_dv_cost", "filter_dependence", "unnecessary_intervention_rate_paired"} <= set(m)


@pytest.mark.parametrize("count", [0, -3, 1.7, float("nan"), float("inf")])
def test_run_pairs_refuses_a_count_that_is_not_a_positive_integer(count):
    with pytest.raises(ValueError):
        run_pairs(noisy_env(max_steps=3), PDAgent, Quiet(), count=count)


def test_run_pairs_flies_a_single_pair_when_count_is_one():
    pairs, _ = run_pairs(noisy_env(max_steps=3), PDAgent, Quiet(), count=1)
    assert len(pairs) == 1


def test_guarded_scenario_pairs_keep_their_soft_constraints():
    env = GuardedApproach(ApproachConfig(start_state=HOLD, max_steps=2), soft=("keep_out",))
    pair = record_pair(env, PDAgent, Quiet(), seed=SEED, agent_seed=AGENT_SEED)
    assert "keep_out" not in pair.on.hard and pair.on.hard == pair.off.hard


def test_the_unfiltered_run_goes_through_the_filter_wrapper():
    # The unfiltered run goes through the same wrapper with the pass-through filter, so each of
    # its records carries a filter result like the filtered run's.
    pair = record_pair(noisy_env(max_steps=2), PDAgent, Quiet(), seed=SEED, agent_seed=AGENT_SEED)
    assert all(r.filter_result is not None for r in pair.off.records)
    assert all(r.filter_result.intervened is False for r in pair.off.records)
