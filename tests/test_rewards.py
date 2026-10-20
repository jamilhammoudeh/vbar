import numpy as np
import pytest
from scipy.integrate import quad

from vbar.dynamics import EARTH, cw_discrete, mean_motion
from vbar.rewards import QuadraticCost, Reward, Transition

N = mean_motion(EARTH.radius + 500e3)
# Burns and keep-out passes of one physical trajectory: (start s, end s, delta-v rate m/s^2).
EVENTS = [(120.0, 470.0, 0.01), (613.0, 641.5, 0.02)]


def split_steps(steps, events=EVENTS):
    """Transitions for consecutive steps of the given durations over the same events.

    Each step is split into segments at the event times inside it; a segment's fuel rate is the
    rate of the event covering it, and the events are also the "keep_out" violation intervals,
    each entered once at its start.
    """
    transitions, t = [], 0.0
    for duration in steps:
        inner = {c - t for e in events for c in e[:2] if t < c < t + duration}
        cuts = sorted({0.0, duration, *inner})
        rows, rates = [], []
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            rows.append([a, b - a, 0.0, 0.0, 0.0])
            rates.append(sum(e[2] for e in events if e[0] <= t + a and t + b <= e[1]))
        spans = [
            (max(e[0], t) - t, min(e[1], t + duration) - t)
            for e in events
            if e[0] < t + duration and e[1] > t
        ]
        entries = [e[0] - t for e in events if t <= e[0] < t + duration]
        transitions.append(
            Transition(
                duration=duration,
                segments=np.array(rows),
                starts=np.zeros((len(rows), 6)),
                fuel_rates=np.array(rates),
                violations={"keep_out": np.array(spans).reshape(-1, 2)},
                entries={"keep_out": np.array(entries)},
            )
        )
        t += duration
    return transitions


def discounted_return(reward, transitions):
    total, weight = 0.0, 1.0
    for tr in transitions:
        total += weight * float(reward(tr))
        weight *= float(reward.discount(tr.duration))
    return total, weight


def exp_integral(tau, a, b):
    return b - a if tau is None else tau * (np.exp(-a / tau) - np.exp(-b / tau))


def test_discount_is_declared_per_second():
    # tau = 3600 s gives 0.99723 at 10 s steps and 0.99861 at 5 s.
    reward = Reward(tau=3600.0)
    assert float(reward.discount(10.0)) == pytest.approx(0.99723, abs=5e-6)
    assert float(reward.discount(5.0)) == pytest.approx(0.99861, abs=5e-6)
    assert float(reward.discount(10.0)) == pytest.approx(float(reward.discount(5.0)) ** 2)
    assert float(Reward().discount(10.0)) == 1.0


PARTITIONS = {
    "10 s": [10.0] * 100,
    "5 s": [5.0] * 200,
    "uneven": list(np.diff(np.r_[0.0, np.sort(np.random.default_rng(0).uniform(0, 1e3, 37)), 1e3])),
}


@pytest.mark.parametrize("tau", [None, 600.0, 3600.0])
def test_return_is_the_same_for_every_partition_of_one_trajectory(tau):
    # The semi-Markov return (Bradtke and Duff, 1995): with rewards integrated over the step and
    # discounts exp(-dt / tau) the return is the integral of exp(-t / tau) r(t) for any partition.
    reward = Reward(
        tau=tau,
        time_weight=2.0,
        fuel_weight=30.0,
        violation_weights={"keep_out": 0.5},
        entry_weights={"keep_out": 7.0},
    )
    exact = -2.0 * exp_integral(tau, 0.0, 1e3)
    for start, end, rate in EVENTS:
        exact -= (30.0 * rate + 0.5) * exp_integral(tau, start, end)
        exact -= 7.0 * (1.0 if tau is None else np.exp(-start / tau))
    for steps in PARTITIONS.values():
        total, _ = discounted_return(reward, split_steps(steps))
        assert total == pytest.approx(exact, rel=1e-12)


@pytest.mark.parametrize("tau", [None, 3600.0])
@pytest.mark.parametrize("terminal", [False, True])
def test_progress_shaping_telescopes_to_minus_the_first_potential(tau, terminal):
    # Sutton and Barto (2nd ed., 2018, sec. 3.4) with an absorbing terminal: the shaped terms sum to
    # gamma^T Phi(s_T) - Phi(s_0), and to -Phi(s_0) when the episode ends at a true terminal.
    rng = np.random.default_rng(1)
    steps = rng.uniform(2.0, 20.0, 50)
    distances = rng.uniform(1.0, 500.0, 51)
    reward = Reward(tau=tau, progress_weight=0.01)
    transitions = [
        Transition(
            duration=d,
            segments=np.array([[0.0, d, 0.0, 0.0, 0.0]]),
            starts=np.zeros((1, 6)),
            fuel_rates=np.zeros(1),
            goal_distance=distances[k],
            next_goal_distance=distances[k + 1],
            terminal=terminal and k == len(steps) - 1,
        )
        for k, d in enumerate(steps)
    ]
    total, discount = discounted_return(reward, transitions)
    end = 0.0 if terminal else discount * distances[-1]
    assert total == pytest.approx(0.01 * (distances[0] - end), rel=1e-12)


def test_success_bonus_needs_a_terminal_step_and_success():
    reward = Reward(tau=3600.0, success_bonus=10.0)
    base = dict(
        duration=10.0,
        segments=np.array([[0.0, 10.0, 0.0, 0.0, 0.0]]),
        starts=np.zeros((1, 6)),
        fuel_rates=np.zeros(1),
    )
    assert float(reward(Transition(**base, success=True, terminal=True))) == pytest.approx(
        10.0 * np.exp(-10.0 / 3600.0), rel=1e-15
    )
    assert float(reward(Transition(**base, success=True, terminal=False))) == 0.0
    assert float(reward(Transition(**base, success=False, terminal=True))) == 0.0


@pytest.mark.parametrize("tau", [None, 900.0])
def test_quadratic_cost_matches_quadrature_along_the_cw_trajectory(tau):
    rng = np.random.default_rng(2)
    q = np.diag([1e-4, 2e-4, 3e-4, 1.0, 2.0, 3.0])
    r = 100.0 * np.eye(3)
    reference = np.array([0.0, -100.0, 0.0, 0.0, 0.0, 0.0])
    cost = QuadraticCost(N, q, r, reference)
    x0 = reference + np.r_[rng.uniform(-50, 50, 3), rng.uniform(-0.1, 0.1, 3)]
    a = rng.uniform(-0.02, 0.02, 3)

    def rate(s):
        phi, gamma = cw_discrete(N, s)
        e = phi @ x0 + gamma @ a - reference
        weight = 1.0 if tau is None else np.exp(-s / tau)
        return weight * (e @ q @ e + a @ r @ a)

    expected, _ = quad(rate, 0.0, 37.0, epsabs=0.0, epsrel=1e-13, limit=200)
    found = cost(x0[None], a[None], np.array([37.0]), tau)
    assert float(found[0]) == pytest.approx(expected, rel=1e-10)


def test_batched_transitions_give_each_row_its_own_reward():
    reward = Reward(tau=600.0, time_weight=1.0, fuel_weight=5.0, progress_weight=0.1)
    rows = [
        Transition(
            duration=d,
            segments=np.array([[0.0, d, 0.0, 0.0, 0.0]]),
            starts=np.zeros((1, 6)),
            fuel_rates=np.array([rate]),
            goal_distance=g,
            next_goal_distance=g - 3.0,
            terminal=end,
        )
        for d, rate, g, end in [(10.0, 0.01, 50.0, False), (4.0, 0.0, 9.0, True)]
    ]
    batch = Transition(
        duration=np.array([10.0, 4.0]),
        segments=np.stack([tr.segments for tr in rows]),
        starts=np.zeros((2, 1, 6)),
        fuel_rates=np.stack([tr.fuel_rates for tr in rows]),
        goal_distance=np.array([50.0, 9.0]),
        next_goal_distance=np.array([47.0, 6.0]),
        terminal=np.array([False, True]),
    )
    # Same formulas row by row; vectorized exp may round differently from the scalar one.
    assert reward(batch) == pytest.approx([float(reward(tr)) for tr in rows], rel=1e-14)


@pytest.mark.parametrize(
    "fields",
    [
        {"tau": 0.0},
        {"tau": -5.0},
        {"tau": np.inf},
        {"tau": 5e-324},
        {"fuel_weight": -1.0},
        {"entry_weights": {"x": np.nan}},
    ],
)
def test_invalid_reward_parameters_are_rejected(fields):
    with pytest.raises(ValueError):
        Reward(**fields)


@pytest.mark.parametrize("duration", [10.0, 0.0])
@pytest.mark.parametrize("tau", [0.0, -5.0, np.inf, np.nan, 5e-324])
def test_quadratic_cost_refuses_time_constants_outside_the_normal_floats(tau, duration):
    cost = QuadraticCost(N, np.eye(6), np.eye(3))
    with pytest.raises(ValueError, match="tau must be finite and at least"):
        cost(np.zeros((1, 6)), np.zeros((1, 3)), np.array([duration]), tau)


def test_quadratic_cost_under_the_smallest_time_constants_is_a_point_cost():
    # Sixty time constants of 1e-305 s: the integral is the start's cost rate times tau.
    cost = QuadraticCost(N, Q_COST, R_COST)
    a = np.array([3e-4, -1e-3, 2e-4])
    found = cost(X_COST[None], a[None], np.array([1e4]), 1e-305)[0]
    assert found == pytest.approx((X_COST @ Q_COST @ X_COST + a @ R_COST @ a) * 1e-305, rel=1e-12)
    Reward(tau=1e-305)


def test_quadratic_cost_under_the_largest_time_constants_is_the_undiscounted_cost():
    # 60 tau overflows to infinity for tau near 1e307; the segment's own length must bound it.
    cost = QuadraticCost(N, Q_COST, R_COST)
    a = np.array([3e-4, -1e-3, 2e-4])
    args = (X_COST[None], a[None], np.array([3000.0]))
    assert cost(*args, 1e307)[0] == pytest.approx(cost(*args, None)[0], rel=1e-14)
    Reward(tau=1e307)


@pytest.mark.parametrize("n", [0.0, -1e-3, float("nan"), float("inf")])
def test_quadratic_cost_needs_a_positive_mean_motion(n):
    with pytest.raises(ValueError, match="mean motion"):
        QuadraticCost(n, np.eye(6), np.eye(3))


@pytest.mark.parametrize(("tau", "duration"), [(30.0, 3000.0), (5.0, 1e4), (1.0, 5e3)])
def test_quadratic_cost_stays_exact_on_segments_much_longer_than_tau(tau, duration):
    # Burn-and-coast decisions can last thousands of seconds; the discounted block exponential
    # grows as exp(d / 2 tau) and lost accuracy, then gave NaN, before segments were split.
    q, r = np.diag([1e-4, 2e-4, 3e-4, 1.0, 2.0, 3.0]), 100.0 * np.eye(3)
    x0 = np.array([20.0, -150.0, 5.0, 0.05, -0.02, 0.01])
    a = np.array([1e-3, -2e-3, 5e-4])

    def rate(s):
        phi, gamma = cw_discrete(N, s)
        e = phi @ x0 + gamma @ a
        return np.exp(-s / tau) * (e @ q @ e + a @ r @ a)

    knots = np.linspace(0.0, 60.0 * tau, 13)
    expected = sum(
        quad(rate, lo, hi, epsabs=0.0, epsrel=1e-13, limit=200)[0]
        for lo, hi in zip(knots[:-1], knots[1:], strict=True)
    )
    expected += quad(rate, knots[-1], duration, limit=200)[0]
    found = QuadraticCost(N, q, r)(x0[None], a[None], np.array([duration]), tau)
    assert float(found[0]) == pytest.approx(expected, rel=1e-10)


@pytest.mark.parametrize(("tau", "duration"), [(None, 1e4), (None, 3e4), (1e5, 3e4), (None, 1e5)])
def test_quadratic_cost_stays_exact_on_long_segments_without_much_discount(tau, duration):
    # Without discounting the CW part of the exponential still spans many orbits; one block
    # was 3e-8 off at 1e4 s and 1e-3 off at 3e4 s with SciPy 1.11. The pieces now leave at most
    # 3e-14 there (2e-15 on current SciPy), well inside the bound.
    q, r = np.diag([1e-4, 2e-4, 3e-4, 1.0, 2.0, 3.0]), 100.0 * np.eye(3)
    x0 = np.array([20.0, -150.0, 5.0, 0.05, -0.02, 0.01])
    a = np.array([1e-4, -2e-4, 5e-5])

    def rate(s):
        phi, gamma = cw_discrete(N, s)
        e = phi @ x0 + gamma @ a
        weight = 1.0 if tau is None else np.exp(-s / tau)
        return weight * (e @ q @ e + a @ r @ a)

    # Knots every fiftieth of an orbit keep the quadrature itself at 1e-13.
    knots = np.linspace(0.0, duration, int(np.ceil(duration * N * 50 / (2 * np.pi))) + 1)
    expected = sum(
        quad(rate, lo, hi, epsabs=0.0, epsrel=1e-13)[0]
        for lo, hi in zip(knots[:-1], knots[1:], strict=True)
    )
    found = QuadraticCost(N, q, r)(x0[None], a[None], np.array([duration]), tau)
    assert float(found[0]) == pytest.approx(expected, rel=1e-12)


Q_COST, R_COST = np.diag([1e-4, 2e-4, 3e-4, 1.0, 2.0, 3.0]), 100.0 * np.eye(3)
X_COST = np.array([20.0, -150.0, 5.0, 0.05, -0.02, 0.01])


def quadrature_cost(a, duration, tau):
    """The cost rate integrated along the closed-form CW trajectory, in pieces of 1/50 orbit."""

    def rate(s):
        phi, gamma = cw_discrete(N, s)
        e = phi @ X_COST + gamma @ a
        weight = 1.0 if tau is None else np.exp(-s / tau)
        return weight * (e @ Q_COST @ e + a @ R_COST @ a)

    knots = np.linspace(0.0, duration, int(np.ceil(duration * N * 50 / (2 * np.pi))) + 1)
    return sum(
        quad(rate, lo, hi, epsabs=0.0, epsrel=1e-13)[0]
        for lo, hi in zip(knots[:-1], knots[1:], strict=True)
    )


@pytest.mark.parametrize(
    ("tau", "duration"), [(None, 0.0073961234567), (500.0, 9.98765432149), (None, 300.0)]
)
def test_quadratic_cost_integrates_the_exact_duration(tau, duration):
    # Real-time latency gives durations off any grid; rounding them to 1 ns was up to 6e-8 off.
    # 300 s splits into two pieces of at most a quarter radian (anything past 226 s splits).
    a = np.array([3e-4, -1e-3, 2e-4])
    found = QuadraticCost(N, Q_COST, R_COST)(X_COST[None], a[None], np.array([duration]), tau)
    assert float(found[0]) == pytest.approx(quadrature_cost(a, duration, tau), rel=1e-13)


def test_quadratic_cost_keeps_the_tail_of_a_cost_growing_under_discount():
    # A held acceleration makes the cost grow like s^4; dropping pieces once the discount fell
    # below 1e-17 lost 1.7e-12 here.
    a = np.array([1e-3, -2e-3, 5e-4])
    found = QuadraticCost(N, Q_COST, R_COST)(X_COST[None], a[None], np.array([1e5]), 1000.0)
    assert float(found[0]) == pytest.approx(quadrature_cost(a, 1e5, 1000.0), rel=1e-13)


def test_quadratic_cost_of_a_long_segment_under_a_short_time_constant_is_cheap():
    # 1e8 pieces of 0.01 s if built; past 60 tau the discount leaves nothing, so 60 are.
    cost = QuadraticCost(N, Q_COST, R_COST)
    a = np.array([3e-4, -1e-3, 2e-4])
    long, short = (cost(X_COST[None], a[None], np.array([d]), 0.01)[0] for d in (1e6, 1.0))
    assert long == pytest.approx(short, rel=1e-15)


def test_a_long_undiscounted_segment_summed_in_blocks_matches_its_two_halves():
    # 2e6 s is 8855 pieces, three blocks of 4096; each half needs two. The two sums differ by
    # 1.4e-14 (3.7e-14 with SciPy 1.11), well inside the bound.
    cost = QuadraticCost(N, Q_COST, R_COST)
    a = np.array([3e-5, -1e-4, 2e-5])
    phi, gamma = cw_discrete(N, 1e6)
    middle = phi @ X_COST + gamma @ a
    whole = cost(X_COST[None], a[None], np.array([2e6]), None)[0]
    halves = cost(np.array([X_COST, middle]), np.array([a, a]), np.array([1e6, 1e6]), None)
    assert whole == pytest.approx(halves.sum(), rel=1e-12)
