import numpy as np
import pytest
from scipy.linalg import expm
from scipy.stats import chi2

from vbar.spacecraft import GaussianNav, GaussMarkovNav, PerfectNav, RangeScaledNav

# Each statistical check below fails a correct sensor with probability 1e-6 (two-sided); the
# seeds are fixed, so the tests are deterministic.
ALPHA = 1e-6
Z = 4.89  # two-sided 1e-6 normal quantile


def random_truth(rng, count):
    return np.c_[rng.uniform(-2000.0, 2000.0, (count, 3)), rng.uniform(-2.0, 2.0, (count, 3))]


def errors(sensor, truth):
    return np.array([sensor.observe(s, 10.0 * k) - s for k, s in enumerate(truth)])


def assert_mean_and_covariance(err, mean, cov):
    count = len(err)
    sigma = np.sqrt(np.diag(cov))
    assert np.all(np.abs(err.mean(axis=0) - mean) < Z * sigma / np.sqrt(count))
    sample = np.cov(err, rowvar=False)
    ratio = (count - 1) * np.diag(sample) / np.diag(cov)
    assert np.all(ratio > chi2.ppf(ALPHA / 2, count - 1))
    assert np.all(ratio < chi2.ppf(1 - ALPHA / 2, count - 1))
    expected = cov / np.outer(sigma, sigma)
    corr = sample / np.outer(np.sqrt(np.diag(sample)), np.sqrt(np.diag(sample)))
    assert np.all(np.abs(corr - expected) < Z / np.sqrt(count))


def test_perfect_navigation_reports_the_truth():
    sensor = PerfectNav()
    sensor.reset(np.random.default_rng(0))
    state = np.array([1.0, -2.0, 3.0, 0.1, -0.2, 0.3])
    assert np.array_equal(sensor.observe(state, 0.0), state)


def test_nav1_noise_matches_its_specification():
    rng = np.random.default_rng(1)
    sensor = GaussianNav()
    sensor.reset(rng)
    err = errors(sensor, random_truth(rng, 20000))
    cov = np.diag(np.repeat([1.0, 1e-6], 3))
    assert_mean_and_covariance(err, np.zeros(6), cov)


def test_episode_bias_is_constant_within_and_drawn_across_episodes():
    rng = np.random.default_rng(2)
    sensor = GaussianNav(sigma_pos=0.5, sigma_vel=1e-3, bias_pos=2.0, bias_vel=5e-3)
    biases = []
    for _ in range(4000):
        sensor.reset(rng)
        biases.append(sensor.bias.copy())
    assert_mean_and_covariance(np.array(biases), np.zeros(6), np.diag(np.repeat([4.0, 2.5e-5], 3)))
    sensor.reset(rng)
    err = errors(sensor, random_truth(rng, 20000))
    cov = np.diag(np.repeat([0.25, 1e-6], 3))
    assert_mean_and_covariance(err, sensor.bias, cov)


def test_noise_carries_no_information_about_the_true_state():
    rng = np.random.default_rng(3)
    sensor = GaussianNav()
    sensor.reset(rng)
    truth = random_truth(rng, 20000)
    err = errors(sensor, truth)
    corr = np.corrcoef(np.c_[err, truth], rowvar=False)[:6, 6:]
    assert np.all(np.abs(corr) < Z / np.sqrt(len(truth)))


def test_same_stream_gives_the_same_observations():
    truth = random_truth(np.random.default_rng(4), 50)
    runs = []
    for _ in range(2):
        sensor = GaussianNav(bias_pos=1.0)
        sensor.reset(np.random.default_rng(5))
        runs.append([sensor.observe(s, 0.0) for s in truth])
    assert np.array_equal(runs[0], runs[1])


def assert_variance(samples, sigma):
    count = len(samples)
    ratio = (count - 1) * np.var(samples, ddof=1) / sigma**2
    assert chi2.ppf(ALPHA / 2, count - 1) < ratio < chi2.ppf(1 - ALPHA / 2, count - 1)
    assert abs(np.mean(samples)) < Z * sigma / np.sqrt(count)


@pytest.mark.parametrize("rho", [0.5, 30.0, 100.0, 1000.0])
def test_range_scaled_errors_follow_the_law_along_and_across_the_line_of_sight(rho):
    sensor = RangeScaledNav(k_range=0.02, k_lateral=0.005, floor=0.05, c_range=1e-4)
    rng = np.random.default_rng(int(rho))
    sensor.reset(rng)
    los = np.array([0.6, -0.48, 0.64])
    truth = np.r_[rho * los, 0.1, -0.2, 0.05]
    err = np.array([sensor.observe(truth, 0.0) - truth for _ in range(20000)])
    along_axis = err[:, :3] @ los
    across = err[:, :3] - np.outer(along_axis, los)
    along, lateral, velocity = sensor.sigmas(rho)
    assert along == max(0.05, 0.02 * rho, 1e-4 * rho**2)
    assert lateral == max(0.05, 0.005 * rho)
    assert_variance(along_axis, along)
    # Across the line of sight the error spans a plane: two independent components.
    other = np.cross(los, [0.0, 0.0, 1.0])
    other /= np.linalg.norm(other)
    assert_variance(across @ other, lateral)
    assert_variance(across @ np.cross(los, other), lateral)
    for axis in range(3):
        assert_variance(err[:, 3 + axis], velocity)


def test_range_scaled_defaults_are_one_percent_of_range():
    along, lateral, velocity = RangeScaledNav().sigmas(500.0)
    assert along == lateral == pytest.approx(5.0) and velocity == pytest.approx(0.05)
    assert RangeScaledNav().sigmas(0.2)[:2] == (0.01, 0.01)


def gauss_markov_pairs(sensor, lag, count):
    """Errors at t = 0 and t = lag from `count` independent episodes, shape (count, 2, 6)."""
    pairs = []
    for child in np.random.SeedSequence(10).spawn(count):
        sensor.reset(np.random.default_rng(child))
        pairs.append([sensor.observe(np.zeros(6), 0.0), sensor.observe(np.zeros(6), lag)])
    return np.array(pairs)


def test_nav2_error_is_stationary_with_the_specified_correlation():
    sensor = GaussMarkovNav()
    count = 4000
    pairs = gauss_markov_pairs(sensor, 10.0, count)
    sigma = np.repeat([0.04, 4e-4], 3)
    for k in range(2):
        assert_mean_and_covariance(pairs[:, k], np.zeros(6), np.diag(sigma**2))
    # Lag covariance of a stationary process is F(lag) P.
    w = 4e-4 / 0.04
    lagged = expm(np.array([[0.0, 1.0], [-(w**2), -2.0 * w]]) * 10.0) @ np.diag([0.04**2, 4e-4**2])
    for axis in range(3):
        for i in range(2):
            for j in range(2):
                a = pairs[:, 1, 3 * i + axis] / sigma[3 * i]
                b = pairs[:, 0, 3 * j + axis] / sigma[3 * j]
                expected = lagged[i, j] / (sigma[3 * i] * sigma[3 * j])
                assert abs(np.mean(a * b) - expected) < Z * np.sqrt((1 + expected**2) / count)


def test_nav2_velocity_error_is_the_rate_of_its_position_error():
    sensor = GaussMarkovNav()
    sensor.reset(np.random.default_rng(11))
    h = 1e-3
    for t in np.arange(1.0, 50.0, 7.0):
        first = sensor.observe(np.zeros(6), t)
        second = sensor.observe(np.zeros(6), t + h)
        # The driving noise moves the velocity by sqrt(q h), about 2.5e-6 m/s over 1 ms.
        assert np.allclose((second[:3] - first[:3]) / h, first[3:], rtol=0.0, atol=2e-5)
    # White noise of the same size fails this by orders of magnitude.
    white = GaussianNav(0.04, 4e-4)
    white.reset(np.random.default_rng(12))
    first, second = white.observe(np.zeros(6), 0.0), white.observe(np.zeros(6), h)
    assert np.max(np.abs((second[:3] - first[:3]) / h - first[3:])) > 1.0


def test_nav2_added_covariance_is_accurate_on_short_and_long_steps():
    sensor = GaussMarkovNav()
    q2 = sensor.intensity[1, 1]
    for h in (1e-6, 1e-3):
        _, added = sensor._discretize(h)
        # Leading order of the integral; damping corrections are of relative order w h <= 1e-5.
        leading = q2 * np.array([[h**3 / 3, h**2 / 2], [h**2 / 2, h]])
        assert np.allclose(added, leading, rtol=1e-4, atol=0.0)
    F, added = sensor._discretize(50.0)
    assert np.allclose(added, sensor.cov - F @ sensor.cov @ F.T, rtol=1e-10, atol=0.0)
    sensor.reset(np.random.default_rng(14))
    for k in range(1, 200):
        assert np.all(np.isfinite(sensor.observe(np.zeros(6), 1e-6 * k)))


def test_nav2_error_does_not_depend_on_the_true_state():
    sensor = GaussMarkovNav()
    rng = np.random.default_rng(13)
    sensor.reset(rng)
    truth = random_truth(rng, 5000)
    err = np.array([sensor.observe(s, 600.0 * k) - s for k, s in enumerate(truth)])
    corr = np.corrcoef(np.c_[err, truth], rowvar=False)[:6, 6:]
    assert np.all(np.abs(corr) < Z / np.sqrt(len(truth)))


def test_nonphysical_sensor_parameters_are_rejected():
    with pytest.raises(ValueError, match="negative"):
        GaussianNav(sigma_pos=-1.0)
    with pytest.raises(ValueError, match="negative"):
        RangeScaledNav(k_range=-0.01)
    with pytest.raises(ValueError, match="positive"):
        GaussMarkovNav(sigma_pos=0.0)
    with pytest.raises(ValueError, match="damping"):
        GaussMarkovNav(damping=0.0)
