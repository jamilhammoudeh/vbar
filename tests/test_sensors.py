import numpy as np
from scipy.stats import chi2

from vbar.spacecraft import GaussianNav, PerfectNav

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
