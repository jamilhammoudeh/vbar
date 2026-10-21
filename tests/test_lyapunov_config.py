"""Gain checks of the CW tracking and Cartesian Lyapunov agents."""

import math

import pytest

from vbar.agents.lyapunov import CWTrackingAgent, LyapunovConfig

NOT_POSITIVE_FINITE = (0.0, -1e-4, math.inf, -math.inf, math.nan)


@pytest.mark.parametrize("k", NOT_POSITIVE_FINITE)
def test_tracking_damping_must_be_positive_and_finite(k):
    with pytest.raises(ValueError, match="k must be positive and finite"):
        CWTrackingAgent(k=k)


@pytest.mark.parametrize("name", ["k1", "k2"])
@pytest.mark.parametrize("value", NOT_POSITIVE_FINITE)
def test_lyapunov_gains_must_be_positive_and_finite(name, value):
    with pytest.raises(ValueError, match=f"{name} must be positive and finite"):
        LyapunovConfig(**{name: value})


def test_smallest_positive_gains_are_accepted():
    assert CWTrackingAgent(k=5e-324).k == 5e-324
    config = LyapunovConfig(k1=5e-324, k2=5e-324)
    assert (config.k1, config.k2) == (5e-324, 5e-324)
