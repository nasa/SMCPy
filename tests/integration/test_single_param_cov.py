import numpy as np
import pytest

from smcpy.smc.particles import Particles


def test_single_param_cov():

    params = {'a': np.array([1, 2, 3])}
    log_weights = np.log([0.25, 0.5, 0.25])
    log_likes = np.ones(3)

    p = Particles(params, log_likes, log_weights)
    cov = p.compute_covariance()

    assert cov == 0.5
    assert cov.shape == (1, 1)
