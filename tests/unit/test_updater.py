import numpy as np
import pytest

from smcpy.smc.updater import Updater


class MockedParticle:

    def __init__(self, params, log_likes, log_weights):
        self.params = params
        self.log_likes = log_likes
        self.log_weights = log_weights


@pytest.fixture
def mocked_particles(mocker):
    params = np.array([[1, 2], [1, 3], [2, 4]])
    log_likes = np.array([-1, -2, -3]).reshape(-1, 1)
    log_weights = np.log([0.1, 0.6, 0.3]).reshape(-1, 1)
    particles = MockedParticle(params, log_likes, log_weights)
    particles.weights = np.array([0.1, 0.6, 0.3]).reshape(-1, 1)
    return particles


@pytest.mark.parametrize('ess_threshold', [-0.1, 1.1])
def test_ess_threshold_valid(ess_threshold):
    with pytest.raises(ValueError):
        Updater(ess_threshold)


@pytest.mark.parametrize('ess_threshold', [1])
def test_update(mocked_particles, ess_threshold, mocker):
    mocker.patch('smcpy.smc.updater.Particles', new=MockedParticle)

    updater =  Updater(ess_threshold)
    delta_phi = 0.1

    expect_log_weights = mocked_particles.log_likes * delta_phi + \
                         mocked_particles.log_weights

    new_particles = updater.update(mocked_particles, delta_phi)

    np.testing.assert_array_equal(new_particles.log_weights, expect_log_weights)
    np.testing.assert_array_equal(new_particles.params, mocked_particles.params)
    np.testing.assert_array_equal(new_particles.log_likes,
                                  mocked_particles.log_likes)

# NEED TO WRITE A TEST FOR RESAMPLING
