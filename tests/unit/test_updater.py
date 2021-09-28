import numpy as np
import pytest

from smcpy.smc.updater import Updater

#@pytest.fixture
class MockedParticles:

    def __init__(self, params, log_likes, log_weights):
        self.param_names = tuple(params.keys())
        self.param_dict = params
        self.params = np.array([params[k] for k in self.param_names]).T
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.weights = np.exp(log_weights)
        self.num_particles = len(log_weights)

    def compute_ess(self):
        return 0.5 * self.params.shape[0]


@pytest.fixture
def mocked_particles(mocker):
    params = {'a': np.array([1, 1, 2]), 'b': np.array([2, 3, 4])}
    log_likes = np.array([-1, -2, -3]).reshape(3,1)
    log_weights = np.log([0.1, 0.6, 0.3]).reshape(3, 1)
    particles = MockedParticles(params, log_likes, log_weights)
    return particles

@pytest.fixture
def param_dict():
    s = (2, 5)
    return {'p1': np.ones(s), 'p2': np.full(s, 2), 'p3': np.full(s, 3)}

@pytest.fixture
def particles(param_dict):
    log_likes = np.ones((2, 5, 1))
    weights = np.array([[[1], [4], [1], [1], [1]],
                        [[1], [6], [1], [1], [1]]])
    log_weights = np.log(weights)
    return Particles(param_dict, log_likes, log_weights)


@pytest.mark.parametrize('ess_threshold', [-0.1, 1.1])
def test_ess_threshold_valid(ess_threshold):
    with pytest.raises(ValueError):
        Updater(ess_threshold)


def test_update(mocked_particles, mocker):
    mocker.patch('smcpy.smc.updater.Particles', new=MockedParticles)

    updater =  Updater(ess_threshold=0.0)
    delta_phi = 0.1

    expect_log_weights = mocked_particles.log_likes * delta_phi + \
                         mocked_particles.log_weights

    new_particles = updater.update(mocked_particles, delta_phi)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_equal(new_particles.log_weights, expect_log_weights)
    np.testing.assert_array_equal(new_particles.params, mocked_particles.params)
    np.testing.assert_array_equal(new_particles.log_likes,
                                  mocked_particles.log_likes)

@pytest.mark.parametrize('random_samples, resample_indices',
                         ((np.array([.3, .3, .7]), [1, 1, 2]),
                          (np.array([.3, .05, .05]), [1, 0, 0]),
                          (np.array([.3, .05, .7]), [1, 0, 2]),
                          (np.array([.05, .05, .7]), [0, 0, 2]),
                          (np.array([.3, .7, .3]), [1, 2, 1])))
def test_update_with_resample(mocked_particles, random_samples,
                              resample_indices, mocker):
    mocker.patch('smcpy.smc.updater.Particles', new=MockedParticles)
    mocker.patch('numpy.random.uniform', return_value=random_samples)

    updater = Updater(ess_threshold=1.0)
    new_weights = np.array([0.1, 0.4, 0.5])
    mocker.patch.object(updater, '_compute_new_weights',
                        return_value=np.log(new_weights))

    new_particles = updater.update(mocked_particles, delta_phi=None)

    assert updater.ess < len(mocked_particles.params)
    assert updater.resampled == True

    np.testing.assert_array_equal(new_particles.log_weights,
                                  np.log(np.ones(3) * 1 / 3))
    np.testing.assert_array_equal(new_particles.log_likes,
                                  mocked_particles.log_likes[resample_indices])
    np.testing.assert_array_equal(new_particles.params,
                                  mocked_particles.params[resample_indices])
