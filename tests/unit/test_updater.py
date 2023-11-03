import numpy as np
import pytest

from smcpy.smc.updater import Updater
from smcpy import VectorMCMC, VectorMCMCKernel
from smcpy.paths import GeometricPath


class MockedParticles:

    def __init__(self, params, log_likes, log_weights):
        self.param_names = tuple(params.keys())
        self.param_dict = params
        self.params = np.array([params[k] for k in self.param_names]).T
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.weights = np.exp(log_weights)
        self.num_particles = len(log_weights)
        self.total_unnorm_log_weight = 99

    def compute_ess(self):
        return 0.5 * self.params.shape[0]


@pytest.fixture
def mocked_particles(mocker):
    params = {'a': np.array([1, 1, 2]), 'b': np.array([2, 3, 4])}
    log_likes = np.array([-1, -2, -3]).reshape(-1, 1)
    log_weights = np.log([0.1, 0.6, 0.3]).reshape(-1, 1)
    particles = MockedParticles(params, log_likes, log_weights)
    return particles


@pytest.mark.parametrize('ess_threshold', [-0.1, 1.1])
def test_ess_threshold_valid(ess_threshold):
    with pytest.raises(ValueError):
        Updater(ess_threshold, mcmc_kernel=None)


def test_update_no_proposal(mocked_particles, mocker):
    mocker.patch('smcpy.smc.updater.Particles', new=MockedParticles)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ['a', 'b'])
    mocker.patch.object(kernel, 'get_log_priors', return_value=np.ones((3, 2)))

    updater =  Updater(ess_threshold=0.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    expect_log_weights = mocked_particles.log_likes * delta_phi + \
                         mocked_particles.log_weights

    new_particles = updater.update(mocked_particles)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_equal(new_particles.log_weights, expect_log_weights)
    np.testing.assert_array_equal(new_particles.params, mocked_particles.params)
    np.testing.assert_array_equal(new_particles.log_likes,
                                  mocked_particles.log_likes)


def test_update_w_proposal(mocked_particles, mocker):
    mocker.patch('smcpy.smc.updater.Particles', new=MockedParticles)

    proposal = mocker.Mock()
    proposal.logpdf.return_value = np.full((3, 1), 4)

    path = GeometricPath(proposal=proposal)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ['a', 'b'], path)
    mocker.patch.object(kernel, 'get_log_priors', return_value=np.ones((3, 2)))

    updater =  Updater(ess_threshold=0.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    expect_log_weights = (mocked_particles.log_likes + 2 - 4) * delta_phi + \
                         mocked_particles.log_weights

    new_particles = updater.update(mocked_particles)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_almost_equal(new_particles.log_weights,
                                         expect_log_weights)
    np.testing.assert_array_almost_equal(new_particles.params,
                                         mocked_particles.params)
    np.testing.assert_array_almost_equal(new_particles.log_likes,
                                         mocked_particles.log_likes)


def test_update_w_proposal_and_req_phi(mocked_particles, mocker):
    mocker.patch('smcpy.smc.updater.Particles', new=MockedParticles)

    proposal = mocker.Mock()
    proposal.logpdf.return_value = np.full((3, 1), 4)

    path = GeometricPath(proposal=proposal, required_phi=0.15)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ['a', 'b'], path)
    mocker.patch.object(kernel, 'get_log_priors', return_value=np.ones((3, 2)))

    updater =  Updater(ess_threshold=0.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    log_numer = 4 * 0 + 2 * 1 + mocked_particles.log_likes * 0.2
    log_denom = 4 * (0.05 / 0.15) + 2 * (0.1 / 0.15) + \
                mocked_particles.log_likes * 0.1
    exp_log_weights = mocked_particles.log_weights + log_numer - log_denom

    new_particles = updater.update(mocked_particles)

    assert updater.ess > 0
    assert updater.resampled == False

    np.testing.assert_array_almost_equal(new_particles.log_weights,
                                         exp_log_weights)
    np.testing.assert_array_almost_equal(new_particles.params,
                                         mocked_particles.params)
    np.testing.assert_array_almost_equal(new_particles.log_likes,
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

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ['a', 'b'])

    updater = Updater(ess_threshold=1.0, mcmc_kernel=kernel)
    delta_phi = 0.1
    kernel.path.phi = 0.1
    kernel.path.phi = kernel.path.phi + delta_phi

    new_weights = np.array([0.1, 0.4, 0.5])
    mocker.patch.object(updater, '_compute_new_weights',
                        return_value=np.log(new_weights))

    new_particles = updater.update(mocked_particles)

    assert updater.ess < len(mocked_particles.params)
    assert updater.resampled == True

    np.testing.assert_array_equal(new_particles.log_weights,
                                  np.log(np.ones(3) * 1 / 3))
    np.testing.assert_array_equal(new_particles.log_likes,
                                  mocked_particles.log_likes[resample_indices])
    np.testing.assert_array_equal(new_particles.params,
                                  mocked_particles.params[resample_indices])

    assert new_particles._total_unlw == 99
# 