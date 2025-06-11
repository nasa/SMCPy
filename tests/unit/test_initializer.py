import numpy as np
import pandas
import pytest

from collections import namedtuple

from smcpy.smc.initializer import Initializer
from smcpy import VectorMCMC, VectorMCMCKernel
from smcpy.paths import GeometricPath


class StubParticles:
    def __init__(self, params, log_likes, log_weights):
        self.params = params
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.attrs = {}


@pytest.fixture
def initializer(stub_mcmc_kernel, mocker):
    mocker.patch("smcpy.smc.initializer.Particles", new=StubParticles)
    initializer = Initializer(stub_mcmc_kernel)
    return initializer


def test_mcmc_kernel_not_kernel_instance():
    with pytest.raises(TypeError):
        Initializer(None, None)


def test_initialize_particles_from_prior(mocker):
    params = np.ones((4, 3))
    log_likes = np.ones((4, 1))
    log_weights = np.log(np.full_like(log_likes, 0.25))
    priors = [mocker.Mock()]
    priors[0].rvs.return_value = np.ones((4, 3))
    rng = mocker.Mock(np.random.default_rng(), autospec=True)

    vmcmc = VectorMCMC(None, None, priors)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b", "c"], rng=rng)
    mocker.patch.object(kernel, "get_log_likelihoods", return_value=log_likes)

    init = Initializer(kernel)
    particles = init.initialize_particles(num_particles=4)

    priors[0].rvs.assert_called_once_with(4, random_state=rng)
    np.testing.assert_array_almost_equal(particles.params, params)
    np.testing.assert_array_almost_equal(particles.log_likes, log_likes)
    np.testing.assert_array_almost_equal(particles.log_weights, log_weights)
    assert particles.attrs["phi"] == 0
    assert particles.attrs["mutation_ratio"] == 0


def test_initialize_particles_from_proposal(mocker):
    params = np.ones((4, 3))
    log_likes = np.ones((4, 1))
    log_weights = np.log(np.full_like(log_likes, 0.25))
    proposal = mocker.Mock()
    proposal.rvs.return_value = np.ones((4, 3))
    rng = mocker.Mock(np.random.default_rng(), autospec=True)

    path = GeometricPath(proposal=proposal)

    vmcmc = VectorMCMC(None, None, None)
    kernel = VectorMCMCKernel(vmcmc, ["a", "b", "c"], path, rng=rng)
    mocker.patch.object(kernel, "get_log_likelihoods", return_value=log_likes)

    init = Initializer(kernel)
    particles = init.initialize_particles(num_particles=4)

    proposal.rvs.assert_called_once_with(4, random_state=rng)
    np.testing.assert_array_almost_equal(particles.params, params)
    np.testing.assert_array_almost_equal(particles.log_likes, log_likes)
    np.testing.assert_array_almost_equal(particles.log_weights, log_weights)
    assert particles.attrs["phi"] == 0
    assert particles.attrs["mutation_ratio"] == 0
