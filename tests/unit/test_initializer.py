import numpy as np
import pandas
import pytest

from collections import namedtuple

from smcpy.smc.initializer import Initializer
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel


class StubParticles:

    def __init__(self, params, log_likes, log_weights):
        self.params = params
        self.log_likes = log_likes
        self.log_weights = log_weights


@pytest.fixture
def initializer(mocker):
    kernel = mocker.Mock(VectorMCMCKernel)
    mocker.patch('smcpy.smc.initializer.Particles', new=StubParticles)
    initializer = Initializer(kernel)
    return initializer


def test_mcmc_kernel_not_kernel_instance():
    with pytest.raises(TypeError):
        initializer = Initializer(None, None)


@pytest.mark.parametrize('num_particles', [5, 10, 50])
def test_initialize_particles(initializer, mocker, num_particles):
    samples = np.ones((2, num_particles, 3))
    expected_log_weights = np.full(samples.shape, np.log(1 / num_particles))

    particles = initializer(samples)

    initializer.mcmc_kernel.get_log_likelihoods.called_once_with(samples)
    initializer.mcmc_kernel.get_log_priors.called_once_with(samples)
    initializer.mcmc_kernel.conv_param_array_to_dict.called_once_with(samples)
    np.testing.assert_array_equal(particles.log_weights, expected_log_weights)

