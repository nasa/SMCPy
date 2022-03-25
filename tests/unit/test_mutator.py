from copy import copy
import numpy as np
import pytest

from smcpy.smc.mutator import Mutator
from smcpy.smc.particles import Particles

class DummyParticles:

    def __init__(self, params, log_likes, log_weights):
        self.params = params
        self.log_likes = log_likes
        self.log_weights = log_weights

    def compute_covariance(self):
        return 4


@pytest.fixture
def mutator(stub_mcmc_kernel):
    return Mutator(mcmc_kernel=stub_mcmc_kernel)


def test_mcmc_kernel_not_kernel_instance():
    with pytest.raises(TypeError):
        initializer = Mutator(None)


def test_mutate(mutator, stub_mcmc_kernel, mocker):
    num_samples = 100
    cov = 0
    phi = 1

    mocked_particles = mocker.Mock(Particles, autospec=True)
    mocked_particles.param_dict = 1
    mocked_particles.log_likes = 2
    mocked_particles.log_weights = 3
    mocked_particles.total_unnorm_log_weight = 99
    mocker.patch.object(mocked_particles, 'compute_covariance',
                        return_value=cov)

    mocker.patch('smcpy.smc.mutator.Particles', new=DummyParticles)

    mocker.patch.object(stub_mcmc_kernel, 'mutate_particles',
                        return_value=[10, 20])

    mutated_particles = mutator.mutate(mocked_particles, phi, num_samples)

    assert mutated_particles.params == 10
    assert mutated_particles.log_likes == 20
    assert mutated_particles.log_weights == 3
    assert mutated_particles._total_unlw == 99

    stub_mcmc_kernel.mutate_particles.assert_called_with(
            mocked_particles.param_dict, mocked_particles.log_likes,
            num_samples, cov, phi)
