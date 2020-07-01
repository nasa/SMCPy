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
def mutator(stub_mcmc_kernel, stub_comm):
    return Mutator(mcmc_kernel=stub_mcmc_kernel, mpi_comm=stub_comm)


def test_mcmc_kernel_not_translator_instance():
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
    mocker.patch.object(mocked_particles, 'compute_covariance',
                        return_value=cov)

    mocker.patch('smcpy.smc.mutator.Particles', new=DummyParticles)

    comm = mocker.Mock()
    mocker.patch.object(comm, 'gather', new=lambda x, root: [x])
    mutator._comm = comm
    mocker.patch.object(mutator, 'partition_and_scatter_particles',
                        return_value=mocked_particles)
    mocker.patch.object(stub_mcmc_kernel, 'mutate_particles',
                        return_value=[10, 20])

    mutated_particles = mutator.mutate(mocked_particles, phi, num_samples)

    assert mutated_particles.params == 10
    assert mutated_particles.log_likes == 20
    assert mutated_particles.log_weights == 3

    stub_mcmc_kernel.mutate_particles.assert_called_with(
            mocked_particles.param_dict, mocked_particles.log_likes,
            num_samples, cov, phi)
