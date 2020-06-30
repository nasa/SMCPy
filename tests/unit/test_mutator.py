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


@pytest.fixture
def mutator(stub_mcmc_kernel, stub_comm):
    return Mutator(mcmc_kernel=stub_mcmc_kernel, mpi_comm=stub_comm)


def test_mcmc_kernel_not_translator_instance():
    with pytest.raises(TypeError):
        initializer = Mutator(None)


def test_mutate(mutator, stub_mcmc_kernel, mocker):
    particles = mocker.Mock(Particles, autospec=True)
    num_mcmc_samples = 1
    phi = 1

    mocker.patch('smcpy.smc.mutator.Particles', new=DummyParticles)

    comm = mocker.Mock()
    mocker.patch.object(comm, 'gather', new=lambda x, root: [x])
    mutator._comm = comm
    mocker.patch.object(mutator, 'partition_and_scatter_particles')
    mocker.patch.object(stub_mcmc_kernel, 'mutate_particles',
                        return_value=[1, 2, 3])

    mutated_particles = mutator.mutate(particles, num_mcmc_samples, phi)

    assert mutated_particles.params == 1
    assert mutated_particles.log_likes == 2
    assert mutated_particles.log_weights == 3

    particles.compute_covariance.assert_called()
    mutator.partition_and_scatter_particles.assert_called()
    stub_mcmc_kernel.mutate_particles.assert_called()
