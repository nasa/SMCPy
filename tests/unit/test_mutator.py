from copy import copy
import numpy as np
import pytest

from smcpy.smc.mutator import Mutator


@pytest.fixture
def mutator(stub_mcmc_kernel, stub_comm):
    return Mutator(mcmc_kernel=stub_mcmc_kernel, mpi_comm=stub_comm)


def test_mcmc_kernel_not_translator_instance():
    with pytest.raises(TypeError):
        initializer = Mutator(None)


def test_mutate(mutator, stub_mcmc_kernel, mocker):
    smc_step = mocker.Mock()
    num_mcmc_samples = 1
    cov = 1
    phi = 1

    mocker.patch.object(smc_step, 'normalize_step_log_weights')
    mocker.patch.object(smc_step, 'copy', return_value=smc_step)

    comm = mocker.Mock()
    mocker.patch.object(comm, 'gather', new=lambda x, root: [x])
    mutator._comm = comm
    mocker.patch.object(mutator, 'partition_and_scatter_particles')
    mocker.patch.object(stub_mcmc_kernel, 'mutate_particles',
                        return_value=np.array([1, 2, 3]))

    new_smc_step = mutator.mutate(smc_step, num_mcmc_samples, cov, phi)

    np.testing.assert_array_equal(new_smc_step.particles, np.array([1, 2, 3]))

    new_smc_step.normalize_step_log_weights.assert_called()
    mutator.partition_and_scatter_particles.assert_called()
    stub_mcmc_kernel.mutate_particles.assert_called()
