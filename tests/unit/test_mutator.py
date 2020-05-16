import numpy as np
import pytest

from smcpy.smc.mutator import Mutator


@pytest.fixture
def mutator(stub_mcmc_kernel, stub_comm):
    return Mutator(mcmc_kernel=stub_mcmc_kernel, mpi_comm=stub_comm)


def test_mcmc_kernel_not_translator_instance():
    with pytest.raises(TypeError):
        initializer = Mutator(None)


def test_mutate(mutator, mocker):
    smc_step = mocker.Mock()
    cov = None
    mocker.patch.object(smc_step, 'get_covariance', return_value=cov)
    mocker.patch.object(mutator, 'partition_and_scatter_particles',
                        return_value = \
                        [mocker.Mock(), mocker.Mock(), mocker.Mock()])
    phi = 1
    n_samples = 1
    expected_params = [1, 2, 5]
    expected_log_likes = [0.1, 0.1, 0.1]

    new_smc_step = mutator.mutate(smc_step, num_mcmc_samples=n_samples, phi=phi)
    particles = new_smc_step.particles

    mutator.partition_and_scatter_particles.assert_called()
    mutator.mcmc_kernel.sample.assert_called()
    assert new_smc_step is not smc_step
    for i, p in enumerate(particles):
        assert p.params['a'] == expected_params[i]
        assert p.log_like == expected_log_likes[i]
