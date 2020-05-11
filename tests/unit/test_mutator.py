import numpy as np
import pytest

from smcpy.smc.mutator import Mutator


@pytest.fixture
def mutator(stub_mcmc_kernel, stub_comm):
    return Mutator(mcmc_kernel=stub_mcmc_kernel, mpi_comm=stub_comm)


def test_mcmc_kernel_not_translator_instance():
    with pytest.raises(TypeError):
        initializer = Mutator(None)


@pytest.mark.parametrize('rank,expected',
                         [(0, list(np.array([[1, 2], [3, 4], [5, 6]]))),
                          (1, []), (2, [])])
def test_partition_particles(rank, expected, mutator, mocker):
    smc_step = mocker.Mock()
    smc_step.particles = [1, 2, 3, 4, 5, 6]
    mocker.patch.object(smc_step, 'normalize_step_log_weights', create=True)

    mocker.patch.object(mutator._comm, 'scatter', new=lambda x, root: x)
    mutator._size = 3
    mutator._rank = rank

    particles = mutator.partition_particles(smc_step)

    smc_step.normalize_step_log_weights.assert_called_once_with()
    np.testing.assert_array_equal(particles, expected)


def test_mutate(mutator, mocker):
    smc_step = mocker.Mock()
    cov = None
    mocker.patch.object(smc_step, 'get_covariance', return_value=cov)
    mocker.patch.object(mutator._comm, 'gather', new=lambda x, root: x)
    mocker.patch.object(mutator, 'partition_particles', return_value = \
                        [mocker.Mock(), mocker.Mock(), mocker.Mock()])

    phi = 1
    n_samples = 1
    expected_params = [1, 2, 5]
    expected_log_likes = [0.1, 0.1, 0.1]

    new_smc_step = mutator.mutate(smc_step, num_mcmc_samples=n_samples, phi=phi)
    particles = new_smc_step.particles

    mutator.partition_particles.assert_called()
    mutator.mcmc_kernel.sample.assert_called()
    assert new_smc_step is not smc_step
    for i, p in enumerate(particles):
        assert p.params['a'] == expected_params[i]
        assert p.log_like == expected_log_likes[i]
