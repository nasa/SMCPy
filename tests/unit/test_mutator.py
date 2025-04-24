from copy import copy
import numpy as np
import pytest

from smcpy.smc.mutator import Mutator
from smcpy.smc.particles import Particles
from smcpy.paths import GeometricPath


class DummyParticles:
    def __init__(self, params, log_likes, log_weights):
        self.params = params
        self.log_likes = log_likes
        self.log_weights = log_weights
        self.attrs = {}

    def compute_covariance(self):
        return 4


@pytest.fixture
def mutator(stub_mcmc_kernel):
    return Mutator(mcmc_kernel=stub_mcmc_kernel)


def test_mcmc_kernel_not_kernel_instance():
    with pytest.raises(TypeError):
        Mutator(None)


def test_mutate(mutator, stub_mcmc_kernel, mocker):
    num_samples = 100
    cov = 0

    mocked_particles = mocker.Mock(Particles, autospec=True)
    mocked_particles.param_dict = 1
    mocked_particles.params = np.array([[333, 333]])
    mocked_particles.log_likes = 2
    mocked_particles.log_weights = 3
    mocked_particles.total_unnorm_log_weight = 99
    mocker.patch.object(mocked_particles, "compute_covariance", return_value=cov)

    mocker.patch("smcpy.smc.mutator.Particles", new=DummyParticles)

    params = np.array([[333, 333]])
    log_likes = [[33]]
    stub_mcmc_kernel.mutate_particles.return_value = (params, log_likes)
    stub_mcmc_kernel.path = GeometricPath()
    stub_mcmc_kernel.path.phi = 1

    mutated_particles = mutator.mutate(mocked_particles, num_samples)

    np.testing.assert_array_equal(mutated_particles.params, np.array([[333, 333]]))
    assert mutated_particles.log_likes == [[33]]
    assert mutated_particles.log_weights == 3
    assert mutated_particles.attrs["total_unnorm_log_weight"] == 99
    assert mutated_particles.attrs["phi"] == 1
    assert mutated_particles.attrs["mutation_ratio"] == 0

    stub_mcmc_kernel.mutate_particles.assert_called_with(
        mocked_particles.param_dict, num_samples, cov
    )


def test_hidden_turn_off_cov_calculation(mocker, mutator, stub_mcmc_kernel):
    mocked_particles = mocker.Mock(Particles, autospec=True)
    mocked_particles.param_dict = {}
    mocker.patch.object(mocked_particles, "compute_covariance")

    stub_mcmc_kernel.path = mocker.Mock()

    params = np.array([[1]])
    log_likes = [[1]]
    mocker.patch.object(
        mutator.mcmc_kernel, "mutate_particles", return_value=(params, log_likes)
    )

    mocker.patch("smcpy.smc.mutator.Particles", new=DummyParticles)

    mutator._compute_cov = False
    mutator.mutate(mocked_particles, 1)

    mocked_particles.compute_covariance.assert_not_called()


@pytest.mark.parametrize(
    "new_param, expected_ratio",
    (
        (np.array([[1, 2], [0, 0], [0, 0], [0, 0]]), 0.25),
        (np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), 0.00),
        (np.array([[0, 1], [1, 1], [2, 0], [1, 1]]), 1.00),
        (np.array([[2, 0], [2, 0], [0, 1], [0, 0]]), 0.75),
    ),
)
def test_calc_mutation_ratio(mocker, mutator, new_param, expected_ratio):
    old = mocker.Mock()
    old.params = np.zeros((4, 2))
    new = mocker.Mock()
    new.params = new_param

    mutation_ratio = mutator._compute_mutation_ratio(old, new)

    assert mutation_ratio == expected_ratio
