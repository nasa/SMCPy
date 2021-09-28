import numpy as np
import pytest

from smcpy import SMCSampler


@pytest.fixture
def phi_sequence():
    return np.linspace(0, 1, 11)


@pytest.fixture
def step_list(phi_sequence, mocker):
    num_particles = 5
    step_list = []
    for phi in phi_sequence[1:]:
        particles = mocker.Mock()
        particles.log_weights = np.ones(num_particles).reshape(-1, 1)
        particles.log_likes = np.ones(num_particles).reshape(-1, 1) * phi
        particles.num_particles = num_particles
        step_list.append(particles)
    return step_list


def test_sample(mocker):
    mocked_init = mocker.Mock()
    mocker.patch('smcpy.smc_sampler.Initializer', return_value=mocked_init)
    mocker.patch('smcpy.smc_sampler.Updater')
    mocker.patch('smcpy.smc_sampler.Mutator')
    mcmc_kernel = mocker.Mock()

    proposal_dist = mocker.Mock()

    num_particles = 100
    num_steps = 10
    num_mcmc_samples = 2
    ess_threshold = 0.2
    phi_sequence = np.ones(num_steps)
    prog_bar = False
    proposal = ({'x1': np.array([1, 2]), 'x2': np.array([3, 3])},
                np.array([0.1, 0.1]))

    smc = SMCSampler(mcmc_kernel)
    mocker.patch.object(smc, '_compute_mutation_ratio')
    mocker.patch.object(smc, 'estimate_marginal_log_likelihoods')

    _ = smc.sample(num_particles, num_mcmc_samples, phi_sequence,
                   ess_threshold, proposal, prog_bar)

    mocked_init.init_particles_from_prior.assert_not_called()
    mocked_init.init_particles_from_samples.assert_called_once()

    call_args = mocked_init.init_particles_from_samples.call_args[0]
    for args in zip(call_args, proposal):
        np.testing.assert_array_equal(*args)


@pytest.mark.parametrize('steps', [2, 3, 4, 10])
def test_marginal_likelihood_estimator(mocker, steps):
    updater = mocker.Mock()
    unnorm_weights = np.ones((2, 5, 1))
    updater._unnorm_log_weights = [np.log(unnorm_weights) for _ in range(steps)]
    expected_Z = np.tile([1] + [5 ** (i + 1) for i in range(steps)], (2, 1))

    Z = SMCSampler(None).estimate_marginal_log_likelihoods(updater)

    np.testing.assert_array_almost_equal(Z, np.log(expected_Z))


@pytest.mark.parametrize('new_param_array, expected_ratio',
                         ((np.array([[1, 2], [0, 0], [0, 0], [0, 0]]), 0.25),
                          (np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), 0.00),
                          (np.array([[0, 1], [1, 1], [2, 0], [1, 1]]), 1.00),
                          (np.array([[2, 0], [2, 0], [0, 1], [0, 0]]), 0.75)))
def test_calc_mutation_ratio(mocker, new_param_array, expected_ratio):
    new_param_array = np.tile(new_param_array, (5, 1, 1))
    expected_ratio = np.array([expected_ratio] * 5)

    old_particles = mocker.Mock()
    old_particles.params = np.zeros((5, 4, 2))
    new_particles = mocker.Mock()
    new_particles.params = new_param_array

    ratio = SMCSampler._compute_mutation_ratio(old_particles, new_particles)

    np.testing.assert_array_equal(ratio, expected_ratio)
