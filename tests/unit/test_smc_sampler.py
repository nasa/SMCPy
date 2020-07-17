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


@pytest.mark.parametrize('prog_bar', [True, False])
def test_sample(mocker, prog_bar):
    init = mocker.patch('smcpy.smc_sampler.Initializer', autospec=True)
    upd = mocker.patch('smcpy.smc_sampler.Updater', autospec=True)
    mut = mocker.patch('smcpy.smc_sampler.Mutator', autospec=True)
    update_bar = mocker.patch('smcpy.smc_sampler.set_bar')

    num_particles = 100
    num_steps = 10
    num_mcmc_samples = 2
    phi_sequence = np.ones(num_steps)
    prog_bar = mocker.patch('smcpy.smc_sampler.tqdm',
                                return_value=phi_sequence[2:])

    mcmc_kernel = mocker.Mock()
    ess_threshold = 0.2

    smc = SMCSampler(mocker.Mock())
    mut_ratio = mocker.patch.object(smc, '_compute_mutation_ratio')
    mll_est = mocker.patch.object(smc, '_estimate_marginal_log_likelihood')
    step_list, _ = smc.sample(num_particles, num_mcmc_samples, phi_sequence,
                              ess_threshold, prog_bar)

    init.assert_called_once_with(smc._mcmc_kernel, phi_sequence[1])
    upd.assert_called_once_with(ess_threshold)
    mut.assert_called_once_with(smc._mcmc_kernel)
    np.testing.assert_array_equal(prog_bar.call_args[0][0], phi_sequence[1:])
    update_bar.assert_called()
    mll_est.assert_called_once()

    assert len(step_list) == len(phi_sequence) - 1

    iterable = zip(mut_ratio.call_args_list, step_list[1:], step_list[:-1])
    for call, new_particles, old_particles in iterable:
        assert call[0][0] == old_particles
        assert call[0][1] == new_particles


def test_marginal_likelihood_estimator(mocker):
    updater = mocker.Mock()
    updater._unnorm_log_weights = [np.ones((5, 1)) for _ in range(5)]
    Z_exp = np.prod([np.sum(np.exp(uw)) for uw in updater._unnorm_log_weights])
    Z = SMCSampler(None)._estimate_marginal_log_likelihood(updater)
    assert Z == np.log(Z_exp)


@pytest.mark.parametrize('new_param_array, expected_ratio',
                         ((np.array([[1, 2], [0, 0], [0, 0], [0, 0]]), 0.25),
                          (np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), 0.00),
                          (np.array([[0, 1], [1, 1], [2, 0], [1, 1]]), 1.00),
                          (np.array([[2, 0], [2, 0], [0, 1], [0, 0]]), 0.75)))
def test_calc_mutation_ratio(mocker, new_param_array, expected_ratio):
    old_particles = mocker.Mock()
    old_particles.params = np.zeros((4, 2))
    new_particles = mocker.Mock()
    new_particles.params = new_param_array

    ratio = SMCSampler._compute_mutation_ratio(old_particles, new_particles)

    assert ratio == expected_ratio
