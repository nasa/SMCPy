import numpy as np
import pytest

from smcpy import FixedSampler
from smcpy.mcmc.kernel_base import MCMCKernel


@pytest.fixture
def mcmc_kernel(mocker):
    return mocker.Mock(MCMCKernel)

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


@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('prog_bar', [True, False])
def test_sample(mocker, rank, prog_bar, mcmc_kernel):
    num_particles = 100
    num_steps = 10
    num_mcmc_samples = 2
    phi_sequence = np.ones(num_steps)
    prog_bar = mocker.patch('smcpy.samplers.tqdm',
                                return_value=phi_sequence[2:])
    expected_step_list = np.ones((num_steps - 1, num_particles))
    expected_step_list[1:] = expected_step_list[1:] + 2

    init_particles = np.array([1] * num_particles)
    mocked_initializer = mocker.Mock()
    mocked_initializer.init_particles_from_prior.return_value = init_particles
    init = mocker.patch('smcpy.sampler_base.Initializer',
                        return_value=mocked_initializer)

    upd_mock = mocker.Mock()
    upd_mock.resample_if_needed = lambda x: x
    upd = mocker.patch('smcpy.samplers.Updater', return_value=upd_mock)

    mocked_mutator = mocker.Mock()
    mocked_mutator.mutate.return_value = np.array([3] * num_particles)
    mut = mocker.patch('smcpy.sampler_base.Mutator',
                       return_value=mocked_mutator)

    update_bar = mocker.patch('smcpy.samplers.set_bar')

    mcmc_kernel._mcmc = mocker.Mock()
    mcmc_kernel._mcmc._rank = rank
    comm = mcmc_kernel._mcmc._comm = mocker.Mock()
    mocker.patch.object(comm, 'bcast', new=lambda x, root: x)
    ess_threshold = 0.2

    smc = FixedSampler(mcmc_kernel)
    mut_ratio = mocker.patch.object(smc, '_compute_mutation_ratio')
    mll_est = mocker.patch.object(smc, '_estimate_marginal_log_likelihoods')
    step_list, mll = smc.sample(num_particles, num_mcmc_samples, phi_sequence,
                                ess_threshold, progress_bar=prog_bar)

    init.assert_called_once_with(smc._mcmc_kernel)
    upd.assert_called_once_with(ess_threshold)
    mut.assert_called_once_with(smc._mcmc_kernel)

    np.testing.assert_array_equal(prog_bar.call_args[0][0], phi_sequence[1:])
    update_bar.assert_called()
    mll_est.assert_called_once()

    assert len(step_list) == len(phi_sequence) - 1
    assert mll is not None
    np.testing.assert_array_equal(step_list, expected_step_list)


def test_sample_with_proposal(mcmc_kernel, mocker):
    mocked_init = mocker.Mock()
    mocker.patch('smcpy.sampler_base.Initializer', return_value=mocked_init)
    mocker.patch('smcpy.sampler_base.Mutator')
    mocker.patch('smcpy.samplers.Updater')

    proposal_dist = mocker.Mock()

    num_particles = 100
    num_steps = 10
    num_mcmc_samples = 2
    ess_threshold = 0.2
    phi_sequence = np.ones(num_steps)
    prog_bar = False
    proposal = ({'x1': np.array([1, 2]), 'x2': np.array([3, 3])},
                np.array([0.1, 0.1]))

    smc = FixedSampler(mcmc_kernel)
    mocker.patch.object(smc, '_compute_mutation_ratio')
    mocker.patch.object(smc, '_estimate_marginal_log_likelihoods')

    _ = smc.sample(num_particles, num_mcmc_samples, phi_sequence,
                   ess_threshold, proposal, prog_bar)

    mocked_init.init_particles_from_prior.assert_not_called()
    mocked_init.init_particles_from_samples.assert_called_once()

    call_args = mocked_init.init_particles_from_samples.call_args[0]
    for args in zip(call_args, proposal):
        np.testing.assert_array_equal(*args)


@pytest.mark.parametrize('steps', [2, 3, 4, 10])
def test_marginal_likelihood_estimator(mocker, steps, mcmc_kernel):
    updater = mocker.Mock()
    unnorm_weights = np.ones((5, 1))
    updater._unnorm_log_weights = [np.log(unnorm_weights) for _ in range(steps)]
    expected_Z = [1] + [5 ** (i + 1) for i in range(steps)]
    smc = FixedSampler(mcmc_kernel)
    smc._updater = updater
    Z = smc._estimate_marginal_log_likelihoods()
    np.testing.assert_array_almost_equal(Z, np.log(expected_Z))


@pytest.mark.parametrize('new_param, expected_ratio',
                         ((np.array([[1, 2], [0, 0], [0, 0], [0, 0]]), 0.25),
                          (np.array([[0, 0], [0, 0], [0, 0], [0, 0]]), 0.00),
                          (np.array([[0, 1], [1, 1], [2, 0], [1, 1]]), 1.00),
                          (np.array([[2, 0], [2, 0], [0, 1], [0, 0]]), 0.75)))
def test_calc_mutation_ratio(mocker, new_param, expected_ratio, mcmc_kernel):
    old = mocker.Mock()
    old.params = np.zeros((4, 2))
    new = mocker.Mock()
    new.params = new_param

    smc = FixedSampler(mcmc_kernel)
    smc._compute_mutation_ratio(old, new)

    assert smc._mutation_ratio == expected_ratio
