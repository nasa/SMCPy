import numpy as np
import pytest

from smcpy import FixedSampler, AdaptiveSampler
from smcpy.mcmc.kernel_base import MCMCKernel

SAMPLERS = 'smcpy.smc.samplers'
SAMPLER_BASE = 'smcpy.smc.sampler_base'


@pytest.fixture
def mcmc_kernel(mocker):
    return mocker.Mock(MCMCKernel)


@pytest.fixture
def phi_sequence():
    return np.linspace(0, 1, 11)


@pytest.fixture
def result_mock(mocker):
    result_mock = mocker.Mock()
    result_mock.is_restart = False
    result_mock.estimate_marginal_log_likelihoods.return_value = 34
    return result_mock


@pytest.mark.parametrize('rank', [0, 1, 2])
@pytest.mark.parametrize('prog_bar', [True, False])
def test_fixed_phi_sample(mocker, rank, prog_bar, mcmc_kernel, result_mock):
    num_particles = 100
    num_steps = 10
    num_mcmc_samples = 2
    phi_sequence = np.ones(num_steps)
    prog_bar = mocker.patch(SAMPLERS + '.tqdm',
                            return_value=phi_sequence[2:])

    init_particles = np.array([1] * num_particles)
    mocked_initializer = mocker.Mock()
    mocked_initializer.init_particles_from_prior.return_value = init_particles
    init = mocker.patch(SAMPLER_BASE + '.Initializer',
                        return_value=mocked_initializer)

    upd_mock = mocker.Mock()
    upd_mock.resample_if_needed = lambda x: x
    upd = mocker.patch(SAMPLERS + '.Updater', return_value=upd_mock)

    mocked_mutator = mocker.Mock()
    mocked_mutator.mutate.return_value = 3
    mut = mocker.patch(SAMPLER_BASE + '.Mutator',
                       return_value=mocked_mutator)

    mocker.patch(SAMPLER_BASE + '.InMemoryStorage',
                 return_value=result_mock)

    update_bar = mocker.patch(SAMPLERS + '.set_bar')

    mcmc_kernel._mcmc = mocker.Mock()
    mcmc_kernel._mcmc._rank = rank
    comm = mcmc_kernel._mcmc._comm = mocker.Mock()
    mocker.patch.object(comm, 'bcast', new=lambda x, root: x)
    ess_threshold = 0.2

    smc = FixedSampler(mcmc_kernel)
    mut_ratio = mocker.patch.object(smc, '_compute_mutation_ratio')
    step_list, mll = smc.sample(num_particles,
                                num_mcmc_samples,
                                phi_sequence,
                                ess_threshold,
                                progress_bar=prog_bar)

    init.assert_called_once_with(smc._mcmc_kernel)
    upd.assert_called_once_with(ess_threshold)
    mut.assert_called_once_with(smc._mcmc_kernel)

    np.testing.assert_array_equal(prog_bar.call_args[0][0], phi_sequence[1:])
    update_bar.assert_called()

    if rank == 0:
        num_saves = len(result_mock.save_step.call_args_list)
        assert num_saves == len(phi_sequence) - 1
    assert mll == 34
    assert smc.step == 3


def test_fixed_phi_sample_with_proposal(mcmc_kernel, mocker, result_mock):
    mocked_init = mocker.Mock()
    mocker.patch(SAMPLER_BASE + '.Initializer', return_value=mocked_init)
    mocker.patch(SAMPLER_BASE + '.Mutator')
    mocker.patch(SAMPLERS + '.Updater')

    mocker.patch(SAMPLER_BASE + '.InMemoryStorage',
                 return_value=result_mock)

    proposal_dist = mocker.Mock()

    num_particles = 100
    num_steps = 10
    num_mcmc_samples = 2
    ess_threshold = 0.2
    phi_sequence = np.ones(num_steps)
    prog_bar = False
    proposal = ({
        'x1': np.array([1, 2]),
        'x2': np.array([3, 3])
    }, np.array([0.1, 0.1]))

    smc = FixedSampler(mcmc_kernel)
    mocker.patch.object(smc, '_compute_mutation_ratio')

    _ = smc.sample(num_particles, num_mcmc_samples, phi_sequence,
                   ess_threshold, proposal, prog_bar)

    mocked_init.init_particles_from_prior.assert_not_called()
    mocked_init.init_particles_from_samples.assert_called_once()

    call_args = mocked_init.init_particles_from_samples.call_args[0]
    for args in zip(call_args, proposal):
        np.testing.assert_array_equal(*args)


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


@pytest.mark.parametrize('phi_old,target_ess,expected_ess_margin',
                         [(1, 0.8, 1), (0.5, 0.8, -0.5364679374),
                          (0, 0.8, -1.8650126), (1, 1, 0)])
def test_adaptive_ess_margin(mocker, mcmc_kernel, phi_old, expected_ess_margin,
                             target_ess):
    phi_new = 1
    particles = mocker.Mock()
    particles.log_likes = np.arange(5)
    particles.num_particles = 5

    smc = AdaptiveSampler(mcmc_kernel)
    ESS_margin = smc.predict_ess_margin(phi_new, phi_old, particles,
                                        target_ess)

    assert pytest.approx(ESS_margin) == expected_ess_margin


def test_adaptive_ess_margin_nan(mocker, mcmc_kernel):
    mocker.patch(SAMPLERS + '.np.sum', return_value=0)

    phi_old = 0
    phi_new = 1
    particles = mocker.Mock()
    particles.log_likes = np.arange(5)
    particles.num_particles = 5
    target_ess = 0.8

    smc = AdaptiveSampler(mcmc_kernel)
    ESS_margin = smc.predict_ess_margin(phi_new, phi_old, particles,
                                        target_ess)

    assert ESS_margin == -particles.num_particles * target_ess


@pytest.mark.parametrize('bisect_return', [0.5, 0.75, 0.8])
def test_adaptive_step_optimization(mocker, mcmc_kernel, bisect_return):
    phi_old = 0.2
    particles = mocker.Mock()
    bisect = mocker.patch(SAMPLERS + '.bisect', return_value=bisect_return)

    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, 'predict_ess_margin', return_value=-2)

    assert smc.optimize_step(particles, phi_old) == bisect_return
    assert bisect.call_args_list[0] == [(smc.predict_ess_margin, 0.2, 1), {
        'args': (0.2, particles, 1)
    }]


@pytest.mark.parametrize('req_phi, phi', [(1, 1), (.5, 1), (1.5, 1), (.9, .9)])
def test_adaptive_step_optimization_gt_1(mocker, mcmc_kernel, req_phi, phi):
    phi_old = 0.8
    particles = mocker.Mock()
    bisect = mocker.patch(SAMPLERS + '.bisect', return_value=1.4)

    smc = AdaptiveSampler(mcmc_kernel)
    ess_predict = mocker.patch.object(smc,
                                      'predict_ess_margin',
                                      return_value=2)

    assert smc.optimize_step(particles, phi_old, required_phi=req_phi) == phi
    assert not bisect.called
    ess_predict.assert_called_once()


@pytest.mark.parametrize(
    'req_phi', (0.77, [0.77], [0.5, 0.77, 0.8], [0.8, 0.5, 0.77, 0.9]))
def test_adaptive_step_optimization_req_phi(mocker, mcmc_kernel, req_phi):
    phi_old = 0.52
    particles = mocker.Mock()
    bisect = mocker.patch(SAMPLERS + '.bisect', return_value=0.8)

    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, 'predict_ess_margin', return_value=-2)

    assert smc.optimize_step(particles, phi_old, required_phi=req_phi) == 0.77


@pytest.mark.parametrize('required_phi, exp_index', [([0.6, 0.5], [1, 2]),
                                                     (0.5, [1])])
def test_adaptive_phi_sample(mocker, mcmc_kernel, required_phi, exp_index,
                             result_mock):
    init_mock = mocker.Mock()
    init_mock.init_particles_from_prior.return_value = 1
    mocker.patch(SAMPLER_BASE + '.Initializer', return_value=init_mock)
    update_mock = mocker.patch(SAMPLERS + '.Updater')

    mocker.patch(SAMPLER_BASE + '.InMemoryStorage',
                 return_value=result_mock)

    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, 'optimize_step', side_effect=[0.5, 0.6, 1.0])
    mocker.patch.object(smc, '_mutator')
    mocker.patch.object(smc._mutator, 'mutate', return_value=0.4)
    mocker.patch.object(smc, '_compute_mutation_ratio')

    steps, _ = smc.sample(num_particles=5,
                          num_mcmc_samples=5,
                          target_ess=1,
                          required_phi=required_phi)

    update_mock.assert_called_once_with(ess_threshold=1)
    init_mock.init_particles_from_prior.assert_called_once_with(5)
    np.testing.assert_array_equal(smc._phi_sequence, [0, 0.5, 0.6, 1.0])
    np.testing.assert_array_equal(smc.req_phi_index, exp_index)
    assert len(result_mock.save_step.call_args_list) == 4
    result_mock.estimate_marginal_log_likelihoods.assert_called_once()
    assert smc.step == 0.4

    smc = AdaptiveSampler(mcmc_kernel)
    assert smc._phi_sequence == []
    assert smc.req_phi_index is None


def test_delta_phi_is_zero_and_loglike_neginf(mocker, mcmc_kernel):
    mock_particles = mocker.Mock()
    mock_particles.num_particles = 10
    mock_particles.log_likes = np.full((10, 1), -np.inf)
    smc = AdaptiveSampler(mcmc_kernel)
    assert smc.predict_ess_margin(1, 1, mock_particles, target_ess=1) == 0


@pytest.mark.parametrize("min_dphi,is_floored", [(0.2, True), (0.1, True),
                                                 (0.01, False), (None, False)])
def test_minimum_delta_phi(mocker, mcmc_kernel, result_mock, min_dphi,
                           is_floored):

    expected_phi_seq = np.cumsum([min_dphi] * int(1 / min_dphi)) if is_floored \
                       else np.cumsum([0.1] * 10)

    init_mock = mocker.Mock()
    init_mock.init_particles_from_prior.return_value = 1
    mocker.patch(SAMPLER_BASE + '.Initializer', return_value=init_mock)
    update_mock = mocker.patch(SAMPLERS + '.Updater')

    mocker.patch(SAMPLER_BASE + '.InMemoryStorage',
                 return_value=result_mock)

    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, 'optimize_step',
                        side_effect=np.linspace(0.1, 1, 10))
    mocker.patch.object(smc, '_mutator')
    mocker.patch.object(smc._mutator, 'mutate', return_value=0.4)
    mocker.patch.object(smc, '_compute_mutation_ratio')

    _, _ = smc.sample(num_particles=5, num_mcmc_samples=5, min_dphi=min_dphi)

    np.testing.assert_array_almost_equal(smc._phi_sequence[1:],
                                         expected_phi_seq)