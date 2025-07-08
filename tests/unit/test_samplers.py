import numpy as np
import pytest

from smcpy.resampler_rngs import *
from smcpy.smc.particles import Particles
from smcpy import FixedPhiSampler, AdaptiveSampler, FixedTimeSampler
from smcpy.mcmc.kernel_base import KernelBase
from smcpy.paths import GeometricPath

SAMPLERS = "smcpy.smc.samplers"
SAMPLER_BASE = "smcpy.smc.samplers"


@pytest.fixture
def mcmc_kernel(mocker):
    return mocker.Mock(KernelBase)


@pytest.fixture
def phi_sequence():
    return np.linspace(0, 1, 11)


@pytest.fixture
def result_mock(mocker):
    result_mock = mocker.Mock()
    result_mock.is_restart = False
    result_mock.estimate_marginal_log_likelihoods.return_value = 34
    return result_mock


@pytest.mark.parametrize("proposal", [True, False])
@pytest.mark.parametrize("rank", [0, 1, 2])
@pytest.mark.parametrize("prog_bar", [True, False])
def test_fixed_phi_sample(mocker, proposal, rank, prog_bar, mcmc_kernel, result_mock):
    num_particles = 100
    num_steps = 10
    num_mcmc_samples = 2
    phi_sequence = np.arange(num_steps)

    path = GeometricPath()

    mcmc_kernel.has_proposal.return_value = proposal
    mcmc_kernel.get_log_likelihoods.return_value = np.ones((num_particles, 1))
    mcmc_kernel.sample_from_prior.return_value = {"a": np.ones(num_particles)}
    mcmc_kernel.sample_from_proposal.return_value = {"a": np.ones(num_particles)}
    mcmc_kernel.path = path

    upd_mock = mocker.Mock()
    upd_mock.resample_if_needed = lambda x: x
    upd = mocker.patch(SAMPLERS + ".Updater", return_value=upd_mock)

    mocked_mutator = mocker.Mock()
    mocked_mut_particles = mocker.Mock()
    mocked_mut_particles.attrs = {"mutation_ratio": 0.3}
    mocked_mutator.mutate.return_value = mocked_mut_particles
    mut = mocker.patch(SAMPLER_BASE + ".Mutator", return_value=mocked_mutator)

    mocker.patch(SAMPLER_BASE + ".InMemoryStorage", return_value=result_mock)

    mcmc_kernel._mcmc = mocker.Mock()
    mcmc_kernel._mcmc._rank = rank
    comm = mcmc_kernel._mcmc._comm = mocker.Mock()
    mocker.patch.object(comm, "bcast", new=lambda x, root: x)
    ess_threshold = 0.2

    smc = FixedPhiSampler(mcmc_kernel)
    step_list, mll = smc.sample(
        num_particles,
        num_mcmc_samples,
        phi_sequence,
        ess_threshold,
    )

    upd.assert_called_once_with(
        ess_threshold,
        mcmc_kernel,
        resample_rng=standard,
        particles_warn_threshold=0.01,
    )
    mut.assert_called_once_with(smc._mcmc_kernel)

    if rank == 0:
        num_saves = len(result_mock.save_step.call_args_list)
        assert num_saves == len(phi_sequence)
    assert mll == 34
    assert smc.step == mocked_mut_particles

    if proposal:
        mcmc_kernel.sample_from_proposal.assert_called_once()
        mcmc_kernel.sample_from_prior.assert_not_called()
    else:
        mcmc_kernel.sample_from_proposal.assert_not_called()
        mcmc_kernel.sample_from_prior.assert_called_once()


@pytest.mark.parametrize("proposal", [True, False])
@pytest.mark.parametrize(
    "phi_old,target_ess,expected_ess_margin",
    [(1, 0.8, 1), (0.5, 0.8, -0.5364679374), (0, 0.8, -1.8650126), (1, 1, 0)],
)
def test_adaptive_ess_margin(
    mocker, mcmc_kernel, phi_old, expected_ess_margin, target_ess, proposal
):
    phi_new = 1
    num_particles = 5
    particles = mocker.Mock()
    particles.log_likes = np.arange(5).reshape(-1, 1)
    particles.num_particles = num_particles
    particles._logsum = Particles._logsum

    prop_mock = mocker.Mock()
    prop_mock.logpdf.return_value = np.ones((num_particles, 1))
    mcmc_kernel.path = GeometricPath(proposal=prop_mock if proposal else None)
    mcmc_kernel.path._phi_list = [0, phi_old]
    mcmc_kernel.has_proposal.return_value = proposal
    mcmc_kernel.get_log_priors.return_value = np.ones((num_particles, 1))

    smc = AdaptiveSampler(mcmc_kernel)
    ESS_margin = smc.predict_ess_margin(phi_new, phi_old, particles, target_ess)

    assert pytest.approx(ESS_margin) == expected_ess_margin
    if phi_new - phi_old > 0:
        if proposal:
            prop_mock.logpdf.assert_called_once()
        mcmc_kernel.get_log_priors.assert_called_once()


def test_adaptive_ess_margin_nan(mocker, mcmc_kernel):
    mocker.patch(SAMPLERS + ".np.sum", return_value=0)

    phi_old = 0
    phi_new = 1
    num_particles = 5
    particles = mocker.Mock()
    particles.log_likes = np.arange(5).reshape(-1, 1)
    particles.num_particles = num_particles
    particles._logsum.return_value = -np.inf
    target_ess = 0.8

    mcmc_kernel.path = GeometricPath()
    mcmc_kernel.has_proposal.return_value = False
    mcmc_kernel.get_log_priors.return_value = np.ones((num_particles, 1))

    smc = AdaptiveSampler(mcmc_kernel)
    ESS_margin = smc.predict_ess_margin(phi_new, phi_old, particles, target_ess)

    assert ESS_margin == -particles.num_particles * target_ess


@pytest.mark.parametrize("bisect_return", [0.5, 0.75, 0.8])
def test_adaptive_step_optimization(mocker, mcmc_kernel, bisect_return):
    phi_old = 0.2
    particles = mocker.Mock()
    bisect = mocker.patch(SAMPLERS + ".bisect", return_value=bisect_return)

    mcmc_kernel.path = GeometricPath()

    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, "predict_ess_margin", return_value=-2)

    assert smc.optimize_step(particles, phi_old) == bisect_return
    assert bisect.call_args_list[0] == [
        (smc.predict_ess_margin, 0.2, 1),
        {"args": (0.2, particles, 1)},
    ]


@pytest.mark.parametrize("req_phi, phi", [(1, 1), (0.5, 1), (1.5, 1), (0.9, 0.9)])
def test_adaptive_step_optimization_gt_1(mocker, mcmc_kernel, req_phi, phi):
    phi_old = 0.8
    particles = mocker.Mock()
    bisect = mocker.patch(SAMPLERS + ".bisect", return_value=1.4)

    mcmc_kernel.path = GeometricPath(required_phi=req_phi)

    smc = AdaptiveSampler(mcmc_kernel)
    ess_predict = mocker.patch.object(smc, "predict_ess_margin", return_value=2)

    assert smc.optimize_step(particles, phi_old) == phi
    assert not bisect.called
    ess_predict.assert_called_once()


@pytest.mark.parametrize(
    "req_phi", (0.77, [0.77], [0.5, 0.77, 0.8], [0.8, 0.5, 0.77, 0.9])
)
def test_adaptive_step_optimization_req_phi(mocker, mcmc_kernel, req_phi):
    phi_old = 0.52
    particles = mocker.Mock()
    bisect = mocker.patch(SAMPLERS + ".bisect", return_value=0.8)

    mcmc_kernel.path = GeometricPath(required_phi=req_phi)

    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, "predict_ess_margin", return_value=-2)

    assert smc.optimize_step(particles, phi_old) == 0.77


@pytest.mark.parametrize("proposal", [True, False])
@pytest.mark.parametrize("required_phi, exp_index", [([0.6, 0.5], [1, 2]), (0.5, [1])])
def test_adaptive_phi_sample(
    mocker, mcmc_kernel, required_phi, exp_index, result_mock, proposal
):
    num_particles = 5
    update_mock = mocker.patch(SAMPLERS + ".Updater")

    mcmc_kernel.path = GeometricPath(required_phi=required_phi)
    mcmc_kernel.has_proposal.return_value = False
    mcmc_kernel.has_proposal.return_value = proposal
    mcmc_kernel.get_log_likelihoods.return_value = np.ones((num_particles, 1))
    mcmc_kernel.sample_from_prior.return_value = {"a": np.ones(num_particles)}
    mcmc_kernel.sample_from_proposal.return_value = {"a": np.ones(num_particles)}

    mocker.patch(SAMPLER_BASE + ".InMemoryStorage", return_value=result_mock)

    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, "optimize_step", side_effect=[0.5, 0.6, 1.0])
    mocker.patch.object(smc, "_mutator")

    mocked_mut_particles = mocker.Mock()
    mocked_mut_particles.attrs = {"mutation_ratio": 0.3}
    mocker.patch.object(smc._mutator, "mutate", return_value=mocked_mut_particles)

    steps, _ = smc.sample(
        num_particles=num_particles,
        num_mcmc_samples=5,
        target_ess=0.8,
    )

    update_mock.assert_called_once_with(
        ess_threshold=1,
        mcmc_kernel=mcmc_kernel,
        resample_rng=standard,
        particles_warn_threshold=0.01,
    )
    np.testing.assert_array_equal(smc._phi_sequence, [0, 0.5, 0.6, 1.0])
    np.testing.assert_array_equal(smc.req_phi_index, exp_index)
    assert len(result_mock.save_step.call_args_list) == 4
    result_mock.estimate_marginal_log_likelihoods.assert_called_once()
    assert smc.step == mocked_mut_particles
    assert smc.step.attrs == {"mutation_ratio": 0.3}

    smc = AdaptiveSampler(mcmc_kernel)
    assert smc._phi_sequence == []
    assert smc.req_phi_index is None

    if proposal:
        mcmc_kernel.sample_from_proposal.assert_called_once()
        mcmc_kernel.sample_from_prior.assert_not_called()
    else:
        mcmc_kernel.sample_from_proposal.assert_not_called()
        mcmc_kernel.sample_from_prior.assert_called_once()


def test_delta_phi_is_zero_and_loglike_neginf(mocker, mcmc_kernel):
    mock_particles = mocker.Mock()
    mock_particles.num_particles = 10
    mock_particles.log_likes = np.full((10, 1), -np.inf)
    mock_particles._logsum = Particles._logsum
    smc = AdaptiveSampler(mcmc_kernel)
    ess_margin = smc.predict_ess_margin(1, 1, mock_particles, target_ess=1)
    assert ess_margin == pytest.approx(0)


@pytest.mark.parametrize(
    "min_dphi,is_floored", [(0.2, True), (0.1, True), (0.01, False), (None, False)]
)
def test_minimum_delta_phi(mocker, mcmc_kernel, result_mock, min_dphi, is_floored):
    expected_phi_seq = (
        np.cumsum([min_dphi] * int(1 / min_dphi))
        if is_floored
        else np.cumsum([0.1] * 10)
    )

    init_mock = mocker.Mock()
    init_mock.init_particles_from_prior.return_value = 1

    init_particles = mocker.Mock()
    init_mock.initialize_particles.return_value = init_particles
    init_particles.attrs = {"mutation_ratio": 0.3}
    mocker.patch(SAMPLER_BASE + ".Initializer", return_value=init_mock)
    update_mock = mocker.patch(SAMPLERS + ".Updater")

    mocker.patch(SAMPLER_BASE + ".InMemoryStorage", return_value=result_mock)

    mcmc_kernel.path = GeometricPath()

    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, "optimize_step", side_effect=np.linspace(0.1, 1, 10))
    mocker.patch.object(smc, "_mutator")

    mocked_mut_particles = mocker.Mock()
    mocked_mut_particles.attrs = {"mutation_ratio": 0.3}
    mocker.patch.object(smc._mutator, "mutate", return_value=mocked_mut_particles)

    _, _ = smc.sample(num_particles=5, num_mcmc_samples=5, min_dphi=min_dphi)

    np.testing.assert_array_almost_equal(smc._phi_sequence[1:], expected_phi_seq)


def test_optimize_step_does_not_alter_req_phi_list(mocker, mcmc_kernel):
    mocker.patch(SAMPLERS + ".bisect", return_value=0.5)
    mcmc_kernel.path = GeometricPath(required_phi=0.2)
    smc = AdaptiveSampler(mcmc_kernel)
    mocker.patch.object(smc, "_full_step_meets_target", return_value=False)

    phi = smc.optimize_step(None, 0.21)

    assert smc._mcmc_kernel.path.required_phi_list == [0.2]
    assert phi == 0.5


@pytest.mark.parametrize(
    "sampler, kwargs",
    (
        (AdaptiveSampler, {}),
        (FixedPhiSampler, {"phi_sequence": [0, 0.5, 1], "ess_threshold": 0.8}),
    ),
)
def test_sampling_strategy_passed_through(sampler, mcmc_kernel, kwargs):
    smc = sampler(mcmc_kernel)
    with pytest.raises(TypeError):
        # TODO this is hacky, just checking bad strategy raises error
        smc.sample(
            num_particles=1, num_mcmc_samples=1, resample_rng="bad-strat", **kwargs
        )


@pytest.mark.parametrize("target_ess", [-100, -3.0, 0, 1, 10.0, 1000])
def test_valid_target_ess(target_ess, mcmc_kernel):
    num_particles = 5

    smc = AdaptiveSampler(mcmc_kernel)

    with pytest.raises(ValueError):
        smc.sample(
            num_particles=num_particles, num_mcmc_samples=2, target_ess=target_ess
        )


@pytest.mark.parametrize(
    "norm_time_threshold, time_knockdown_factor", (((1, 0), (0, 1), (0.5, 0.5)))
)
def test_fixed_time_initialized(
    mcmc_kernel, norm_time_threshold, time_knockdown_factor
):
    time = 10

    smc = FixedTimeSampler(
        mcmc_kernel=mcmc_kernel,
        time=time,
        norm_time_threshold=norm_time_threshold,
        time_knockdown_factor=time_knockdown_factor,
    )

    assert smc.buffer_time == norm_time_threshold * time
    assert smc.final_time == time_knockdown_factor * time


def test_fixed_time_initialized_default(mcmc_kernel):
    smc = FixedTimeSampler(mcmc_kernel=mcmc_kernel, time=1)

    assert smc.buffer_time == 0.8
    assert smc.final_time == 0.95

    assert smc._time_history == [0]
    assert smc._buffer_phi == None
    assert smc._start_time == None


def test_fixed_time_track_time_elapsed(mocker, mcmc_kernel):
    smc = FixedTimeSampler(mcmc_kernel=mcmc_kernel, time=100)
    smc._start_time = 1

    mocker.patch("time.time", side_effect=[2, 4, 7])
    mocker.patch(SAMPLERS + ".SamplerBase._do_smc_step")

    smc._do_smc_step(phi=1, num_mcmc_samples=1)
    assert smc._time_history == [0, 1]
    assert smc.prev_time_step == 1

    smc._do_smc_step(phi=1, num_mcmc_samples=1)
    assert smc._time_history == [0, 1, 3]
    assert smc.prev_time_step == 2

    smc._do_smc_step(phi=1, num_mcmc_samples=1)
    assert smc._time_history == [0, 1, 3, 6]
    assert smc.prev_time_step == 3


def test_fixed_time_check_buffer_phi(mocker, mcmc_kernel):
    norm_time_threshold = 0.5
    time_knockdown_factor = 0.8
    time = 100

    mocker.patch("time.time", side_effect=[2, 51, 52])
    mocker.patch(SAMPLERS + ".SamplerBase._do_smc_step")

    smc = FixedTimeSampler(
        mcmc_kernel=mcmc_kernel,
        time=time,
        norm_time_threshold=norm_time_threshold,
        time_knockdown_factor=time_knockdown_factor,
    )
    smc._start_time = 1

    smc._do_smc_step(phi=0.1, num_mcmc_samples=1)
    assert smc._buffer_phi == None

    smc._do_smc_step(phi=0.6, num_mcmc_samples=1)
    assert smc._buffer_phi == 0.6

    smc._do_smc_step(phi=0.9, num_mcmc_samples=1)
    assert smc._buffer_phi == 0.6


def test_fixed_time_predict_buffer_phi(mocker, mcmc_kernel):
    norm_time_threshold = 0.5
    time_knockdown_factor = 0.8
    time = 100

    mocker.patch("time.time", side_effect=[26, 49])
    mocker.patch(SAMPLERS + ".SamplerBase._do_smc_step")
    mocker.patch(SAMPLERS + ".AdaptiveSampler.optimize_step", return_value=0.1)
    smc = FixedTimeSampler(
        mcmc_kernel=mcmc_kernel,
        time=time,
        norm_time_threshold=norm_time_threshold,
        time_knockdown_factor=time_knockdown_factor,
    )
    smc._start_time = 1

    smc._do_smc_step(phi=0.6, num_mcmc_samples=1)
    assert smc._buffer_phi == 0.6

    output_phi = smc.optimize_step(particles=1, phi_old=1)
    assert output_phi == 0.6

    smc._do_smc_step(phi=0.6, num_mcmc_samples=1)
    assert smc._buffer_phi == 0.6


@pytest.mark.parametrize("last_time_history", ((30), (50)))
def test_fixed_time_predict_phi_overshoot(mocker, mcmc_kernel, last_time_history):
    time = 100
    time_knockdown_factor = 0.9

    mocker.patch(SAMPLERS + ".AdaptiveSampler.optimize_step", return_value=0.999)
    smc = FixedTimeSampler(
        mcmc_kernel=mcmc_kernel, time=time, time_knockdown_factor=time_knockdown_factor
    )
    smc._time_history.append(last_time_history)

    phi = smc.optimize_step(particles=1, phi_old=1)
    assert phi == 1


@pytest.mark.parametrize("optimize_phi, expected_output", (((0.1, 0.5)), ((0.9, 0.9))))
def test_fixed_time_take_max_phi(mocker, mcmc_kernel, optimize_phi, expected_output):
    norm_time_threshold = 0.0
    time_knockdown_factor = 0.8
    time = 100

    mocker.patch(SAMPLERS + ".AdaptiveSampler.optimize_step", return_value=optimize_phi)

    smc = FixedTimeSampler(
        mcmc_kernel=mcmc_kernel,
        time=time,
        norm_time_threshold=norm_time_threshold,
        time_knockdown_factor=time_knockdown_factor,
    )
    smc._buffer_phi = 0.5
    smc._time_history.append(0)

    original_phi = smc.optimize_step(particles=1, phi_old=1)
    assert original_phi == expected_output


def test_fixed_time_return_default_adaptive_phi(mocker, mcmc_kernel):
    time = 100

    numpy_mock = mocker.patch("numpy.interp")
    mocker.patch(
        SAMPLERS + ".AdaptiveSampler.optimize_step",
        return_value=0.1,
    )

    smc = FixedTimeSampler(
        mcmc_kernel=mcmc_kernel,
        time=time,
    )

    original_phi = smc.optimize_step(particles=1, phi_old=1)
    numpy_mock.assert_not_called()
    np.testing.assert_almost_equal(original_phi, 0.1)
