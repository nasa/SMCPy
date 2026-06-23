import numpy as np
import pytest

from smcpy import FixedPhiSampler, AdaptiveSampler
from smcpy.mcmc.kernel_base import KernelBase
from smcpy.paths import GeometricPath

SAMPLER_BASE = "smcpy.smc.samplers"


class DummyResult:
    def __init__(self):
        self.phi_sequence = [68, 67]
        self.is_restart = True

    def __getitem__(self, idx):
        return 66


@pytest.fixture
def mcmc_kernel(mocker):
    mcmc_kernel = mocker.Mock(KernelBase)
    mcmc_kernel.path = GeometricPath()
    return mcmc_kernel


@pytest.fixture
def smc_w_context(mocker, mcmc_kernel):
    results = mocker.Mock()
    context_manager = mocker.patch(SAMPLER_BASE + ".ContextManager")
    context_manager.get_context.return_value = results
    return AdaptiveSampler(mcmc_kernel)


def test_context_initialize_on_restart(smc_w_context):
    smc_w_context._result = DummyResult()

    smc_w_context._initialize(num_particles=1)

    assert smc_w_context._step == 66
    assert smc_w_context._phi_sequence == [68, 67]


def test_context_initialize_no_restart(mocker, smc_w_context):
    mocker.patch.object(smc_w_context, "_initializer")
    smc_w_context._result.is_restart = False
    _ = smc_w_context._initialize(num_particles=1)
    smc_w_context._initializer.initialize_particles.assert_called_once_with(1)


@pytest.mark.parametrize("assigned, expected, write", [(None, 1, 0), (2, 2, 1)])
def test_context_step_property(smc_w_context, assigned, expected, write):
    smc_w_context._step = 1
    smc_w_context.step = assigned
    assert smc_w_context.step == expected
    if write:
        smc_w_context._result.save_step.assert_called_once_with(assigned)


def test_restart_restores_path_phi(mocker, mcmc_kernel):
    phi_sequence = [0, 0.3, 0.7, 0.9]
    last_phi = phi_sequence[-1]

    result = DummyResult()
    result.phi_sequence = phi_sequence
    mocker.patch(SAMPLER_BASE + ".InMemoryStorage", return_value=result)

    smc = AdaptiveSampler(mcmc_kernel)
    smc._result = result
    smc._initialize(num_particles=100)

    assert mcmc_kernel.path.phi == last_phi


@pytest.mark.parametrize(
    "stored_phi_sequence, provided_phi_sequence, expected_called, expected_seq",
    [
        (
            [0, 0.2, 0.5],
            [0, 0.2, 0.5, 0.7, 0.9, 1.0],
            [0.7, 0.9, 1.0],
            [0, 0.2, 0.5, 0.7, 0.9, 1.0],
        ),
        (
            [0, 0.2, 0.5],
            [0, 0.1, 0.3, 0.6, 0.8, 1.0],
            [0.6, 0.8, 1.0],
            [0, 0.2, 0.5, 0.6, 0.8, 1.0],
        ),
    ],
)
def test_fixed_phi_restart_resumes_from_last_stored_phi(
    mocker,
    mcmc_kernel,
    stored_phi_sequence,
    provided_phi_sequence,
    expected_called,
    expected_seq,
):
    result = DummyResult()
    result.phi_sequence = stored_phi_sequence
    result.estimate_marginal_log_likelihoods = mocker.Mock(return_value=[])
    mocker.patch(SAMPLER_BASE + ".InMemoryStorage", return_value=result)
    mocker.patch(SAMPLER_BASE + ".Updater")

    smc = FixedPhiSampler(mcmc_kernel, show_progress_bar=False)
    smc._result = result
    do_smc_step = mocker.patch.object(smc, "_do_smc_step")

    smc.sample(
        num_particles=10,
        num_mcmc_samples=2,
        phi_sequence=provided_phi_sequence,
        ess_threshold=0.5,
    )

    called_phis = [call.args[0] for call in do_smc_step.call_args_list]
    assert called_phis == expected_called
    assert list(smc._phi_sequence) == expected_seq
