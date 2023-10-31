import numpy as np
import pytest

from smcpy import FixedSampler, AdaptiveSampler
from smcpy.mcmc.kernel_base import MCMCKernel

SAMPLER_BASE = 'smcpy.smc.sampler_base'

class DummyResult:

    def __init__(self):
        self.phi_sequence = [68, 67]
        self.is_restart = True

    def __getitem__(self, idx):
        return 66


@pytest.fixture
def mcmc_kernel(mocker):
    return mocker.Mock(MCMCKernel)


@pytest.fixture
def smc_w_context(mocker, mcmc_kernel):
    results = mocker.Mock()
    context_manager = mocker.patch(SAMPLER_BASE + '.ContextManager')
    context_manager.get_context.return_value = results
    return AdaptiveSampler(mcmc_kernel)


def test_context_initialize_on_restart(smc_w_context):
    smc_w_context._result = DummyResult()

    smc_w_context._initialize(num_particles=1)

    assert smc_w_context._step == 66
    assert smc_w_context._phi_sequence == [68, 67]


def test_context_initialize_no_restart(mocker, smc_w_context):
    mocker.patch.object(smc_w_context, '_initializer')
    smc_w_context._result.is_restart = False
    _ = smc_w_context._initialize(num_particles=1)
    smc_w_context._initializer.initialize_particles.assert_called_once_with(1)


@pytest.mark.parametrize('assigned, expected, write', [(None, 1, 0), (2, 2, 1)])
def test_context_step_property(smc_w_context, assigned, expected, write):
    smc_w_context._step = 1
    smc_w_context.step = assigned
    assert smc_w_context.step == expected
    if write:
        smc_w_context._result.save_step.assert_called_once_with(assigned)
