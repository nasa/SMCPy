import numpy as np
import pytest

from smcpy import FixedSampler, AdaptiveSampler
from smcpy.mcmc.kernel_base import MCMCKernel


@pytest.fixture
def mcmc_kernel(mocker):
    return mocker.Mock(MCMCKernel)


@pytest.fixture
def smc_w_context(mocker, mcmc_kernel):
    results = mocker.Mock()
    context_manager = mocker.patch('smcpy.sampler_base.ContextManager')
    context_manager.get_context.return_value = results
    return AdaptiveSampler(mcmc_kernel)


def test_context_initialize_on_restart(smc_w_context):
    smc_w_context._result.is_restart.return_value = True
    smc_w_context._result.load_for_restart.return_value = [68, 67]

    smc_w_context._initialize(num_particles=1, proposal=None)

    smc_w_context._result.load_for_restart.assert_called_once()
    assert smc_w_context._step == 68
    assert smc_w_context._phi_sequence == 67


def test_context_initialize_no_restart(mocker, smc_w_context):
    mocker.patch.object(smc_w_context, '_initializer')
    smc_w_context._result.is_restart = False
    _ = smc_w_context._initialize(num_particles=1, proposal=None)
    smc_w_context._initializer.init_particles_from_prior.assert_called_once()


@pytest.mark.parametrize('assigned, expected, write', [(None, 1, 0), (2, 2, 1)])
def test_context_step_property(smc_w_context, assigned, expected, write):
    smc_w_context._step = 1
    smc_w_context.step = assigned
    assert smc_w_context.step == expected
    if write:
        smc_w_context._result.save_step.assert_called_once_with(assigned)
