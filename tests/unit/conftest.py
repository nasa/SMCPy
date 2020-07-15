import numpy as np
import pytest

from smcpy.mcmc.kernel_base import MCMCKernel


@pytest.fixture
def stub_comm(mocker):
    comm = mocker.Mock()
    comm.gather = lambda x, root: [x]
    return comm


@pytest.fixture
def stub_mcmc_kernel(mocker):
    stub_mcmc_kernel = mocker.Mock(MCMCKernel)
    mocker.patch.object(stub_mcmc_kernel, 'get_log_likelihoods', create=True,
                        return_value=np.array([0.1, 0.1, 0.1, 0.2, 0.2]))
    mocker.patch.object(stub_mcmc_kernel, 'get_log_priors', create=True,
                        return_value=np.array([0.2, 0.2, 0.2, 0.3, 0.3]))
    mocker.patch.object(stub_mcmc_kernel, 'sample_from_prior', create=True,
                        return_value={'a': np.array([1, 1, 1, 2, 2]),
                                      'b': np.array([2, 2, 2, 3, 3])})
    return stub_mcmc_kernel
