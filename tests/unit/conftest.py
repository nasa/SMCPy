import pytest

from smcpy.mcmc.translator_base import Translator


@pytest.fixture
def stub_comm(mocker):
    comm = mocker.Mock()
    comm.gather = lambda x, root: [x]
    return comm


@pytest.fixture
def stub_mcmc_kernel(mocker):
    stub_mcmc_kernel = mocker.Mock(Translator)
    mocker.patch.object(stub_mcmc_kernel, 'sample_from_prior', create=True,
                        return_value={'a': [1, 1, 1, 2, 2]})
    mocker.patch.object(stub_mcmc_kernel, 'get_log_likelihood', create=True,
                        side_effect=[0.1, 0.1, 0.1, 0.2, 0.2])
    mocker.patch.object(stub_mcmc_kernel, 'get_log_prior', create=True,
                        side_effect=[0.2, 0.2, 0.2, 0.3, 0.3])
    mocker.patch.object(stub_mcmc_kernel, 'sample', create=True)
    mocker.patch.object(stub_mcmc_kernel, 'get_final_trace_values', create=True,
                        side_effect=[{'a': 1, 'b': 3},
                                     {'a': 2, 'b': 4},
                                     {'a': 5, 'b': 6}])
    return stub_mcmc_kernel
