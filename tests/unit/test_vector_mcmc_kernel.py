import numpy as np
import pytest

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel


@pytest.fixture
def mcmc(mocker):
    return VectorMCMC(mocker.Mock(), mocker.Mock())


@pytest.fixture
def kernel(mcmc):
    return VectorMCMCKernel(mcmc, ['a', 'b'])


def test_conv_param_array_to_dict(kernel):
    params = np.tile([2, 1], (5, 10, 1))
    expected_dict = {'a': np.full((5, 10), 2), 'b': np.full((5, 10), 1)}

    dict_ = kernel.conv_param_array_to_dict(params)

    np.testing.assert_array_equal(dict_['a'], expected_dict['a'])
    np.testing.assert_array_equal(dict_['b'], expected_dict['b'])


def test_param_dict_to_array(kernel):
    param_dict = {'a': np.full((5, 10), 2), 'b': np.full((5, 10), 1)}
    expected_params = np.tile([2, 1], (5, 10, 1))

    params = kernel.conv_param_dict_to_array(param_dict)

    np.testing.assert_array_equal(params, expected_params)
