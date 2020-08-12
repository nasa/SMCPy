import numpy as np
import pytest

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_kernel import VectorMCMCKernel

@pytest.fixture
def stub_model(mocker):
    return mocker.Mock()


@pytest.fixture
def data(mocker):
    return mocker.Mock()


@pytest.fixture
def priors(mocker):
    return mocker.Mock()


@pytest.fixture
def vector_mcmc(stub_model, data, priors):
    return VectorMCMC(stub_model, data, priors)


def test_kernel_instance(vector_mcmc, stub_model, data, priors):
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=tuple('a'))
    assert vmcmc._mcmc._eval_model == stub_model
    assert vmcmc._mcmc._data == data
    assert vmcmc._mcmc._priors == priors
    assert vmcmc._param_order == tuple('a')


def test_sample_from_prior(vector_mcmc, mocker):
    mocked_sample = np.array([[1, 2], [3, 4], [5, 6]])
    mocker.patch.object(vector_mcmc, 'sample_from_priors',
                        return_value=mocked_sample)
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=['a', 'b'])

    samples = vmcmc.sample_from_prior(num_samples=3)

    np.testing.assert_array_equal(samples['a'], mocked_sample[:, 0])
    np.testing.assert_array_equal(samples['b'], mocked_sample[:, 1])


@pytest.mark.parametrize('param_dict, expected',
                         ([{'a': 1, 'b': 2}, np.array([[1]])],
                          [{'a': [1, 1], 'b': [1, 2]}, np.array([[1], [1]])]))
def test_kernel_log_likelihoods(vector_mcmc, param_dict, expected, mocker):
    mocker.patch.object(vector_mcmc, 'evaluate_log_likelihood',
                        new=lambda x: x[:, 0].reshape(-1, 1))
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=['a', 'b'])
    
    log_likes = vmcmc.get_log_likelihoods(param_dict)
    np.testing.assert_array_equal(log_likes, expected)


@pytest.mark.parametrize('param_dict, expected',
                         ([{'a': 1, 'b': 2}, np.array([[3]])],
                          [{'a': [1, 1], 'b': [1, 2]}, np.array([[2], [3]])]))
def test_kernel_log_priors(vector_mcmc, param_dict, expected, mocker):
    mocker.patch.object(vector_mcmc, 'evaluate_log_priors',
                        new=lambda x: x)
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=['b', 'a'])
    
    log_priors = vmcmc.get_log_priors(param_dict)
    np.testing.assert_array_equal(log_priors, expected)


@pytest.mark.parametrize('num_vectorized', (1, 5))
def test_mutate_particles(vector_mcmc, num_vectorized, mocker):
    num_samples = 2
    phi = 1
    cov = np.eye(2)

    param_array = np.array([[1, 2]] * num_vectorized)
    param_dict = dict(zip(['a', 'b'], param_array.T))
    log_likes = np.ones((num_vectorized, 1)) * 1

    mocked_return = (np.array([[2, 3]] * num_vectorized),
                     np.array([[2]] * num_vectorized))
    smc_metropolis = mocker.patch.object(vector_mcmc, 'smc_metropolis',
                                         return_value=mocked_return)

    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=['a', 'b'])
    mutated = vmcmc.mutate_particles(param_dict, log_likes, num_samples,
                                     cov, phi)
    new_param_dict = mutated[0]
    new_log_likes = mutated[1]

    expected_params = {'a': np.array([2] * num_vectorized),
                       'b': np.array([3] * num_vectorized)}
    expected_log_likes = np.ones((num_vectorized, 1)) * 2

    np.testing.assert_array_equal(new_param_dict['a'], expected_params['a'])
    np.testing.assert_array_equal(new_param_dict['b'], expected_params['b'])
    np.testing.assert_array_equal(new_log_likes, expected_log_likes)

    calls = smc_metropolis.call_args[0]
    for i, expect in enumerate([param_array, num_samples, cov, phi]):
        np.testing.assert_array_equal(calls[i], expect)
