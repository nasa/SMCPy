import numpy as np
import pytest

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.mcmc.vector_mcmc_translator import VectorMCMCTranslator

@pytest.fixture
def stub_model(mocker):
    return mocker.Mock()


@pytest.fixture
def data(mocker):
    return mocker.Mock()


@pytest.fixture
def prior_pdfs(mocker):
    return mocker.Mock()


@pytest.fixture
def vector_mcmc(stub_model, data, prior_pdfs):
    return VectorMCMC(stub_model, data, prior_pdfs)


def test_translator_instance(vector_mcmc, stub_model, data, prior_pdfs):
    vmcmc = VectorMCMCTranslator(vector_mcmc, tuple('a'))
    assert vmcmc._mcmc._model == stub_model
    assert vmcmc._mcmc._data == data
    assert vmcmc._mcmc._prior_pdfs == prior_pdfs
    assert vmcmc._param_order == tuple('a')


@pytest.mark.parametrize('param_dict, expected',
                         ([{'a': 1, 'b': 2}, np.array([[1]])],
                          [{'a': [1, 1], 'b': [1, 2]}, np.array([[1], [1]])]))
def test_translator_log_likelihoods(vector_mcmc, param_dict, expected, mocker):
    mocker.patch.object(vector_mcmc, 'evaluate_log_likelihood',
                        new=lambda x: x[:, 0].reshape(-1, 1))
    vmcmc = VectorMCMCTranslator(vector_mcmc, param_order=['a', 'b'])
    
    log_likes = vmcmc.get_log_likelihood(param_dict)
    np.testing.assert_array_equal(log_likes, expected)


@pytest.mark.parametrize('param_dict, expected',
                         ([{'a': 1, 'b': 2}, np.array([[3]])],
                          [{'a': [1, 1], 'b': [1, 2]}, np.array([[2], [3]])]))
def test_translator_log_priors(vector_mcmc, param_dict, expected, mocker):
    mocker.patch.object(vector_mcmc, 'evaluate_log_priors',
                        new=lambda x: x)
    vmcmc = VectorMCMCTranslator(vector_mcmc, param_order=['b', 'a'])
    
    log_priors = vmcmc.get_log_prior(param_dict)
    np.testing.assert_array_equal(log_priors, expected)


@pytest.mark.parametrize('num_vectorized', (1, 5))
def test_mutate_particles(vector_mcmc, num_vectorized, mocker):
    num_samples = 2
    proposal_cov = np.eye(2)
    phi = 1

    particles = [mocker.Mock()] * num_vectorized
    for p in particles:
        p.params = {'a': 1, 'b': 2}
        p.log_like = 1

    mocked_return = (np.array([[2, 3]] * num_vectorized),
                     np.array([[2]] * num_vectorized))
    smc_metropolis = mocker.patch.object(vector_mcmc, 'smc_metropolis',
                                         return_value=mocked_return)

    vmcmc = VectorMCMCTranslator(vector_mcmc, param_order=['a', 'b'])
    mutated = vmcmc.mutate_particles(particles, num_samples, proposal_cov, phi)

    expected_input_array = np.array([[1, 2]] * num_vectorized)
    expected_particles = [mocker.Mock()] * num_vectorized
    for p in expected_particles:
        p.params = {'a': 2, 'b': 3}
        p.log_like = 2

    np.testing.assert_array_equal([p.params for p in mutated],
                                  [p.params for p in expected_particles])
    np.testing.assert_array_equal([p.log_like for p in mutated],
                                  [p.log_like for p in expected_particles])

    calls = smc_metropolis.call_args[0]
    for i, expected in enumerate([expected_input_array, num_samples,
                                  proposal_cov, phi]):
        np.testing.assert_array_equal(calls[i], expected)
