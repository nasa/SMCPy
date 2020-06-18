import numpy as np
import pytest
import sys

from smcpy.mcmc.vector_mcmc import *

@pytest.fixture
def stub_model():
    x = np.array([1, 2, 3])
    def evaluation(q):
        output = (q[:, 0, None] * 2 + 3.25 * q[:, 1, None] - \
                  q[:, 2, None] ** 2) * x
        return output
    return evaluation


@pytest.fixture
def data():
    return np.array([5, 4, 9])


@pytest.fixture
def prior_pdfs():
    return [lambda x: x, lambda x: x + 1, lambda x: x *2]


@pytest.fixture
def vector_mcmc(stub_model, data, prior_pdfs):
    return VectorMCMC(stub_model, data, prior_pdfs, std_dev=None)


@pytest.mark.parametrize('inputs, std_dev', (
                      ((np.array([[0, 1, 0.5]]), 1/np.sqrt(2 * np.pi)),
                       (np.array([[0, 1, 0.5]] * 4), 1/np.sqrt(2 * np.pi)),
                       (np.array([[0, 1, 0.5, 1/np.sqrt(2 * np.pi)]] * 4), None)
                      )))
def test_vectorized_likelihood(vector_mcmc, inputs, std_dev):
    vector_mcmc._fixed_std_dev = std_dev

    expected_like = np.array(inputs.shape[0] * [[np.exp(-8 * np.pi)]])
    expected_log_like = np.log(expected_like)

    log_like = vector_mcmc.evaluate_log_likelihood(inputs)

    np.testing.assert_array_almost_equal(log_like, expected_log_like)


@pytest.mark.parametrize('inputs', (np.array([[0.1, 1, 0.5]]),
                                    np.array([[0.1, 1, 0.5]] * 4)))
def test_vectorized_prior(vector_mcmc, inputs):
    log_prior = vector_mcmc.evaluate_log_priors(inputs)

    expected_log_prior = np.log(np.array([[0.1, 2, 1]] * inputs.shape[0]))
    np.testing.assert_array_almost_equal(log_prior, expected_log_prior)


@pytest.mark.parametrize('inputs', (np.array([[0, 1, 0.5]]),
                                    np.array([[0, 1, 0.5]] * 4)))
def test_vectorized_proposal(vector_mcmc, inputs, mocker):
    mvn_mock = mocker.patch('numpy.random.multivariate_normal',
                            return_value=np.ones(inputs.shape))
    cov = np.eye(inputs.shape[1])
    expected = inputs + 1

    inputs_new = vector_mcmc.proposal(inputs, cov=cov)

    calls = mvn_mock.call_args[0]
    np.testing.assert_array_equal(inputs_new, expected)
    np.testing.assert_array_equal(calls[0], np.zeros(cov.shape[0]))
    np.testing.assert_array_equal(calls[1], cov)
    np.testing.assert_array_equal(calls[2], inputs.shape[0])


@pytest.mark.parametrize('new_inputs, old_inputs', (
                        (np.array([[1, 1, 1]]), np.array([[2, 2, 2]])),
                        (np.array([[1, 1, 1]] * 4), np.array([[2, 2, 2]] * 4))))
def test_vectorized_accept_ratio(vector_mcmc, new_inputs, old_inputs, mocker):
    mocked_new_log_likelihood = np.ones([new_inputs.shape[0], 1])
    mocked_old_log_likelihood = np.ones([new_inputs.shape[0], 1])
    mocked_new_log_priors = new_inputs
    mocked_old_log_priors = old_inputs

    accpt_ratio = vector_mcmc.acceptance_ratio(mocked_new_log_likelihood,
                                               mocked_old_log_likelihood,
                                               mocked_new_log_priors,
                                               mocked_old_log_priors)

    expected = np.exp(np.array([[4 - 7.]] * new_inputs.shape[0]))
    np.testing.assert_array_equal(accpt_ratio, expected)


@pytest.mark.parametrize('new_inputs, old_inputs', (
                        (np.array([[1, 1, 1]]), np.array([[2, 2, 2]])),
                        (np.array([[1, 1, 1]] * 4), np.array([[2, 2, 2]] * 4))))
def test_vectorized_selection(vector_mcmc, new_inputs, old_inputs, mocker):
    mocked_uniform_samples = np.ones([new_inputs.shape[0], 1]) * 0.5
    acceptance_ratios = np.c_[[0.25, 1.2, 0.25, 0.75]][:new_inputs.shape[0]]

    accepted_inputs = vector_mcmc.selection(new_inputs, old_inputs,
                                            acceptance_ratios,
                                            mocked_uniform_samples)

    expected = np.array([[2, 2, 2], [1, 1, 1], [2, 2, 2], [1, 1, 1]])
    np.testing.assert_array_equal(accepted_inputs,
                                  expected[:new_inputs.shape[0]])


@pytest.mark.parametrize('phi', (0.5, 1))
@pytest.mark.parametrize('num_samples', (1, 2))
def test_vectorized_smc_metropolis(vector_mcmc, phi, num_samples, mocker):
    inputs = np.ones([10, 3])
    cov = np.eye(3)
    vector_mcmc._std_dev = 1
    vector_mcmc._prior_pdfs = [mocker.Mock(), mocker.Mock(), mocker.Mock()]

    mocker.patch('numpy.random.uniform')
    mocker.patch.object(vector_mcmc, 'acceptance_ratio', return_value=inputs)
    mocker.patch.object(vector_mcmc, 'proposal',
                        side_effect=[inputs + 1, inputs + 2])
    mocker.patch.object(vector_mcmc, 'selection',
                        new=lambda new_log_like, x, y, z: new_log_like)
    log_like = mocker.patch.object(vector_mcmc, 'evaluate_log_likelihood',
                                   return_value = inputs * 2)

    new_inputs, new_log_likes = vector_mcmc.smc_metropolis(inputs, num_samples,
                                                           cov, phi)

    expected_new_inputs = inputs + num_samples
    expected_new_log_likes = inputs * 2 * phi

    np.testing.assert_array_equal(new_inputs, expected_new_inputs)
    np.testing.assert_array_equal(new_log_likes, expected_new_log_likes)

    assert log_like.call_count == num_samples + 1
    assert vector_mcmc._prior_pdfs[0].call_count == num_samples + 1


@pytest.mark.parametrize('num_samples', (1, 5))
def test_vectorized_smc_metropolis(vector_mcmc, num_samples, mocker):
    inputs = np.ones([10, 3])
    cov = np.eye(3)
    vector_mcmc._std_dev = 1
    vector_mcmc._prior_pdfs = [mocker.Mock(), mocker.Mock(), mocker.Mock()]

    mocker.patch('numpy.random.uniform')
    mocker.patch.object(vector_mcmc, 'acceptance_ratio', return_value=inputs)
    mocker.patch.object(vector_mcmc, 'proposal',
                    side_effect=[inputs + i for i in range(1, num_samples + 1)])
    mocker.patch.object(vector_mcmc, 'selection',
                        new=lambda new_log_like, x, y, z: new_log_like)
    log_like = mocker.patch.object(vector_mcmc, 'evaluate_log_likelihood')

    expected_chain = np.zeros([10, 3, num_samples + 1])
    expected_chain[:, :, 0] = inputs
    for i in range(num_samples):
        expected_chain[:, :, i + 1] = expected_chain[:, :, i].copy() + 1

    chain = vector_mcmc.metropolis(inputs, num_samples, cov)

    np.testing.assert_array_equal(chain, expected_chain)

    assert log_like.call_count == num_samples + 1
    assert vector_mcmc._prior_pdfs[0].call_count == num_samples + 1
