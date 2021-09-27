import numpy as np
import pytest
import sys

from smcpy.mcmc.vector_mcmc import *


@pytest.fixture
def vector_mcmc(mocker):
    return VectorMCMC(mocker.Mock(), mocker.Mock(), mocker.Mock(),
                      log_like_args=None)


@pytest.mark.parametrize('inputs,expect_log_prior',
                         ([np.ones((2, 2, 2)), np.zeros((2, 2, 1))],
                          [np.array([[[1, 1], [1, -1]], [[1, 1], [1, 1]]]),
                           np.array([[[0], [-np.inf]], [[0], [0]]])],
                          [np.array([[[1, 1], [-1, 1]], [[1, 1], [1, 1]]]),
                           np.zeros((2, 2, 1))]))
def test_vectorized_prior(vector_mcmc, inputs, expect_log_prior):
    log_prior = vector_mcmc.evaluate_log_priors(inputs)
    np.testing.assert_array_almost_equal(log_prior, expect_log_prior)


def test_vectorized_likelihood(mocker, vector_mcmc):
    inputs = np.ones((2, 2, 2))
    vector_mcmc._log_like_func = mocker.Mock()
    _ = vector_mcmc.evaluate_log_likelihood(inputs)
    vector_mcmc._log_like_func.assert_called_once_with(inputs)
    

def test_vectorized_proposal(vector_mcmc, mocker):
    inputs = np.ones((2, 5, 2)) * 0.5
    mvn_mock = mocker.patch('smcpy.mcmc.mcmc_base.np.random.normal',
                            return_value=np.ones(inputs.shape))
    cov = np.tile(np.array([[2, 0.5], [0.5, 2]]), (inputs.shape[0], 1, 1))
    expected = inputs + np.tile([1.41421356, 1.72285978], (2, 5, 1))

    inputs_new = vector_mcmc.proposal(inputs, cov=cov)

    mvn_mock.assert_called_once_with(0, 1, inputs.shape)
    np.testing.assert_array_almost_equal(inputs_new, expected)


def test_vectorized_accept_ratio(vector_mcmc, mocker):
    new_ll = np.ones((2, 5, 1))
    new_lp = np.ones((2, 5, 1))
    old_ll = np.ones((2, 5, 1))
    old_lp = np.zeros((2, 5, 1))
    exp_accpt_ratio = np.exp(np.ones((2, 5, 1)))

    accpt_ratio = vector_mcmc.acceptance_ratio(new_ll, old_ll, new_lp, old_lp)

    np.testing.assert_array_equal(accpt_ratio, exp_accpt_ratio)


def test_check_log_priors_for_zero_prob(vector_mcmc):
    log_priors = np.ones((5, 3, 1))
    log_priors[0, 2, 0] = -np.inf
    with pytest.raises(ValueError):
        vector_mcmc._check_log_priors_for_zero_probability(log_priors)


def test_get_rejected(vector_mcmc, mocker):
    accpt_ratio = np.array([[[1], [1], [.5]], [[.1], [1], [1]]])
    uniform_samples = np.array([[[.5], [.5], [.75]], [[.2], [.5], [.5]]])
    expected_rejected = np.array([[[0], [0], [1]], [[1], [0], [0]]])

    uniform = mocker.patch('smcpy.mcmc.mcmc_base.np.random.uniform',
                           return_value=uniform_samples)
    rejected = vector_mcmc.get_rejected(accpt_ratio)

    np.testing.assert_array_equal(rejected, expected_rejected)


def test_vectorized_mcmc_step(vector_mcmc, mocker):
    cov = np.ones((2, 3, 3))
    old_inputs = np.zeros((2, 5, 3))
    new_inputs = np.ones((2, 5, 3))
    old_log_like = np.zeros((2, 5, 1))
    new_log_like = np.ones((2, 5, 1))
    old_log_priors = np.zeros((2, 5, 1))
    new_log_priors = np.ones((2, 5, 1))
    phi = 0.5

    expected_rejected = np.zeros((2, 5, 1))
    expected_rejected[0, 2, 0] = 1

    expected_inputs = new_inputs.copy()
    expected_inputs[0, 2, :] = 0
    expected_log_like = new_log_like.copy()
    expected_log_like[0, 2, 0] = 0
    expected_log_priors = new_log_priors.copy()
    expected_log_priors[0, 2, 0] = 0

    vector_mcmc.proposal = mocker.Mock(return_value=new_inputs)
    vector_mcmc.evaluate_log_priors = mocker.Mock(return_value=new_log_priors)
    vector_mcmc.evaluate_log_likelihood = mocker.Mock(return_value=new_log_like)
    vector_mcmc.acceptance_ratio = mocker.Mock()
    vector_mcmc.get_rejected = mocker.Mock(return_value=expected_rejected)

    inputs, log_like, log_priors, rejected = \
        vector_mcmc.perform_mcmc_step(old_inputs, cov, old_log_like,
                                      old_log_priors, phi)

    np.testing.assert_array_equal(inputs, expected_inputs)
    np.testing.assert_array_equal(log_like, expected_log_like)
    np.testing.assert_array_equal(log_priors, expected_log_priors)
    np.testing.assert_array_equal(rejected, expected_rejected)

    vector_mcmc.proposal.called_once_with(old_inputs, cov)
    vector_mcmc.acceptance_ratio.called_once_with(new_log_like * phi,
                                    log_like * phi, new_log_priors, log_priors)


def test_vectorized_smc_metropolis(vector_mcmc, mocker):
    inputs = np.ones((2, 10, 3))
    num_samples = 3
    cov = np.tile(np.eye(3), (2, 1, 1))
    phi = 0.5
    rejected_1 = np.zeros((2, 10, 1))
    rejected_1[1, 2, 0] = 1
    rejected_1[1, 5, 0] = 1
    rejected_1[1, 7, 0] = 1
    rejected_2 = np.ones((2, 10, 1))
    rejected_3 = np.ones((2, 10, 1))

    expected_cov = cov.copy()
    expected_cov[0] *= 2 * 1/5 * 1/5
    expected_cov[1] *= 1/5 * 1/5

    vector_mcmc._initialize_probabilities = mocker.Mock(return_value=[1, 2])
    vector_mcmc.perform_mcmc_step = mocker.Mock(side_effect=\
                                                 [(inputs, 1, 2, rejected_1),
                                                  (inputs, 1, 2, rejected_2),
                                                  (inputs, 1, 2, rejected_3)])

    x, y = vector_mcmc.smc_metropolis(inputs, num_samples, cov, phi)

    np.testing.assert_array_equal(x, inputs)
    np.testing.assert_array_equal(y, 1)
    np.testing.assert_array_almost_equal(cov, expected_cov)
    assert len(vector_mcmc.perform_mcmc_step.call_args_list) == num_samples
    vector_mcmc._initialize_probabilities.called_once_with(inputs)
