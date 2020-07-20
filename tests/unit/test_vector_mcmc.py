import numpy as np
import pytest
import sys

from smcpy.mcmc.vector_mcmc import *



class DummyPrior:

    def __init__(self, pdf_func, sample_func):
        self._pdf = pdf_func
        self._rvs = sample_func

    @property
    def pdf(self):
        return self._pdf

    @property
    def rvs(self):
        return self._rvs


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
def prior_samplers():
    return [lambda x: np.array([1] * x), lambda x: np.array([2] * x),
            lambda x: np.array([3] * x)]


@pytest.fixture
def priors(prior_pdfs, prior_samplers):
    priors = []
    for i, funcs in enumerate(zip(prior_pdfs, prior_samplers)):
        priors.append(DummyPrior(*funcs))
    return priors


@pytest.fixture
def vector_mcmc(stub_model, data, priors):
    return VectorMCMC(stub_model, data, priors, std_dev=None)


def test_vectorized_prior_sampling(vector_mcmc):
    prior_samples = vector_mcmc.sample_from_priors(num_samples=4)

    np.testing.assert_array_equal(prior_samples, np.tile([1, 2, 3], (4, 1)))


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
    cov_scale = 1 #2.38 ** 2 / inputs.shape[1]
    expected = inputs + 1

    inputs_new = vector_mcmc.proposal(inputs, cov=cov)

    calls = mvn_mock.call_args[0]
    np.testing.assert_array_equal(inputs_new, expected)
    np.testing.assert_array_equal(calls[0], np.zeros(cov.shape[0]))
    np.testing.assert_array_equal(calls[1], cov_scale * cov)
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


@pytest.mark.parametrize('adapt_interval,expected_cov',
                         ([3, np.eye(2)], [4, np.array([[2, 1], [1, 3]])]))
def test_vectorized_proposal_adaptation(vector_mcmc, adapt_interval,
                                        expected_cov, mocker):
    num_parallel_chains = 3
    current_sample_count = 8
    num_samples = 10
    old_cov = np.eye(2)
    chain = np.zeros([num_parallel_chains, 2, num_samples])
    mocker.patch('numpy.cov', return_value=np.array([[2, 1], [1, 3]]))

    cov = vector_mcmc.adapt_proposal_cov(old_cov, chain, current_sample_count,
                                         adapt_interval)

    np.testing.assert_array_equal(cov, expected_cov)


@pytest.mark.parametrize('phi', (0.5, 1))
@pytest.mark.parametrize('num_samples', (1, 2))
def test_vectorized_smc_metropolis(vector_mcmc, phi, num_samples, mocker):
    inputs = np.ones([10, 3])
    cov = np.eye(3)
    vector_mcmc._std_dev = 1
    vector_mcmc._priors = [mocker.Mock(DummyPrior, autospec=True),
                           mocker.Mock(DummyPrior, autospec=True),
                           mocker.Mock(DummyPrior, autospec=True)]

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
    expected_new_log_likes = inputs * 2

    np.testing.assert_array_equal(new_inputs, expected_new_inputs)
    np.testing.assert_array_equal(new_log_likes, expected_new_log_likes)

    assert log_like.call_count == num_samples + 1
    assert vector_mcmc._priors[0].pdf.call_count == num_samples + 1


@pytest.mark.parametrize('adapt_delay', (0, 2))
@pytest.mark.parametrize('adapt_interval', (None, 1))
@pytest.mark.parametrize('num_samples', (1, 5))
def test_vectorized_metropolis(vector_mcmc, num_samples, adapt_interval,
                               adapt_delay, mocker):
    inputs = np.ones([10, 3])
    cov = np.eye(3)
    vector_mcmc._std_dev = 1
    vector_mcmc._priors = [mocker.Mock(DummyPrior, autospec=True),
                           mocker.Mock(DummyPrior, autospec=True),
                           mocker.Mock(DummyPrior, autospec=True)]

    mocker.patch('numpy.random.uniform')

    mocker.patch('numpy.random.uniform')
    mocker.patch.object(vector_mcmc, 'acceptance_ratio', return_value=inputs)
    mocker.patch.object(vector_mcmc, 'proposal',
                    side_effect=[inputs + i for i in range(1, num_samples + 1)])
    mocker.patch.object(vector_mcmc, 'selection',
                        new=lambda new_log_like, x, y, z: new_log_like)
    log_like = mocker.patch.object(vector_mcmc, 'evaluate_log_likelihood')
    adapt = mocker.patch.object(vector_mcmc, 'adapt_proposal_cov')

    expected_chain = np.zeros([10, 3, num_samples + 1])
    expected_chain[:, :, 0] = inputs
    for i in range(num_samples):
        expected_chain[:, :, i + 1] = expected_chain[:, :, i].copy() + 1

    chain = vector_mcmc.metropolis(inputs, num_samples, cov, adapt_interval,
                                   adapt_delay)

    np.testing.assert_array_equal(chain, expected_chain)

    num_expected_adapt_calls = 0
    if num_samples > adapt_delay:
        num_expected_adapt_calls = num_samples - adapt_delay

    assert log_like.call_count == num_samples + 1
    assert vector_mcmc._priors[0].pdf.call_count == num_samples + 1
    assert adapt.call_count == num_expected_adapt_calls
