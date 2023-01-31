import numpy as np
import pytest
import sys

from smcpy.mcmc.vector_mcmc import *


@pytest.fixture
def stub_model():
    x = np.array([1, 2, 3])

    def evaluation(q):
        assert q.shape[1] == 3
        output = (q[:, 0, None] * 2 + 3.25 * q[:, 1, None] - \
                  q[:, 2, None] ** 2) * x
        return output

    return evaluation


@pytest.fixture
def data():
    return np.array([5, 4, 9])


@pytest.fixture
def priors(mocker):
    priors = [mocker.Mock() for _ in range(3)]
    for i, p in enumerate(priors):
        p.rvs = lambda x, i=i: np.array([1 + i] * x)
        p.pdf = lambda x, i=i: x + i
        delattr(p, 'dim')

    multivar_prior = mocker.Mock()
    multivar_prior.rvs = lambda x: np.tile([4, 5, 6], (x, 1))
    multivar_prior.pdf = lambda x: np.sum(x - 0.5, axis=1)
    multivar_prior.dim = 3
    priors.append(multivar_prior)

    return priors


@pytest.fixture
def vector_mcmc(stub_model, data, priors):
    return VectorMCMC(stub_model, data, priors, log_like_args=None)


@pytest.mark.parametrize('num_samples', [1, 2, 4])
def test_vectorized_prior_sampling(vector_mcmc, num_samples):
    prior_samples = vector_mcmc.sample_from_priors(num_samples=num_samples)
    expected_samples = np.tile([1, 2, 3, 4, 5, 6], (num_samples, 1))
    np.testing.assert_array_equal(prior_samples, expected_samples)


@pytest.mark.parametrize('inputs', (np.array(
    [[0.1, 1, 0.5, 3, 2, 1]]), np.array([[0.1, 1, 0.5, 3, 2, 1]] * 4)))
def test_vectorized_prior(vector_mcmc, inputs):
    log_prior = vector_mcmc.evaluate_log_priors(inputs)
    expected_prior = np.array([[0.1, 2, 2.5, 4.5]] * inputs.shape[0])
    np.testing.assert_array_almost_equal(log_prior, np.log(expected_prior))


@pytest.mark.parametrize(
    'inputs', (np.array([[0.1, 0.5]]), np.array([[0.1, 1, 1, 0.5]] * 4)))
def test_prior_input_mismatch_throws_error(vector_mcmc, inputs):
    with pytest.raises(ValueError):
        vector_mcmc.evaluate_log_priors(inputs)


@pytest.mark.parametrize(
    'inputs, std_dev',
    (((np.array([[0, 1, 0.5]]), 1 / np.sqrt(2 * np.pi)),
      (np.array([[0, 1, 0.5]] * 4), 1 / np.sqrt(2 * np.pi)),
      (np.array([[0, 1, 0.5, 1 / np.sqrt(2 * np.pi)]] * 4), None))))
def test_vectorized_default_likelihood(vector_mcmc, inputs, std_dev):
    vector_mcmc._log_like_func._args = std_dev

    expected_like = np.array(inputs.shape[0] * [[np.exp(-8 * np.pi)]])
    expected_log_like = np.log(expected_like)

    log_like = vector_mcmc.evaluate_log_likelihood(inputs)

    np.testing.assert_array_almost_equal(log_like, expected_log_like)


@pytest.mark.parametrize('inputs',
                         (np.array([[0, 1, 0.5]]), np.array([[0, 1, 0.5]] * 4))
                         )
def test_vectorized_proposal(vector_mcmc, inputs, mocker):
    chol_mock = mocker.patch('numpy.linalg.cholesky', return_value=2)
    norm_mock = mocker.patch('numpy.random.normal', return_value=np.array([1]))
    matmul_mock = mocker.patch('numpy.matmul',
                               return_value=np.ones(inputs.shape).T)

    n_param = inputs.shape[1]
    cov = np.eye(n_param)
    expected_proposal = inputs + 1

    proposal = vector_mcmc.proposal(inputs, cov=cov)

    np.testing.assert_array_equal(proposal, expected_proposal)
    chol_mock.assert_called_once_with(cov)
    norm_mock.assert_called_once_with(0, 1, inputs.shape)
    matmul_mock.assert_called_once_with(2, 1)


@pytest.mark.filterwarnings('ignore: Covariance matrix is')
def test_vectorized_proposal_non_psd_cov(vector_mcmc, mocker):
    inputs = np.ones((10, 2))
    eigval = np.array([-0.25, 2.5])
    eigvec = np.array([[0.4, -0.9], [-0.9, -0.4]])
    cov = mocker.Mock()
    expected_cov = np.array([[2.025, 0.9], [0.9, 0.4]]) + 1e-14 * np.eye(2)

    chol_mock = mocker.patch('numpy.linalg.cholesky',
                             side_effect=(np.linalg.linalg.LinAlgError,
                                          np.eye(2)))
    mocker.patch('numpy.linalg.eigh', return_value=(eigval, eigvec))

    _ = vector_mcmc.proposal(inputs, cov=cov)

    assert chol_mock.call_args_list[0].args[0] == cov
    np.testing.assert_array_equal(chol_mock.call_args_list[1].args[0],
                                  expected_cov)


@pytest.mark.parametrize(
    'new_inputs, old_inputs',
    ((np.array([[1, 1, 1]]), np.array([[2, 2, 2]])),
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


def test_vectorized_get_rejections(vector_mcmc, mocker):
    acceptance_ratios = np.c_[[0.25, 1.2, 0.25, 0.75]]
    uniform_samples = np.full((4, 1), 0.5)
    uniform_mock = mocker.patch('smcpy.mcmc.vector_mcmc.np.random.uniform',
                                return_value=uniform_samples)
    expected = np.c_[[True, False, True, False]]

    rejected = vector_mcmc.get_rejections(acceptance_ratios)

    uniform_mock.assert_called_once()
    np.testing.assert_array_equal(expected, rejected)


@pytest.mark.parametrize('adapt_interval,adapt_delay,adapt', [(3, 0, False),
                                                              (4, 0, True),
                                                              (8, 0, True),
                                                              (11, 0, False),
                                                              (3, 5, True),
                                                              (4, 5, False),
                                                              (8, 5, False),
                                                              (2, 1, False),
                                                              (3, 2, True),
                                                              (None, 1, False),
                                                              (2, 8, True)])
def test_vectorized_proposal_adaptation(vector_mcmc, adapt_interval, adapt,
                                        adapt_delay, mocker):
    parallel_chains = 3
    sample_count = 8
    total_samples = sample_count + 1
    old_cov = np.eye(2)
    chain = np.zeros([parallel_chains, 2, total_samples])
    cov_mock = mocker.patch('numpy.cov', return_value=np.eye(2) * 2)

    cov = vector_mcmc.adapt_proposal_cov(old_cov, chain, sample_count,
                                         adapt_interval, adapt_delay)
    expected_cov = old_cov

    if adapt:
        expected_cov = np.eye(2) * 2
        if sample_count > adapt_delay:
            n_samples_for_cov_calc = (sample_count - adapt_delay)
        else:
            n_samples_for_cov_calc = adapt_interval
        expected_call = np.zeros((2, parallel_chains * n_samples_for_cov_calc))

        np.testing.assert_array_equal(cov_mock.call_args[0][0], expected_call)
    np.testing.assert_array_equal(cov, expected_cov)


@pytest.mark.parametrize('phi', (0.5, 1))
@pytest.mark.parametrize('num_samples', (1, 2))
@pytest.mark.parametrize('num_accepted', (0, 5, 10))
def test_vectorized_smc_metropolis(vector_mcmc, phi, num_samples, num_accepted,
                                   mocker):
    inputs = np.ones([10, 3])
    cov = np.eye(3)
    rejected = np.array([True] * (10 - num_accepted) + [False] * num_accepted)

    pobj = mocker.patch.object
    prior_mock = pobj(vector_mcmc, 'evaluate_log_priors')
    chk_prior_mock = pobj(vector_mcmc,
                          '_check_log_priors_for_zero_probability')
    like_nz_mock = pobj(vector_mcmc, '_eval_log_like_if_prior_nonzero')
    like_mock = pobj(vector_mcmc, 'evaluate_log_likelihood')
    prop_mock = pobj(vector_mcmc, 'proposal')
    accpt_mock = pobj(vector_mcmc, 'acceptance_ratio')
    reject_mock = pobj(vector_mcmc, 'get_rejections', return_value=rejected)

    mocker.patch('smcpy.mcmc.vector_mcmc.np.where', new=lambda x, y, z: y)

    vector_mcmc.smc_metropolis(inputs, num_samples, cov, phi)

    assert like_mock.call_count == 1
    assert chk_prior_mock.call_count == 1
    assert like_nz_mock.call_count == num_samples
    assert prior_mock.call_count == num_samples + 1
    assert prop_mock.call_count == num_samples
    assert accpt_mock.call_count == num_samples
    assert reject_mock.call_count == num_samples

    expected_cov = np.eye(3)
    for i, call in enumerate(prop_mock.call_args_list):
        np.testing.assert_array_equal(call[0][1], expected_cov)
        if num_accepted > 7:
            expected_cov *= 2
        if num_accepted < 2:
            expected_cov *= 1 / 5


@pytest.mark.parametrize('num_samples', (1, 2))
def test_vectorized_metropolis(vector_mcmc, num_samples, mocker):
    inputs = np.ones([10, 3])
    cov = np.eye(3)
    adapt_delay = 0
    adapt_interval = 1
    expected_chain = np.ones([10, 3, num_samples + 1])
    mock = mocker.Mock()

    init_mock = mocker.patch.object(vector_mcmc,
                                    '_initialize_probabilities',
                                    return_value=(mock, mock))
    step_mock = mocker.patch.object(vector_mcmc,
                                    '_perform_mcmc_step',
                                    return_value=(inputs, mock, mock, mock))
    adapt_mock = mocker.patch.object(vector_mcmc, 'adapt_proposal_cov')

    chain = vector_mcmc.metropolis(inputs, num_samples, cov, adapt_interval,
                                   adapt_delay)

    np.testing.assert_array_equal(chain, expected_chain)
    init_mock.assert_called_once()
    step_mock.call_count == num_samples


@pytest.mark.parametrize('num_chains', (1, 3, 5))
@pytest.mark.parametrize('method', ('smc_metropolis', 'metropolis'))
def test_metropolis_inputs_out_of_bounds(mocker, stub_model, data, num_chains,
                                         method):
    vmcmc = VectorMCMC(stub_model, data, priors, log_like_args=1)
    mocker.patch.object(vmcmc,
                        'evaluate_log_priors',
                        return_value=np.ones((num_chains, 1)) * -np.inf)

    with pytest.raises(ValueError):
        vmcmc.__getattribute__(method)(np.ones((1, 3)),
                                       num_samples=0,
                                       cov=None,
                                       phi=None)


def test_multi_dim_priors(mocker):
    n_priors = 5
    n_input_params = 7
    n_inputs = 3

    inputs = np.tile(np.arange(n_input_params), (n_inputs, 1))
    priors = [mocker.Mock() for i in range(n_priors)]
    expected_priors = np.zeros((inputs.shape[0], n_priors))
    expected_prior_calls = [
        inputs[:, 0], inputs[:, 1], inputs[:, 2], inputs[:, 3:6], inputs[:, 6]
    ]

    for p in priors:
        p.pdf.return_value = np.array([[1] * inputs.shape[0]])
        p.dim = 1
    priors[3].dim = 3

    mcmc = VectorMCMC(mocker.Mock(), mocker.Mock(), priors)

    np.testing.assert_array_equal(mcmc.evaluate_log_priors(inputs),
                                  expected_priors)
    for i, exp_call in enumerate(expected_prior_calls):
        np.testing.assert_array_equal(priors[i].pdf.call_args[0][0],
                                      exp_call.reshape(n_inputs, -1))


def test_metropolis_no_like_calc_if_zero_prior_prob(mocker, data):
    mocked_model = mocker.Mock(
        side_effect=[np.ones((5, 3)), np.tile(data, (3, 1))])
    input_params = np.ones((5, 3)) * 0.1

    some_zero_priors = np.ones((5, 3)) * 2
    some_zero_priors[1, 0] = -np.inf
    some_zero_priors[4, 2] = -np.inf

    mocked_proposal = np.ones((5, 3)) * 0.2
    mocked_proposal[2, 0] = 0.3

    vmcmc = VectorMCMC(mocked_model, data, priors=None, log_like_args=1)
    mocker.patch.object(vmcmc, '_check_log_priors_for_zero_probability')
    mocker.patch.object(vmcmc, 'proposal', return_value=mocked_proposal)
    mocker.patch.object(vmcmc,
                        'evaluate_log_priors',
                        side_effect=[np.ones((5, 3)), some_zero_priors])

    expected_chain = np.zeros((5, 3, 2))
    expected_chain[:, :, 0] = input_params
    expected_chain[:, :, 1] = mocked_proposal
    expected_chain[1, :, 1] = expected_chain[1, :, 0]
    expected_chain[4, :, 1] = expected_chain[1, :, 0]

    chain = vmcmc.metropolis(input_params, num_samples=1, cov=np.eye(3))

    np.testing.assert_array_equal(mocked_model.call_args[0][0],
                                  mocked_proposal[[0, 2, 3]])
    np.testing.assert_array_equal(chain, expected_chain)
