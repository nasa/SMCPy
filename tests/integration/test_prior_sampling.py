import numpy as np
import pytest

from scipy.stats import invwishart

from smcpy.mcmc.vector_mcmc import VectorMCMC
from smcpy.priors import ImproperUniform, InvWishart


@pytest.mark.parametrize('insert_index', [0, 1, 2])
def test_prior_sampling(mocker, insert_index):
    n_samples = 10
    dim = 3
    n_cov_terms = int(dim * (dim + 1) / 2)

    np.random.seed(1)
    cov_sample = invwishart(dim, np.eye(dim)).rvs(n_samples)
    exp_cov_samples = []
    for cov in cov_sample:
        x = np.triu(cov)
        x = x[abs(x) > 0]
        exp_cov_samples.append(x)
    cov_sample = np.array(exp_cov_samples)

    np.random.seed(1)

    smcpy_prior = InvWishart(dim, np.eye(dim))

    model = mocker.Mock()
    data = mocker.Mock()
    mock_prior = mocker.Mock()
    mock_prior.rvs.return_value = np.ones((n_samples, 1))

    priors = [mock_prior, mock_prior]
    priors.insert(insert_index, smcpy_prior)

    expect_sample = np.ones((n_samples, 2 + n_cov_terms))
    expect_sample[:, insert_index:insert_index + n_cov_terms] = cov_sample

    vmcmc = VectorMCMC(model, data, priors)

    samples = vmcmc.sample_from_priors(n_samples)
    np.testing.assert_array_equal(samples, expect_sample)
