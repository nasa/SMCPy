"""Exercise covariance-prior sampling with the real SciPy distributions."""

import numpy as np
import pytest
from scipy.stats import invwishart

from smcpy import ImproperCov, InvWishart, VectorMCMC


def make_prior(prior_class, dim, dof, scale):
    if prior_class is InvWishart:
        return InvWishart(dof, scale)
    return ImproperCov(dim, dof=dof, S=scale)


@pytest.mark.parametrize("prior_class", [InvWishart, ImproperCov])
@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("num_samples", [1, 4])
@pytest.mark.parametrize("rng_class", [np.random.default_rng, np.random.RandomState])
def test_covariance_samples_keep_axes_and_scipy_values(
    prior_class, dim, num_samples, rng_class
):
    factor = np.arange(dim * dim, dtype=float).reshape(dim, dim) / 10
    scale = factor @ factor.T + np.eye(dim)
    original_scale = scale.copy()
    dof = dim + 3
    rng, reference_rng = rng_class(173), rng_class(173)
    prior = make_prior(prior_class, dim, dof, scale)

    samples = prior.rvs(num_samples, random_state=rng)
    reference = np.asarray(
        invwishart(dof, scale).rvs(num_samples, random_state=reference_rng)
    ).reshape(num_samples, dim, dim)
    row, col = np.triu_indices(dim)

    assert samples.shape == (num_samples, dim * (dim + 1) // 2)
    np.testing.assert_array_equal(samples, reference[:, row, col])
    np.testing.assert_array_equal(scale, original_scale)
    np.testing.assert_array_equal(rng.random(5), reference_rng.random(5))


@pytest.mark.parametrize("dim", [1, 2, 3])
@pytest.mark.parametrize("num_samples", [1, 4])
def test_improper_covariance_default_distribution(dim, num_samples):
    samples = ImproperCov(dim).rvs(num_samples, random_state=42)
    reference = np.asarray(
        invwishart(dim, np.eye(dim)).rvs(num_samples, random_state=42)
    ).reshape(num_samples, dim, dim)
    row, col = np.triu_indices(dim)
    np.testing.assert_array_equal(samples, reference[:, row, col])
    assert samples.shape == (num_samples, dim * (dim + 1) // 2)


@pytest.mark.parametrize("prior_class", [InvWishart, ImproperCov])
@pytest.mark.parametrize("dim", [1, 2])
@pytest.mark.parametrize("num_samples", [1, 4])
def test_vector_mcmc_can_sample_covariance_priors(prior_class, dim, num_samples):
    prior = make_prior(prior_class, dim, dim + 3, np.eye(dim))

    def model(params):
        return params[:, :1]

    sampler = VectorMCMC(model, np.zeros(1), [prior], log_like_args=1.0)
    sampler.rng = np.random.default_rng(57)

    samples = sampler.sample_from_priors(num_samples)
    assert samples.shape == (num_samples, prior.dim)
    assert np.isfinite(samples).all()
    assert np.isfinite(sampler.evaluate_log_priors(samples)).all()


@pytest.mark.parametrize("prior_class", [InvWishart, ImproperCov])
@pytest.mark.parametrize("dim", [1, 2])
def test_covariance_sampling_without_explicit_random_state(prior_class, dim):
    prior = make_prior(prior_class, dim, dim + 3, np.eye(dim))
    samples = prior.rvs(1)
    assert samples.shape == (1, prior.dim)
    assert np.isfinite(prior.logpdf(samples)).all()
