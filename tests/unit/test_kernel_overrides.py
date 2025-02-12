import pytest

from smcpy import VectorMCMCKernel
from smcpy.mcmc.vector_mcmc import *
from smcpy.smc.sampler_base import SamplerBase
from smcpy.paths import GeometricPath


def test_vectormcmc_kernel_no_overrride():
    vmcmc = VectorMCMC(model=None, data=None, priors=None)
    original_method = vmcmc.evaluate_log_posterior
    kernel = VectorMCMCKernel(vmcmc, param_order=["a", "b"], path=None)
    kernel._mcmc.evaluate_log_posterior = original_method

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    log_like = np.full_like(inputs[:, [0]], 2)
    log_priors = np.ones_like(inputs)

    expected_post = np.full(log_like.shape[0], 4)

    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    np.testing.assert_array_equal(post, expected_post)


def test_vectormcmc_kernel_geometric_path_overrride():
    path = GeometricPath()

    vmcmc = VectorMCMC(model=None, data=None, priors=None)
    kernel = VectorMCMCKernel(vmcmc, param_order=["a", "b"], path=path)

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    log_like = np.full_like(inputs[:, [0]], 2)
    log_priors = np.ones_like(inputs)

    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    np.testing.assert_array_equal(post, np.full(log_like.shape, 2))


def test_vectormcmc_kernel_geometric_path_overrride_manual_updates():
    path = GeometricPath()

    vmcmc = VectorMCMC(model=None, data=None, priors=None)
    kernel = VectorMCMCKernel(vmcmc, param_order=["a", "b"], path=path)

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    log_like = np.full_like(inputs[:, [0]], 2)
    log_priors = np.ones_like(inputs)

    path.phi = 0.5
    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    np.testing.assert_array_equal(post, np.full(log_like.shape, 3))

    path.phi = 1
    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    np.testing.assert_array_equal(post, np.full(log_like.shape, 4))


def test_smc_step_updates_phi(mocker):
    path = GeometricPath()
    kernel = VectorMCMCKernel(mocker.Mock(), ["a", "b"], path=path)

    sampler = SamplerBase(mcmc_kernel=kernel)
    mocker.patch.object(sampler, "_updater")
    mocker.patch.object(sampler, "_mutator")
    mocker.patch.object(sampler, "_compute_mutation_ratio")
    sampler.step = 1

    sampler._do_smc_step(phi=0.1, num_mcmc_samples=1)

    assert path.phi == 0.1


def test_vectormcmc_kernel_phi_update_must_be_monotonic(mocker):
    path = GeometricPath()
    path.phi = 0.5
    kernel = VectorMCMCKernel(mocker.Mock(), ["a", "b"], path=path)

    sampler = SamplerBase(mcmc_kernel=kernel)
    mocker.patch.object(sampler, "_updater")
    mocker.patch.object(sampler, "_mutator")
    mocker.patch.object(sampler, "_compute_mutation_ratio")
    sampler.step = 1

    with pytest.raises(ValueError):
        sampler._do_smc_step(phi=0.49, num_mcmc_samples=1)


def test_vectormcmc_kernel_default_path_is_geometric(mocker):
    kernel = VectorMCMCKernel(mocker.Mock(), ["a", "b"], path=None)

    assert kernel.path.phi == 0
    assert isinstance(kernel.path, GeometricPath)


def test_vectormcmc_kernel_geometric_path_overrride_with_proposal(mocker):
    mock_proposal = mocker.Mock()
    path = GeometricPath(proposal=mock_proposal)
    path.phi = 0.2

    vmcmc = VectorMCMC(model=None, data=None, priors=None)
    kernel = VectorMCMCKernel(vmcmc, param_order=["a", "b"], path=path)

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    log_like = np.full_like(inputs[:, [0]], 2)
    log_priors = np.ones_like(inputs)
    mock_proposal.logpdf.return_value = np.full_like(log_like, 3)

    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    # log posterior = 3 * .8 + 2 * 0.2 + 2 * 0.2 = 3.2
    np.testing.assert_array_equal(post, np.full(log_like.shape, 3.2))


@pytest.mark.parametrize("req_phi", [[1.0], 0.4, [0.1, 0.3]])
def test_vectormcmc_kernel_geometric_path_overrride_w_req_phi(mocker, req_phi):
    mock_proposal = mocker.Mock()
    path = GeometricPath(proposal=mock_proposal, required_phi=req_phi)
    path.phi = 0.2

    vmcmc = VectorMCMC(model=None, data=None, priors=None)
    kernel = VectorMCMCKernel(vmcmc, param_order=["a", "b"], path=path)

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    log_like = np.full_like(inputs[:, [0]], 2)
    log_priors = np.ones_like(inputs)
    mock_proposal.logpdf.return_value = np.full_like(log_like, 3)

    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    if path.required_phi_list == []:
        # log posterior = 3 * .8 + 2 * 0.2 + 2 * 0.2 = 3.2
        np.testing.assert_array_equal(post, np.full(log_like.shape, 3.2))
    elif path.phi < min(path.required_phi_list):
        # log posterior = 3 * 0.5 + 2 * 0.5 + 2 * 0.2 = 2.9
        np.testing.assert_array_equal(post, np.full(log_like.shape, 2.9))
    else:
        # log posterior = 3 * 0 + 2 * 1 + 2 * 0.2 = 2.4
        np.testing.assert_array_equal(post, np.full(log_like.shape, 2.4))
