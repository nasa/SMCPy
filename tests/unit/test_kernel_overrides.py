import pytest

from smcpy.mcmc.vector_mcmc import *
from smcpy import VectorMCMCKernel
from smcpy.paths import GeometricPath


def test_vectormcmc_kernel_no_overrride():
    vmcmc = VectorMCMC(model=None, data=None, priors=None)
    original_method = vmcmc.evaluate_log_posterior
    kernel = VectorMCMCKernel(vmcmc, param_order=['a', 'b'], path=None)
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
    kernel = VectorMCMCKernel(vmcmc, param_order=['a', 'b'], path=path)

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    log_like = np.full_like(inputs[:, [0]], 2)
    log_priors = np.ones_like(inputs)

    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    np.testing.assert_array_equal(post, np.full(log_like.shape[0], 2))


def test_vectormcmc_kernel_geometric_path_overrride_manual_updates():
    path = GeometricPath()

    vmcmc = VectorMCMC(model=None, data=None, priors=None)
    kernel = VectorMCMCKernel(vmcmc, param_order=['a', 'b'], path=path)

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    log_like = np.full_like(inputs[:, [0]], 2)
    log_priors = np.ones_like(inputs)

    path.phi = 0.5
    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    np.testing.assert_array_equal(post, np.full(log_like.shape[0], 3))

    path.phi = 1
    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    np.testing.assert_array_equal(post, np.full(log_like.shape[0], 4))


def test_vectormcmc_kernel_updates_phi(mocker):
    path = GeometricPath()

    vmcmc_mock = mocker.Mock()
    vmcmc_mock.smc_metropolis.return_value = (np.ones((1, 2)), None)

    kernel = VectorMCMCKernel(vmcmc_mock, ['a', 'b'], path=path)
    kernel.mutate_particles({'a': [2], 'b': [2]}, 1, None, phi=0.1)

    assert path.phi == 0.1


def test_vectormcmc_kernel_default_path_is_geometric(mocker):
    kernel = VectorMCMCKernel(mocker.Mock(), ['a', 'b'], path=None)

    assert kernel._path.phi == 0
    assert isinstance(kernel._path, GeometricPath)


def test_vectormcmc_kernel_phi_update_must_be_monotonic(mocker):
    path = GeometricPath()
    path.phi = 0.5

    vmcmc_mock = mocker.Mock()
    vmcmc_mock.smc_metropolis.return_value = (np.ones((1, 2)), None)

    kernel = VectorMCMCKernel(vmcmc_mock, ['a', 'b'], path=path)
    with pytest.raises(ValueError):
        kernel.mutate_particles({'a': [2], 'b': [2]}, 1, None, phi=0.49)


def test_vectormcmc_kernel_geometric_path_overrride_with_proposal(mocker):
    mock_proposal = mocker.Mock()
    path = GeometricPath(proposal=mock_proposal)
    path.phi = 0.2

    vmcmc = VectorMCMC(model=None, data=None, priors=None)
    kernel = VectorMCMCKernel(vmcmc, param_order=['a', 'b'], path=path)

    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    log_like = np.full_like(inputs[:, [0]], 2)
    log_priors = np.ones_like(inputs)
    mock_proposal.log_pdf.return_value = np.full_like(log_like, 3)

    post = kernel._mcmc.evaluate_log_posterior(inputs, log_like, log_priors)

    # log posterior = 3 * .8 + 2 * 0.2 + 2 * 0.2 = 3.2
    np.testing.assert_array_equal(post, np.full(log_like.shape[0], 3.2))
