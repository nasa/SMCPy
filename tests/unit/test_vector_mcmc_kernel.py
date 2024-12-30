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
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=tuple("a"))
    assert vmcmc._mcmc._eval_model == stub_model
    assert vmcmc._mcmc._data == data
    assert vmcmc._mcmc._priors == priors
    assert vmcmc._param_order == tuple("a")


def test_sample_from_prior(vector_mcmc, mocker):
    mocked_sample = np.array([[1, 2], [3, 4], [5, 6]])
    mocker.patch.object(vector_mcmc, "sample_from_priors", return_value=mocked_sample)
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=["a", "b"])

    samples = vmcmc.sample_from_prior(num_samples=3)

    np.testing.assert_array_equal(samples["a"], mocked_sample[:, 0])
    np.testing.assert_array_equal(samples["b"], mocked_sample[:, 1])


def test_sample_from_proposal(vector_mcmc, mocker):
    mocked_sample = np.array([[1, 2], [3, 4], [5, 6]])
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=["a", "b"])
    mocked_proposal = mocker.Mock()
    rvs = mocker.patch.object(mocked_proposal, "rvs", return_value=mocked_sample)
    vmcmc.path._proposal = mocked_proposal

    samples = vmcmc.sample_from_proposal(num_samples=3)

    np.testing.assert_array_equal(samples["a"], mocked_sample[:, 0])
    np.testing.assert_array_equal(samples["b"], mocked_sample[:, 1])
    rvs.assert_called_once_with(3, random_state=vmcmc.rng)


@pytest.mark.parametrize(
    "param_dict, expected",
    (
        [{"a": np.array([1]), "b": np.array([2])}, np.array([[1]])],
        [{"a": np.array([1, 1]), "b": np.array([1, 2])}, np.array([[1], [1]])],
    ),
)
def test_kernel_log_likelihoods(vector_mcmc, param_dict, expected, mocker):
    mocker.patch.object(
        vector_mcmc, "evaluate_log_likelihood", new=lambda x: x[:, 0].reshape(-1, 1)
    )
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=["a", "b"])

    log_likes = vmcmc.get_log_likelihoods(param_dict)
    np.testing.assert_array_equal(log_likes, expected)


@pytest.mark.parametrize(
    "param_dict, expected",
    (
        [{"a": np.array([1]), "b": np.array([2])}, np.array([[3]])],
        [{"a": np.array([1, 1]), "b": np.array([1, 2])}, np.array([[2], [3]])],
    ),
)
def test_kernel_log_priors(vector_mcmc, param_dict, expected, mocker):
    mocker.patch.object(vector_mcmc, "evaluate_log_priors", new=lambda x: x)
    vmcmc = VectorMCMCKernel(vector_mcmc, param_order=["b", "a"])

    log_priors = vmcmc.get_log_priors(param_dict)
    np.testing.assert_array_equal(log_priors, expected)


@pytest.mark.parametrize("num_vectorized", (1, 5))
def test_mutate_particles(vector_mcmc, num_vectorized, mocker):
    num_samples = 2
    cov = np.eye(2)

    param_array = np.array([[1, 2]] * num_vectorized)
    param_dict = dict(zip(["a", "b"], param_array.T))

    mocked_return = (
        np.array([[2, 3]] * num_vectorized),
        np.array([[2]] * num_vectorized),
    )
    smc_metropolis = mocker.patch.object(
        vector_mcmc, "smc_metropolis", return_value=mocked_return
    )

    kernel = VectorMCMCKernel(vector_mcmc, param_order=["a", "b"])
    kernel.path.phi = 1
    mutated = kernel.mutate_particles(param_dict, num_samples, cov)
    new_param_dict = mutated[0]
    new_log_likes = mutated[1]

    expected_params = {
        "a": np.array([2] * num_vectorized),
        "b": np.array([3] * num_vectorized),
    }
    expected_log_likes = np.ones((num_vectorized, 1)) * 2

    np.testing.assert_array_equal(new_param_dict["a"], expected_params["a"])
    np.testing.assert_array_equal(new_param_dict["b"], expected_params["b"])
    np.testing.assert_array_equal(new_log_likes, expected_log_likes)

    calls = smc_metropolis.call_args[0]
    for i, expect in enumerate([param_array, num_samples, cov]):
        np.testing.assert_array_equal(calls[i], expect)


def test_convert_dict_to_array_when_objects(vector_mcmc):
    kernel = VectorMCMCKernel(vector_mcmc, param_order=["a", "b"])
    param_dict = {
        "a": np.array([np.array] * 3),
        "b": np.array([np.array] * 3),
    }
    array = kernel._conv_param_dict_to_array(param_dict)
    np.testing.assert_array_equal(array, np.array([[np.array, np.array]] * 3))


def test_kernel_calls_set_rng(vector_mcmc, mocker):
    seed = 1
    rng = np.random.default_rng(seed)
    mocker.patch.object(VectorMCMCKernel, "set_mcmc_rng")
    kernel = VectorMCMCKernel(vector_mcmc, param_order=[], rng=rng)
    assert kernel.rng == rng
    kernel.set_mcmc_rng.assert_called_once_with(rng)


def test_kernel_sets_rng(vector_mcmc):
    seed = 1
    rng = np.random.default_rng(seed)
    kernel = VectorMCMCKernel(vector_mcmc, param_order=[], rng=rng)
    assert kernel._mcmc._rng == rng


def test_kernel_overwrites_with_default(vector_mcmc):
    orig_rng = vector_mcmc.rng
    kernel = VectorMCMCKernel(vector_mcmc, param_order=[], rng=None)
    assert kernel._mcmc.rng != orig_rng
    assert kernel.rng == kernel._mcmc.rng
