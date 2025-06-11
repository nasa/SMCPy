import numpy as np
import pytest

from smcpy.hierarch_log_likelihoods import ApproxHierarch, MVNHierarchModel
from smcpy.log_likelihoods import BaseLogLike


def test_import_approx_hierarch():
    like = ApproxHierarch(1, 2, 3)

    assert like._model == 1
    assert like._data == 2
    assert like._args == 3
    assert isinstance(like, BaseLogLike)


def test_approx_hierarch_call(mocker):
    n_samples = 15
    nre1 = 3
    nre2 = 3
    nre3 = 5

    conditionals = [
        np.tile(np.arange(1, nre1 + 1), (n_samples, 1)),
        np.tile(np.arange(1, nre2 + 1), (n_samples, 1)),
        np.tile(np.arange(1, nre3 + 1), (n_samples, 1)),
    ]
    model = mocker.Mock(
        side_effect=[
            np.log(conditionals[0]),
            np.log(conditionals[1]),
            np.log(conditionals[2]),
        ]
    )
    data_log_priors = [np.log([5] * nre1), np.log([7] * nre2), np.log([9] * nre3)]
    data = [
        np.array([[1, 2]] * nre1),
        np.array([[2, 3]] * nre2),
        np.array([[3, 4]] * nre3),
    ]
    model_class = mocker.Mock(return_value=model)
    marginal_log_likes = np.log([1, 2, 3])
    args = (marginal_log_likes, data_log_priors)
    inputs = np.ones((n_samples, 5))

    expected_like = np.prod([1 / 3 * 6 / 5, 2 / 3 * 6 / 7, 3 / 5 * 15 / 9])
    expected_log_like = np.log(expected_like)
    expected_log_likes = np.tile(expected_log_like, (n_samples, 1))

    like = ApproxHierarch(model_class, data, args)
    log_likes = like(inputs)

    model_class.assert_called_once_with(inputs)
    for i, d in enumerate(data):
        call = model.call_args_list[i][0][0]
        np.testing.assert_array_equal(call, d)

    np.testing.assert_array_almost_equal(log_likes, expected_log_likes)


@pytest.mark.parametrize("n_inputs, n_hyper", [(2, 3), (4, 10)])
def test_hierarch_mvnormal_model_init(n_inputs, n_hyper):
    n_samples = 10
    n_total = n_inputs + n_hyper
    inputs = np.tile(np.arange(n_total), (n_samples, 1))
    model = MVNHierarchModel(inputs)

    np.testing.assert_array_equal(model._inputs, inputs[:, :n_inputs])
    np.testing.assert_array_equal(model._hyperparams, inputs[:, n_inputs:])

    symmetric_matrix = model._cov[0]
    assert symmetric_matrix.shape[0] == symmetric_matrix.shape[1]
    np.testing.assert_array_equal(symmetric_matrix, symmetric_matrix.T)
    np.testing.assert_array_equal(
        symmetric_matrix[np.triu_indices(n_inputs)],
        np.arange(n_total - 1, n_inputs - 1, -1)[::-1],
    )


@pytest.mark.parametrize("n_inputs,n_hyper", [(1, 3), (7, 5), (15, 294)])
def test_hierarch_mvnormal_model_bad_dim(n_inputs, n_hyper):
    n_total = n_inputs + n_hyper
    inputs = np.tile(np.arange(n_total), (10, 1))
    with pytest.raises(IndexError):
        model = MVNHierarchModel(inputs)


def test_hiearch_mvnormal_model_call():
    num_posterior_samples = 10
    num_samples = 3

    proposed_params = np.array([[0, 2]] * num_samples)
    proposed_hyperparams = np.array([[4, -1, 2]] * num_samples)
    proposal = np.hstack((proposed_params, proposed_hyperparams))

    data = np.array([[1, 2]] * num_posterior_samples)
    expected_log_like = -1 / 2 * (2 * np.log(2 * np.pi) + np.log(7) + 2 / 7)
    expected_log_like = np.full((num_samples, num_posterior_samples), expected_log_like)

    model = MVNHierarchModel(proposal)
    log_like = model(data)

    np.testing.assert_array_equal(log_like, expected_log_like)
