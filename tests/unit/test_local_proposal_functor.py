import pytest
import numpy as np

from smcpy.utils.mcmc_utils import *


@pytest.mark.parametrize("points", [(np.array([])), (np.array([[0, 0]]))])
def test_at_least_2_points_required(points):
    with pytest.raises(ValueError):
        local_mcmc = LocalMCMCProposal()
        local_mcmc(points, None)


def test_compute_covariance_identical_points():
    inputs = np.array([[1, 2], [1, 2], [1, 2]])
    scale_factor = 1.0

    local_mcmc = LocalMCMCProposal()

    with pytest.raises(ValueError):
        local_mcmc.compute_covariance(inputs, scale_factor)


def test_cov_scaling():
    inputs = np.array([[1, 3], [1, 2], [1, 1]])
    scale_factor = 1.0

    local_mcmc = LocalMCMCProposal()
    result = local_mcmc.compute_covariance(inputs, scale_factor)

    assert result.shape == (3, 2, 2)
    for i in range(len(inputs)):
        continue


def test_compute_covariance_different_points():
    inputs = np.array([[1, 2], [3, 4], [5, 6]])
    scale_factor = 1.0

    local_mcmc = LocalMCMCProposal()
    result = local_mcmc.compute_covariance(inputs, scale_factor)

    assert result.shape == (inputs.shape[0], inputs.shape[1], inputs.shape[1])
    for cov in result:
        symmetric = np.allclose(cov, cov.T)
        assert symmetric


def test_median_distance_identical_points():
    inputs = np.array([[1, 1], [1, 1], [1, 1]])
    local_mcmc = LocalMCMCProposal()
    result = local_mcmc.median_distance(inputs)
    assert result == 0.0


def test_median_distance_collinear_points():
    inputs = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])

    local_mcmc = LocalMCMCProposal()
    result = local_mcmc.median_distance(inputs)
    # Distances: 1, 1, 1, 2, 2, 3 -> median should be 1.5
    expected = 1.5
    assert result == expected


@pytest.mark.parametrize("sigma", [(1), (5), (1 / 2)])
def test_rbf_valid_sigma(sigma):
    inputs = np.array([[0, 0], [1, 1], [2, 2]])
    local_mcmc = LocalMCMCProposal()

    result = local_mcmc.rbf(inputs, sigma)

    assert isinstance(result, np.ndarray)
    assert result.shape == (inputs.shape[0], inputs.shape[0])

    assert np.allclose(result, result.T)

    # Diagonal should be all ones (distance from point to itself = 0)
    assert np.allclose(np.diag(result), 1.0)


@pytest.mark.parametrize("sigma", [(0), (0.0), (-0), (-0.0), (-1), (-5), (-1 / 2)])
def test_rbf_nonpositive_sigma(sigma):
    with pytest.raises(ValueError):
        inputs = np.array([[0, 0], [1, 1], [2, 2]])

        local_mcmc = LocalMCMCProposal()
        local_mcmc.rbf(inputs, sigma)


def test_proposal_mathematical_correctness(mocker):
    cov = np.array([[[1, 0], [0, 1]], [[4, 0], [0, 4]]])

    chol1 = np.array([[1, 0], [0, 1]])
    chol2 = np.array([[2, 0], [0, 2]])

    z = np.array([[1, 0], [0, 1]])
    mock_default_rng = mocker.patch("numpy.random.default_rng")
    mock_rng = mocker.Mock()
    mock_rng.normal.return_value = z
    mock_default_rng.return_value = mock_rng

    local_mcmc = LocalMCMCProposal()
    local_mcmc._ensure_psd_cov_and_do_chol_decomp = mocker.Mock()
    local_mcmc._ensure_psd_cov_and_do_chol_decomp.side_effect = [chol1, chol2]

    inputs = np.array([[0, 0], [10, 10]])
    result = local_mcmc.proposal(inputs, cov)

    # delta[0] = chol1 @ z[0] = [[1,0],[0,1]] @ [1,0] = [1,0]
    # delta[1] = chol2 @ z[1] = [[2,0],[0,2]] @ [0,1] = [0,2]
    # result[0] = [0,0] + [1,0] = [1,0]
    # result[1] = [10,10] + [0,2] = [10,12]
    expected = np.array([[1, 0], [10, 12]])
    np.testing.assert_array_equal(result, expected)


def test_call(mocker):
    scale_factor = 1.0
    inputs = np.array([[1, 2], [3, 4], [5, 6]])

    mock_orig_cov = np.array([[[1, 0], [0, 1]], [[2, 1], [1, 2]], [[1, 1], [1, 1]]])
    mock_prop_cov = np.array(
        [[[1.1, 0.1], [0.1, 1.1]], [[2.1, 1.1], [1.1, 2.1]], [[1.2, 1.2], [1.2, 1.2]]]
    )

    local_mcmc = LocalMCMCProposal()
    local_mcmc.compute_covariance = mocker.Mock()
    local_mcmc.compute_covariance.side_effect = [mock_orig_cov, mock_prop_cov]

    mock_prop_inputs = np.array([[1.1, 2.1], [3.1, 4.1], [5.1, 6.1]])

    mock_proposal = mocker.patch.object(
        local_mcmc, "proposal", return_value=mock_prop_inputs
    )
    orig_inputs, orig_cov, prop_inputs, prop_cov = local_mcmc(inputs, scale_factor)

    np.testing.assert_array_equal(orig_inputs, inputs)
    np.testing.assert_array_equal(orig_cov, mock_orig_cov)
    np.testing.assert_array_equal(prop_inputs, mock_prop_inputs)
    np.testing.assert_array_equal(prop_cov, mock_prop_cov)

    assert local_mcmc.compute_covariance.call_count == 2
    local_mcmc.compute_covariance.assert_has_calls(
        [
            mocker.call(inputs, scale_factor),
            mocker.call(mock_prop_inputs, scale_factor),
        ]
    )

    mock_proposal.assert_called_once_with(inputs, mock_orig_cov)
