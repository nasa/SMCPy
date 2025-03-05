import numpy as np
import pytest

from smcpy import MVNRandomEffects


@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_mvnrand_eff(n_samples):
    theta = np.array([1, 2, 3])
    theta_s = np.array([[3, 2.5, 4], [5.8, 4.0, 5.5], [3, 3, 4.25], [1, 2, 2.4]])
    inputs = np.concatenate((theta, theta_s.flatten())).reshape(1, -1)
    inputs = np.tile(inputs, (n_samples, 1))

    precision = np.array([[1, -0.1, -1.5], [-0.1, 2, -1.2], [-1.5, -1.2, 4]])
    te_covariance = np.linalg.inv(precision)
    re_std = [1, 2, 4, 2]
    data = np.array([[30, 79, 29, 9]]).T
    model = lambda x: np.sum(x**2, axis=1).reshape(-1, 1)

    expected_LL = np.array([[-24.29314836]] * n_samples)

    idx1, idx2 = np.triu_indices(te_covariance.shape[0])
    args = (list(te_covariance[idx1, idx2]), re_std)

    mvnre = MVNRandomEffects(model, data, args)

    np.testing.assert_array_almost_equal(mvnre(inputs), expected_LL)
