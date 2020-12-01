import numpy as np
import pytest

from smcpy.log_likelihoods import multisource_normal


@pytest.mark.parametrize("args_0", [(1, 7), (8, 0)])
@pytest.mark.parametrize("args_1", [(1 / np.sqrt(np.pi), 1 / np.sqrt(np.pi))])
def test_multisource_normal_fixed_std(args_0, args_1):
    inputs = np.ones((5, 6))
    data = np.ones(8)
    model = lambda x: np.ones((x.shape[0], data.shape[0])) * 2
    args = [args_0, args_1]

    expected_log_like = -np.log(2) * 8 / 2 - 4 * np.pi

    log_like = multisource_normal(inputs, model, data, args)

    np.testing.assert_array_equal(log_like, expected_log_like)
