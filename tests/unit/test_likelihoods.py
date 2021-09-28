import numpy as np
import pytest

from smcpy import Normal


def test_normal_log_like(mocker):
    inputs = np.ones((2, 5, 3))
    data = np.ones(8)
    model = mocker.Mock(return_value=np.ones((2, 5, 8)) * 2)

    expected_like = np.log(1 / (2 * np.pi) ** 4 * np.exp(-4))
    expected_like = np.tile(expected_like, (2, 5, 1))

    norm = Normal(model, data)
    like = norm(inputs)

    np.testing.assert_array_equal(like, expected_like)


def test_norm_log_like_nan_output(mocker):
    inputs = np.ones((2, 5, 3))
    data = np.ones(8)
    model = mocker.Mock(return_value=np.array([[[np.nan]], [[1]]]))

    norm = Normal(model, data)
    with pytest.raises(ValueError):
        _ = norm(inputs)
