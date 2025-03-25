import numpy as np
import pytest

from smcpy.utils.noise_generator import generate_noisy_data


def test_nothing_to_generate():
    std = 1
    model_output = np.array([[]])
    noisy_data = generate_noisy_data(model_output, std)

    np.testing.assert_array_equal(noisy_data, np.array([[]]))


@pytest.mark.parametrize(
    "model_output, expected_size",
    [
        (
            np.array([[]]),
            0,
        ),
        (np.array([[1, 2, 3], [2, 4, 6]]), 6),
    ],
)
def test_noise_amount_generated(model_output, expected_size):
    std = 1
    noisy_data = generate_noisy_data(model_output, std=std)

    assert noisy_data.size == expected_size


@pytest.mark.parametrize("std", [-1, -1000, -33])
def test_valid_std(std):
    model_output = np.array([[]])

    with pytest.raises(ValueError):
        generate_noisy_data(model_output, std)


def test_scalar_std_val():
    std = 0
    model_output = np.array([[1, 2, 3], [1, 2, 3]])
    noisy_data = generate_noisy_data(model_output, std)
    np.testing.assert_array_equal(noisy_data, model_output)


def test_noise_array(mocker):
    std_array = [0.1, 0.3]
    model_output = np.array([[1, 2, 3], [2, 4, 6]])

    mock_rng = mocker.Mock()
    mock_rng.normal.return_value = np.array([[1, 2]] * 3)
    mocker.patch(
        "smcpy.utils.noise_generator.np.random.default_rng", return_value=mock_rng
    )

    noisy_data = generate_noisy_data(model_output, std_array)
    expected_output = np.array([[2, 3, 4], [4, 6, 8]])

    np.testing.assert_array_equal(noisy_data, expected_output)
    np_normal_call_args = mock_rng.normal.call_args_list[0][0]
    assert np_normal_call_args == (0, std_array, (3, 2))


@pytest.mark.parametrize(
    "model_output", [np.array([1, 2, 3, 4, 5]), np.array([[[0, 0, 0]]])]
)
def test_model_output_valid_2d(model_output):
    std = 1
    with pytest.raises(ValueError):
        generate_noisy_data(model_output, std)


def test_compare_seed_val():
    std = 1
    model_output = np.array([[0, 0, 0], [0, 0, 0]])

    noisy_data = generate_noisy_data(model_output, std, 1)
    noisy_data_same_seed = generate_noisy_data(model_output, std, 1)
    noisy_data_diff_seed = generate_noisy_data(model_output, std, 2)

    np.testing.assert_array_equal(noisy_data, noisy_data_same_seed)
    np.testing.assert_raises(
        AssertionError, np.testing.assert_array_equal, noisy_data, noisy_data_diff_seed
    )
