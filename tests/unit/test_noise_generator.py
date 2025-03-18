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
