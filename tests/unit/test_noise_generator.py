import numpy as np
import pytest

from smcpy.utils.noise_generator import generate_noisy_data


def test_nothing_to_generate():
    std = 1
    output = np.array([[]])
    noisy_data = generate_noisy_data(std, output)

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
    noisy_data = generate_noisy_data(std=std, output=model_output)

    assert noisy_data.size == expected_size


@pytest.mark.parametrize("std", [-1, -1000, -33])
def test_valid_std(std):
    model_output = np.array(
        [[3, 3, 3, 3, 3], [4, 4, 4, 4, 4], [8, 8, 8, 8, 8], [11, 12, 13, 15, 18]]
    )
    with pytest.raises(ValueError):
        generate_noisy_data(std, model_output)


def test_provided_std_val():
    std = 0
    model_output = np.array([[1, 2, 3], [1, 2, 3]])
    noisy_data = generate_noisy_data(std, model_output)
    np.testing.assert_array_equal(noisy_data, model_output)


def test_noise_array(mocker):
    std_array = [0.1, 0.3]
    model_output = np.array([[1, 2, 3], [2, 4, 6]])

    mockednp = mocker.patch(
        "smcpy.utils.noise_generator.np.random.normal",
        return_value=np.array([[1, 2]] * 3),
    )
    noisy_data = generate_noisy_data(std_array, model_output)
    expected_output = np.array([[2, 3, 4], [4, 6, 8]])

    np.testing.assert_array_equal(noisy_data, expected_output)

    first_call = mockednp.call_args_list[0]
    call_args = first_call[0]
    assert call_args == (0, std_array, (3, 2))


@pytest.mark.parametrize(
    "model_output", [np.array([1, 2, 3, 4, 5]), np.array([[[0, 0, 0]]])]
)
def test_model_output_valid_2d(model_output):
    std = 1
    with pytest.raises(ValueError):
        generate_noisy_data(std, model_output)


@pytest.mark.parametrize("std", [[1, 2, 3], [5, 5, 5, 5]])
def test_valid_number_std_array(std):
    model_output = np.array([[1, 2, 3], [2, 4, 6]])
    with pytest.raises(ValueError):
        generate_noisy_data(std, model_output)
