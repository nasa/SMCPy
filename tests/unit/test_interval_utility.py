import numpy as np
import pytest

from smcpy.utils.intervals import compute_intervals


def test_credible_interval_empty_outputs():
    output = np.array([[]])
    alpha = 0.95

    intervals = compute_intervals(output=output, alpha=alpha)

    np.testing.assert_array_equal(intervals, np.array([]))


@pytest.mark.parametrize(
    "alpha, expected_intervals",
    [
        (0.5, np.array([[0.4, 4, 40], [0.2, 2, 20]])),
        (0.25, np.array([[0.45, 4.5, 45], [0.15, 1.5, 15]])),
    ],
)
def test_credible_interval(alpha, expected_intervals):
    output = np.array(
        [[0.1, 1, 10], [0.2, 2, 20], [0.3, 3, 30], [0.4, 4, 40], [0.5, 5, 50]]
    )

    intervals = compute_intervals(output=output, alpha=alpha)

    np.testing.assert_array_almost_equal(intervals, expected_intervals)


@pytest.mark.parametrize(
    "alpha, expected_intervals",
    [
        (0.5, np.array([[0.4, 4, 40], [0.2, 2, 20]])),
        (0.25, np.array([[0.45, 4.5, 45], [0.15, 1.5, 15]])),
    ],
)
def test_credible_interval_reversed(alpha, expected_intervals):
    output = np.array(
        [[0.5, 5, 50], [0.4, 4, 40], [0.3, 3, 30], [0.2, 2, 20], [0.1, 1, 10]]
    )

    intervals = compute_intervals(output=output, alpha=alpha)

    np.testing.assert_array_almost_equal(intervals, expected_intervals)


@pytest.mark.parametrize(
    "alpha, expected_intervals",
    [
        (0.5, np.array([[0.4, 4, 40], [0.2, 2, 20]])),
        (0.25, np.array([[0.45, 4.5, 45], [0.15, 1.5, 15]])),
    ],
)
def test_credible_interval_unordered(alpha, expected_intervals):
    output = np.array(
        [[0.4, 4, 40], [0.1, 1, 10], [0.5, 5, 50], [0.2, 2, 20], [0.3, 3, 30]]
    )

    intervals = compute_intervals(output=output, alpha=alpha)

    np.testing.assert_array_almost_equal(intervals, expected_intervals)


@pytest.mark.parametrize("alpha_invalid", [-1, -0.5, 2, -2])
def test_valid_alpha(alpha_invalid):
    output = np.array(
        [[0.5, 5, 50], [0.4, 4, 40], [0.3, 3, 30], [0.2, 2, 20], [0.1, 1, 10]]
    )

    with pytest.raises(ValueError):
        compute_intervals(output=output, alpha=alpha_invalid)


@pytest.mark.parametrize(
    "alpha, expected_intervals",
    [
        (0, np.array([[0.5, 5, 50], [0.1, 1, 10]])),
        (1, np.array([[0.3, 3, 30], [0.3, 3, 30]])),
    ],
)
def test_edge_cases_alpha(alpha, expected_intervals):
    output = np.array(
        [[0.5, 5, 50], [0.4, 4, 40], [0.3, 3, 30], [0.2, 2, 20], [0.1, 1, 10]]
    )

    intervals = compute_intervals(output=output, alpha=alpha)
    np.testing.assert_array_almost_equal(intervals, expected_intervals)
