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
