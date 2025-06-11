import numpy as np
import pytest

from smcpy.proposals import MultivarIndependent


@pytest.fixture
def dist1(mocker):
    dist1 = mocker.Mock()
    dist1.logpdf.return_value = np.ones((3, 1))
    dist1.rvs.return_value = np.ones((3, 1))
    return dist1


@pytest.fixture
def dist2(mocker):
    dist2 = mocker.Mock()
    dist2.logpdf.return_value = np.full((3, 1), 2)
    dist2.rvs.return_value = np.ones((3, 2))
    return dist2


def test_multivarindp_proposal_logpdf(dist1, dist2):
    inputs = np.array([[5, 6, 6]] * 3)

    dist1.rvs.return_value = np.ones(1)
    dist2.rvs.return_value = np.ones((1, 2))

    cp = MultivarIndependent(dist1, dist2)
    logpdf = cp.logpdf(inputs)

    np.testing.assert_array_equal(dist1.logpdf.call_args[0][0], np.full((3, 1), 5))
    np.testing.assert_array_equal(dist2.logpdf.call_args[0][0], np.full((3, 2), 6))

    np.testing.assert_array_equal(logpdf, np.full((3, 1), 3))


def test_multivarindp_proposal_rvs(dist1, dist2):
    cp = MultivarIndependent(dist1, dist2)

    rng = np.random.default_rng()
    samples = cp.rvs(3, random_state=rng)

    dist1.rvs.assert_called_with(3, random_state=rng)
    dist2.rvs.assert_called_with(3, random_state=rng)

    np.testing.assert_array_equal(samples, np.ones((3, 3)))
