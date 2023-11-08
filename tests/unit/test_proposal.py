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

    cp = MultivarIndependent(dist1, dist2)
    logpdf = cp.logpdf(inputs)

    dist1.logpdf.called_once_with(np.full((3, 1), 5))
    dist2.logpdf.called_once_with(np.full((3, 2), 6))

    np.testing.assert_array_equal(logpdf, np.full((3, 1), 3))


def test_multivarindp_proposal_rvs(dist1, dist2):
    cp = MultivarIndependent(dist1, dist2)

    samples = cp.rvs(3)

    dist1.rvs.called_once_with(3)
    dist2.rvs.called_once_with(3)

    np.testing.assert_array_equal(samples, np.ones((3, 3)))