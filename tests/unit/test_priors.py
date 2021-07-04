import numpy as np
import pytest

from smcpy.priors import ImproperUniform, InvWishart

@pytest.mark.parametrize('sign', [-1, 1])
@pytest.mark.parametrize('x, expected', [(0, np.ones(1)), (1, np.ones(1)),
                                         (1e24, np.ones(1)),
                                         (np.ones(3), np.ones(3)),
                                         (np.ones((3, 1)), np.ones(3))])
def test_improper_uniform(sign, x, expected):
    prior = ImproperUniform()
    assert np.array_equal(prior.pdf(sign * x), expected)


@pytest.mark.parametrize('x', [np.ones((3, 2)), np.ones((2, 3))])
def test_improper_uniform_shape_error(x):
    prior = ImproperUniform()
    with pytest.raises(ValueError):
        prior.pdf(x)


@pytest.mark.parametrize('bounds, expected',
                         [((-5, 5), np.array([0, 1, 1, 1, 1, 1, 0])),
                          ((0, 5), np.array([0, 0, 0, 1, 1, 1, 0])),
                          ((0, None), np.array([0, 0, 0, 1, 1, 1, 1])),
                          ((None, None), np.array([1, 1, 1, 1, 1, 1, 1])),
                          ((None, 0), np.array([1, 1, 1, 1, 0, 0, 0])),
                          ((-10, None), np.array([0, 1, 1, 1, 1, 1, 1]))])
def test_improper_uniform_bounds(bounds, expected):
    x = np.array([-1000, -5, -2, 0, 2, 5, 1000])
    prior = ImproperUniform(*bounds)
    np.testing.assert_array_equal(prior.pdf(x), expected)


@pytest.mark.parametrize('n', [1, 2, 3, 4])
def test_invwishart_dim(n):
    iw = InvWishart(n)
    assert iw.dim == (n + 1) * n / 2


@pytest.mark.parametrize('num_samples', [1, 5, 10])
def test_invwishart_sample(mocker, num_samples):
    cov_dim = 3
    cov_sample = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
    expected_sample = np.tile(np.arange(6), (num_samples, 1))

    mock_invwis = mocker.Mock()
    mock_invwis.rvs.return_value = np.tile(cov_sample, (num_samples, 1, 1))
    mock_invwis_class = mocker.patch('smcpy.priors.invwishart',
                                     return_value=mock_invwis)

    iw = InvWishart(cov_dim)

    np.testing.assert_array_equal(iw.rvs(num_samples), expected_sample)
    mock_invwis.rvs.assert_called_once_with(num_samples)
    iw_class_call = mock_invwis_class.call_args[0]
    assert iw_class_call[0] == cov_dim
    np.testing.assert_array_equal(iw_class_call[1], np.eye(cov_dim))


@pytest.mark.parametrize('num_samples', [1, 5, 10])
def test_invwishart_pdf(mocker, num_samples):
    cov_dim = 3
    samples = np.tile(np.arange(6), (num_samples, 1))
    cov_sample = np.array([[0, 1, 2], [1, 3, 4], [2, 4, 5]])
    expected_cov = np.tile(cov_sample, (num_samples, 1, 1))
    expected_cov = np.transpose(expected_cov, axes=(1, 2, 0))
    expected_prior_probs = np.ones(num_samples)

    mock_invwis = mocker.Mock()
    mock_invwis.pdf.return_value = np.ones(num_samples)
    mock_invwis_class = mocker.patch('smcpy.priors.invwishart',
                                     return_value=mock_invwis)

    iw = InvWishart(cov_dim)

    np.testing.assert_array_equal(iw.pdf(samples), expected_prior_probs)
    np.testing.assert_array_equal(mock_invwis.pdf.call_args[0][0], expected_cov)


@pytest.mark.parametrize('num_samples', [1, 5])
def test_invwishart_zero_prob(mocker, num_samples):
    samples = np.tile(np.arange(6), (num_samples, 1))

    mock_invwis = mocker.Mock()
    mock_invwis.pdf.side_effect = np.linalg.LinAlgError
    mock_invwis_class = mocker.patch('smcpy.priors.invwishart',
                                     return_value=mock_invwis)

    iw = InvWishart(3)

    np.testing.assert_array_equal(iw.pdf(samples), np.zeros((num_samples, 1)))
