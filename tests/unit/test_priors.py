import numpy as np
import pytest

from smcpy.priors import ImproperUniform

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
