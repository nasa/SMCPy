import pytest
import numpy as np

from copy import copy

from smcpy.smc.particles import Particles

@pytest.fixture
def param_dict():
    s = (2, 5)
    return {'p1': np.ones(s), 'p2': np.full(s, 2), 'p3': np.full(s, 3)}

@pytest.fixture
def particles(param_dict):
    log_likes = np.ones((2, 5, 1))
    weights = np.array([[[1], [4], [1], [1], [1]],
                        [[1], [6], [1], [1], [1]]])
    log_weights = np.log(weights)
    return Particles(param_dict, log_likes, log_weights)


def test_set_particles_params_not_dict():
    with pytest.raises(TypeError):
        Particles(np.ones(5), 1, 1)


def test_set_particles_likes_wrong_shape(param_dict):
    with pytest.raises(ValueError):
        Particles(param_dict, np.ones((2, 4, 3)), np.ones((2, 5, 3)))


def test_set_particles_weights_wrong_shape(param_dict):
    with pytest.raises(ValueError):
        Particles(param_dict, np.ones((2, 5, 3)), np.ones((2, 4, 3)))


def test_set_params(particles):
    expected_param_names = {'p1', 'p2', 'p3'}
    expected_params = np.tile([1, 2, 3], (2, 5, 1))
    expected_p2 = np.full((2, 5), 2)

    assert set(particles._param_names) == expected_param_names
    np.testing.assert_array_equal(particles.params, expected_params)
    np.testing.assert_array_equal(particles.param_dict['p2'], expected_p2)


def test_set_likes(particles):
    expected_log_likes = np.ones((2, 5, 1))
    np.testing.assert_array_equal(particles.log_likes, expected_log_likes)


def test_set_weights(particles):
    expect_wts = np.array([[[.125], [.5], [.125], [.125], [.125]],
                           [[.1], [.6], [.1], [.1], [.1]]])
    expect_log_wts = np.log(expect_wts)
    np.testing.assert_array_almost_equal(particles.log_weights, expect_log_wts)
    np.testing.assert_array_almost_equal(particles.weights, expect_wts)


def test_particles_copy(particles):
    particles_copy = particles.copy()
    assert particles_copy is not particles
    assert isinstance(particles_copy, Particles)


def test_compute_ess(particles):
    expected_ess = np.array([[[3.2]], [[2.5]]])
    np.testing.assert_array_almost_equal(particles.compute_ess(), expected_ess)


@pytest.mark.parametrize('params, weights',
                         (({'a': np.array([[1, 2], [1, 2]]),
                            'b': np.array([[2, 3], [2, 3]])},
                           np.ones((2, 2, 1))),
                          ({'a': np.array([[1, 5/3], [1, 5/3]]),
                            'b': np.array([[4, 2], [4, 2]])},
                           np.array([[[1], [3]], [[1], [3]]]))))
def test_compute_mean(params, weights):
    log_likes = np.ones((2, 2, 1))
    expected_means = np.array([[[1.5, 2.5]], [[1.5, 2.5]]])

    particles = Particles(params, log_likes, np.log(weights))

    np.testing.assert_array_equal(particles.compute_mean(package=False),
                                  expected_means)


@pytest.mark.parametrize('params, weights, expected_var',
             (({'a': np.array([[1, 2], [1, 2]]),
                'b': np.array([[2, 3], [2, 3]])},
               np.ones((2, 2, 1)),
               np.full((2, 1, 2), 0.5)),
              ({'a': np.array([[1, 5/3], [1, 5/3]]),
                'b': np.array([[4, 2], [4, 2]])},
               np.array([[[1], [3]], [[1], [3]]]),
               np.array([[[2/9, 2.]], [[2/9, 2.]]]))))
def test_compute_variance(params, weights, expected_var):
    log_likes = np.ones((2, 2, 1))

    particles = Particles(params, log_likes, np.log(weights))
    variance = particles.compute_variance(package=False)

    np.testing.assert_array_almost_equal(variance, expected_var)


def test_compute_std_dev(mocker, particles):
    particles.compute_variance = mocker.Mock(return_value=np.full((2, 1, 3), 4))
    expected_std = np.full((2, 1, 3), 2)
    np.testing.assert_array_equal(particles.compute_std_dev(package=False),
                                  expected_std)

def test_compute_covariance(mocker):
    params = {'a': [1.1, 1.0, 0.8], 'b': [2.2, 2.1, 1.9]}
    log_likes = np.ones(3)
    log_weights = np.log([0.1, 0.7, 0.2])

    particles = Particles(params, log_likes, log_weights)
    mocker.patch.object(particles, 'compute_mean',
                        return_value=np.array([0.97, 2.06]))

    scale = 1 / (1 - np.sum(np.array([0.1, 0.7, 0.2]) ** 2))
    expected_cov = np.array([[0.0081, 0.0081], [0.0081, 0.0082]]) * scale
    np.testing.assert_array_almost_equal(particles.compute_covariance(),
                                         expected_cov)


#@pytest.mark.filterwarnings('ignore: Covariance matrix is')
#@pytest.mark.parametrize('params, weights, expected_var',
#        (({'a': [1, 2], 'b': [2, 3]}, [1, 1], np.array([0.5, 0.5])),
#         ({'a': [1, 5/3], 'b': [4, 2]}, [1, 3], np.array([2/9, 2.]))))
#def test_non_positive_def_cov_is_independent(params, weights, expected_var,
#                                             mocker):
#    log_likes = np.ones(2)
#    particles = Particles(params, log_likes, np.log(weights))
#    mocker.patch.object(particles, '_is_positive_definite', return_value=False)
#    expected_cov = np.eye(2) * expected_var
#
#    cov = particles.compute_covariance()
#
#    np.testing.assert_array_almost_equal(cov, expected_cov)
