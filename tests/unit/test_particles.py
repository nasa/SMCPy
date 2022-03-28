import pytest
import numpy as np

from copy import copy

from smcpy.smc.particles import Particles

@pytest.fixture
def dummy_params():
    return {'a': [1] * 10, 'b': [2] * 10}


@pytest.fixture
def dummy_param_array():
    return np.array([[1, 2]] * 10)


@pytest.fixture
def dummy_log_likes():
    return [0.1] * 10


@pytest.fixture
def dummy_log_weights():
    return [0.2] * 10


@pytest.fixture
def particles(dummy_params, dummy_log_likes, dummy_log_weights):
    return Particles(dummy_params, dummy_log_likes, dummy_log_weights)


def test_set_particles(particles, dummy_params, dummy_log_likes,
                       dummy_log_weights, dummy_param_array):

    params = dummy_param_array
    log_likes = np.array(dummy_log_likes).reshape(-1, 1)
    normalized_weights = np.array([0.1] * 10).reshape(-1, 1)
    normalized_log_weights = np.log(normalized_weights)
    unnormalized_log_weights = np.array([[0.2]] * 10)

    assert particles.param_names == ('a', 'b')
    assert particles.num_particles == 10
    np.testing.assert_array_equal(particles.params, params)
    np.testing.assert_array_equal(particles.log_likes, log_likes)
    np.testing.assert_array_equal(particles.log_weights, normalized_log_weights)
    np.testing.assert_array_almost_equal(particles.weights, normalized_weights)
    np.testing.assert_array_equal(particles.param_dict['a'], params[:, 0])
    np.testing.assert_array_equal(particles.param_dict['b'], params[:, 1])

    assert isinstance(particles.total_unnorm_log_weight, float)
    assert particles.total_unnorm_log_weight == pytest.approx(2.50258509)
    assert particles.attrs == dict()

def test_params_value_error():
    params = {'a': [1, 2], 'c': [2], 'b': 4}
    with pytest.raises(ValueError):
        Particles(params, None, None)


def test_params_type_error():
    with pytest.raises(TypeError):
        Particles([], None, None)


@pytest.mark.parametrize('log_likes', (4, [], [1], np.ones(3), np.ones(5)))
def test_log_likes_value_errors(log_likes):
    params = {'a': [5] * 4}
    with pytest.raises(ValueError):
        Particles(params, log_likes, None)


@pytest.mark.parametrize('log_weights', (4, [], [1], np.ones(3), np.ones(5)))
def test_log_weights_value_errors(log_weights):
    params = {'a': [5] * 4}
    log_likes = [5] * 4
    with pytest.raises(ValueError):
        Particles(params, log_likes, log_weights)


def test_particles_copy(particles):
    particles_copy = particles.copy()
    assert particles_copy is not particles
    assert isinstance(particles_copy, Particles)


def test_compute_ess(particles, dummy_log_weights):
    expected_norm_log_weights = np.array([0.1] * 10)
    expected_ess = 1 / np.sum(expected_norm_log_weights ** 2)
    assert particles.compute_ess() == pytest.approx(expected_ess)


@pytest.mark.parametrize('params, weights',
                         (({'a': [1, 2], 'b': [2, 3]}, [1, 1]),
                          ({'a': [1, 5/3], 'b': [4, 2]}, [1, 3])))
def test_compute_mean(params, weights):
    log_likes = np.ones(2)
    expected_means = {'a': 1.5, 'b': 2.5}

    particles = Particles(params, log_likes, np.log(weights))

    assert particles.compute_mean() == expected_means


@pytest.mark.parametrize('params, weights, expected_var',
        (({'a': [1, 2], 'b': [2, 3]}, [1, 1], {'a': 0.5, 'b': 0.5}),
         ({'a': [1, 5/3], 'b': [4, 2]}, [1, 3], {'a': 2/9, 'b':2.})))
def test_compute_variance(params, weights, expected_var):
    log_likes = np.ones(2)

    particles = Particles(params, log_likes, np.log(weights))

    assert particles.compute_variance() == pytest.approx(expected_var)


@pytest.mark.parametrize('params, weights, expected_var',
        (({'a': [1, 2], 'b': [2, 3]}, [1, 1], {'a': 0.5, 'b': 0.5}),
         ({'a': [1, 5/3], 'b': [4, 2]}, [1, 3], {'a': 2/9, 'b':2.})))
def test_get_std_dev(params, weights, expected_var):
    log_likes = np.ones(2)

    particles = Particles(params, log_likes, np.log(weights))

    expected_var['a'] = np.sqrt(expected_var['a'])
    expected_var['b'] = np.sqrt(expected_var['b'])
    assert particles.compute_std_dev() == pytest.approx(expected_var)

def test_compute_covariance(mocker):
    params = {'a': [1.1, 1.0, 0.8], 'b': [2.2, 2.1, 1.9]}
    param_array = np.array(list(params.values()))
    log_likes = np.ones(3)
    weights = np.array([0.1, 0.7, 0.2])
    log_weights = np.log(weights)

    particles = Particles(params, log_likes, log_weights)
    mocker.patch.object(particles, '_is_positive_definite', return_value=True)
    cov_mock = mocker.patch('smcpy.smc.particles.np.cov',
                            return_value=np.array(1))

    cov = particles.compute_covariance()

    expected_call = mocker.call(param_array, ddof=0, aweights=weights)
    assert cov == np.array([[1]])
    assert cov.shape == (1, 1)
    assert cov_mock.called_once_with(expected_call)
