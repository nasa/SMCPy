import pytest
import numpy as np

from copy import copy

from smcpy.smc.smc_step import SMCStep
from smcpy.particles.particle import Particle


@pytest.fixture
def particle(mocker):
    return mocker.Mock(Particle)


@pytest.fixture
def smc_step():
    return SMCStep()


def test_set_particles(particle, smc_step):
    smc_step.particles = [particle] * 4
    np.testing.assert_array_equal(smc_step.particles, [particle] * 4)


def test_type_error_when_particle_not_list(smc_step):
    with pytest.raises(TypeError):
        smc_step.particles = "Not a list of particles"


def test_type_error_not_list_of_particles(particle, smc_step):
    with pytest.raises(TypeError):
        smc_step.particles = [particle, "Not a particle"] * 2


def test_copy_step(smc_step):
    step_copy = smc_step.copy()
    assert step_copy is not smc_step
    assert isinstance(step_copy, SMCStep)


def test_get_likes(smc_step, particle):
    particle.log_like = 0
    smc_step.particles = [particle] * 3
    np.testing.assert_array_equal(smc_step.get_likes(), [1] * 3)


def test_get_log_likes(smc_step, particle):
    particle.log_like = 0
    smc_step.particles = [particle] * 3
    np.testing.assert_array_equal(smc_step.get_log_likes(), [0] * 3)


def test_get_mean(smc_step, particle, mocker):
    particle.params = {'a': 1, 'b': 2}
    smc_step.particles = [particle] * 3
    mocker.patch.object(smc_step, 'normalize_step_weights',
                        return_value=[1] * 3)
    assert smc_step.get_mean() == {'a': 3., 'b': 6.}


def test_get_variance(smc_step, particle, mocker):
    particle.params = {'a': 1, 'b': 2}
    smc_step.particles = [particle] * 3
    mocker.patch.object(smc_step, 'normalize_step_weights',
                        return_value = np.array([-2] *3))
    mocker.patch.object(smc_step, 'get_mean', return_value = {'a': 3., 'b': 6.})
    assert smc_step.get_variance() == {'a': 24 / 11, 'b': 96 / 11}


def test_get_std_dev(smc_step, particle, mocker):
    particle.params = {'a': 1, 'b': 2}
    smc_step.particles = [particle] * 3
    mocker.patch.object(smc_step, 'normalize_step_weights',
                        return_value = np.array([-2] *3))
    mocker.patch.object(smc_step, 'get_mean', return_value = {'a': 3., 'b': 6.})
    assert smc_step.get_std_dev() == {'a': np.sqrt(24 / 11),
                                      'b': np.sqrt(96 / 11)}


def test_get_log_weights(smc_step, particle):
    particle.log_weight = 1
    smc_step.particles = [particle] * 3
    assert np.array_equal(smc_step.get_log_weights(), [1] * 3)


def test_get_covariance(smc_step, particle, mocker):
    p1 = copy(particle)
    p1.params = {'a': 1.1, 'b': 2.2}
    p2 = copy(particle)
    p2.params = {'a': 1.0, 'b': 2.1}
    p3 = copy(particle)
    p3.params = {'a': 0.8, 'b': 1.9}

    smc_step.particles = [p1, p2, p3]
    mocker.patch.object(smc_step, 'normalize_step_weights',
                        return_value=np.array([0.1, 0.7, 0.2]))
    mocker.patch.object(smc_step, 'get_mean',
                        return_value={'a': 0.97, 'b': 2.06})

    scale = 1 / (1 - np.sum(np.array([0.1, 0.7, 0.2]) ** 2))
    expected_cov = np.array([[0.0081, 0.0081], [0.0081, 0.0082]]) * scale
    np.testing.assert_array_almost_equal(smc_step.get_covariance(),
                                         expected_cov)


@pytest.mark.filterwarnings('ignore: current step')
def test_covariance_not_positive_definite_is_eye(smc_step, particle, mocker):
    particle.params = {'a': 1.1, 'b': 2.2}
    smc_step.particles = [particle] * 3
    mocker.patch.object(smc_step, 'normalize_step_weights',
                        return_value=np.array([0.1, 0.7, 0.2]))
    mocker.patch.object(smc_step, 'get_mean', return_value={'a': 1, 'b': 2})
    mocker.patch.object(smc_step, '_is_positive_definite', return_value=False)
    np.testing.assert_array_equal(smc_step.get_covariance(), np.eye(2))


def test_normalize_step_log_weights(smc_step, particle, mocker):
    particle.log_weight = 20
    smc_step.particles = [particle]
    mocker.patch.object(smc_step, 'normalize_step_weights',
                        return_value=np.array([1.]))
    smc_step.normalize_step_log_weights()
    assert smc_step.particles[0].log_weight == 0


@pytest.mark.parametrize('log_weights', (np.array([1, 1, 2]),
                                         np.array([2, 4, 2])))
def test_normalize_step_weights(smc_step, log_weights, mocker):
    mocker.patch.object(smc_step, 'get_log_weights', return_value=log_weights)
    expected = np.exp(log_weights - 1) / np.sum(np.exp(log_weights - 1))
    norm_weights = smc_step.normalize_step_weights()
    np.testing.assert_array_almost_equal(norm_weights, expected)
    assert np.sum(norm_weights) == pytest.approx(1)
                            


def test_compute_ess(smc_step, mocker):
    mocker.patch.object(smc_step, 'normalize_step_log_weights')
    mocker.patch.object(smc_step, 'get_log_weights',
                        return_value=[np.log(2), np.log(5)])
    assert smc_step.compute_ess() == pytest.approx(1 / 29.)


def test_get_params(smc_step, particle):
    particle.params = {'a': 1}
    smc_step.particles = [particle] * 3
    np.testing.assert_array_equal(smc_step.get_params('a'), np.array([1] * 3))


def test_get_param_dicts(smc_step, particle):
    particle.params = {'a': 1, 'b': 2}
    smc_step.particles = [particle] * 3
    assert smc_step.get_param_dicts() == [{'a': 1, 'b': 2}] * 3


def test_resample(smc_step, particle, mocker):

    p1 = copy(particle)
    p1.params = {'a': 1}
    p2 = copy(particle)
    p2.params = {'a': 2}
    p3 = copy(particle)
    p3.params = {'a': 3}

    smc_step.particles = [p1, p2, p3]
    mocker.patch.object(smc_step, 'normalize_step_log_weights')
    mocker.patch.object(smc_step, 'get_log_weights',
                        return_value=np.log([.1, .5, .4]))
    mocker.patch('numpy.random.uniform', return_value=np.array([1, 0.6, 0.12]))
    smc_step.resample()

    particles = smc_step.particles

    assert all([particles[0].params == p3.params,
                particles[1].params == p3.params,
                particles[2].params == p2.params])
    assert sum([np.exp(p.log_weight) for p in particles]) == pytest.approx(1)


@pytest.mark.parametrize('ess_threshold,call_expected',[(0.1, False),
                         (0.49, False), (0.51, True), (0.9, True)])
def test_resample_if_needed(smc_step, ess_threshold, call_expected, mocker):
    mocker.patch.object(smc_step, 'compute_ess', return_value=0.5)
    mocker.patch.object(smc_step, 'resample')
    smc_step.resample_if_needed(ess_threshold)
    assert smc_step.resample.called is call_expected


def test_update_weights(smc_step, particle, mocker):
    p1 = copy(particle)
    p1.params = {'a': 1}
    p1.log_weight = 1
    p1.log_like = 0.1
    p2 = copy(particle)
    p2.params = {'a': 2}
    p2.log_weight = 1
    p2.log_like = 0.2
    smc_step.particles = [p1, p2]

    mocker.patch.object(smc_step, 'normalize_step_log_weights')

    smc_step.update_weights(delta_phi=0.1)

    smc_step.normalize_step_log_weights.assert_called()
    assert smc_step.particles[0].log_weight == 1.01
    assert smc_step.particles[1].log_weight == 1.02
