import pytest
import numpy as np
from smcpy.particles.smc_step import SMCStep
from smcpy.particles.particle import Particle


@pytest.fixture
def particle_list():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return 5 * [particle]


@pytest.fixture
def mixed_particle_list():
    particle_1 = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    particle_2 = Particle({'a': 2, 'b': 3}, 0.1, -0.1)
    return 3 * [particle_1] + 2 * [particle_2]


@pytest.fixture
def step_tester():
    return SMCStep()


@pytest.fixture
def filled_step(step_tester, particle_list):
    step_tester.fill_step(particle_list)
    return step_tester


@pytest.fixture
def mixed_step(step_tester, mixed_particle_list):
    step_tester.fill_step(mixed_particle_list)
    return step_tester


def test_type_error_when_particle_not_list(step_tester):
    with pytest.raises(TypeError):
        step_tester.fill_step("Bad param type")


def test_type_error_not_particle_class(step_tester):
    with pytest.raises(TypeError):
        step_tester.fill_step([1, 2, 3])


def test_private_variable_creation(step_tester, particle_list):
    step_tester.fill_step(particle_list)
    assert step_tester.particles == particle_list


def test_get_likes(filled_step):
    assert np.array_equal(filled_step.get_likes(), [pytest.approx(0.818730753078)] * 5)


def test_get_log_likes(filled_step):
    assert filled_step.get_log_likes()[0] == -0.2


def test_get_mean(filled_step):
    assert filled_step.get_mean()['a'] == 1.0


def test_get_weights(filled_step):
    assert np.array_equal(filled_step.get_weights(), [0.2] * 5)


def test_calcuate_covariance(filled_step):
    filled_step.calculate_covariance() == np.array([[0, 0], [0, 0]])


def test_compute_ess(filled_step):
    assert filled_step.compute_ess() == pytest.approx(5.0)


def test_get_params(filled_step):
    assert np.array_equal(filled_step.get_params('a'), np.array(5 * [1]))


def test_get_param_dicts(filled_step):
    assert filled_step.get_param_dicts() == 5 * [{'a': 1, 'b': 2}]


def test_resample(mixed_step):
    prior_particle = mixed_step.particles
    mixed_step.resample()
    assert mixed_step.particles != prior_particle


def test_resample_uniform(mixed_step):
    mixed_step.resample()
    weights = mixed_step.get_weights()
    assert np.testing.assert_almost_equal(max(weights) - min(weights), 0)


def test_print_particle_info(filled_step, capfd):
    filled_step.print_particle_info(3)
    out, err = capfd.readouterr()
    assert "params = {'a': 1, 'b': 2}" in out
