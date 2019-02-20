import pytest
import numpy as np
from smcpy.particles.smc_step import SMCStep
from smcpy.particles.particle import Particle


@pytest.fixture
def particle_list():
    particle = Particle({'a': 1, 'b': 2}, 0.2, -0.2)
    return 5 * [particle]


@pytest.fixture
def step_tester():
    return SMCStep()


@pytest.fixture
def filled_step(step_tester, particle_list):
    step_tester.fill_step(particle_list)
    return step_tester


def test_type_error_when_particle_not_list(step_tester):
    with pytest.raises(TypeError):
        step_tester.fill_step("Bad param type")


def test_type_error_not_particle_class(step_tester):
    with pytest.raises(TypeError):
        step_tester.fill_step([1, 2, 3])


def test_private_variable_creation(step_tester, particle_list):
    step_tester.fill_step(particle_list)
    assert step_tester._particles == particle_list


def test_get_likes(filled_step):
    assert filled_step.get_likes()[0] == pytest.approx(0.818730753078)


def test_get_log_likes(filled_step):
    assert filled_step.get_log_likes()[0] == -0.2


def test_get_mean(filled_step):
    assert filled_step.get_mean()['a'] == 1.0


def test_get_weights(filled_step):
    assert filled_step.get_weights()[0] == 0.2


def test_calcuate_covariance(filled_step):
    filled_step.calculate_covariance() == np.array([[0, 0], [0, 0]])


def test_compute_ess(filled_step):
    assert filled_step.compute_ess() == pytest.approx(5.0)


def test_get_params(filled_step):
    assert np.array_equal(filled_step.get_params('a'), np.array(5 * [1]))


def test_get_param_dicts(filled_step):
    assert filled_step.get_param_dicts() == 5 * [{'a': 1, 'b': 2}]


def test_resample(filled_step):
    prior_particle = filled_step._particles
    filled_step.resample(overwrite=True)
    assert filled_step._particles != prior_particle


def test_print_particle_info(filled_step, capfd):
    filled_step.print_particle_info(3)
    out, err = capfd.readouterr()
    assert "params = {'a': 1, 'b': 2}" in out
