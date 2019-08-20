import pytest
import numpy as np

arr_alm_eq = np.testing.assert_array_almost_equal


def test_add_particle(filled_step, particle):
    orig_num_particles = len(filled_step.particles)
    filled_step.add_particle(particle)
    assert len(filled_step.particles) == orig_num_particles + 1


def test_copy_step(filled_step):
    filled_step_copy = filled_step.copy()
    filled_step_copy.particles == []
    assert filled_step.particles != []


def test_type_error_when_particle_not_list(step_tester):
    with pytest.raises(TypeError):
        step_tester.set_particles("Bad param type")


def test_type_error_not_particle_class(step_tester):
    with pytest.raises(TypeError):
        step_tester.set_particles([1, 2, 3])


def test_private_variable_creation(step_tester, particle_list):
    step_tester.set_particles(particle_list)
    assert step_tester.particles == particle_list


def test_get_likes(filled_step):
    assert np.array_equal(filled_step.get_likes(),
                          [pytest.approx(0.818730753078)] * 5)


def test_get_log_likes(filled_step):
    assert np.array_equal(filled_step.get_log_likes(), [-0.2] * 5)


def test_get_mean(filled_step):
    assert filled_step.get_mean()['a'] == 1.0


def test_get_std_dev(mixed_step):
    assert mixed_step.get_std_dev()['a'] == np.sqrt(0.3)


def test_get_variance(mixed_step):
    assert mixed_step.get_variance()['a'] == 0.3


def test_get_log_weights(filled_step):
    assert np.array_equal(filled_step.get_log_weights(), [0.2] * 5)


def test_get_covariance_not_positive_definite(filled_step):
    assert np.array_equal(filled_step.get_covariance(), np.eye(2))


def test_get_covariance(linear_step):
    a = linear_step.get_params('a')
    b = linear_step.get_params('b')
    exp_cov = np.cov(a, b)
    arr_alm_eq(linear_step.get_covariance(), exp_cov)


def test_normalize_step_log_weights(mixed_weight_step):
    mixed_weight_step.normalize_step_log_weights()
    for index, p in enumerate(mixed_weight_step.particles):
        p.log_weight = np.exp(p.log_weight)
    assert sum(mixed_weight_step.get_log_weights()) == 1


def test_normalize_step_weights(mixed_weight_step):
    normalized_weights = mixed_weight_step.normalize_step_weights()
    assert sum(normalized_weights) == 1


def test_compute_ess(filled_step):
    assert filled_step.compute_ess() == pytest.approx(5.0)


def test_get_params(filled_step):
    assert np.array_equal(filled_step.get_params('a'), np.array(5 * [1]))


def test_get_param_dicts(filled_step):
    assert filled_step.get_param_dicts() == 5 * [{'a': 1, 'b': 2}]


def test_resample(mixed_step):
    np.random.seed(1)
    prior_particle = mixed_step.particles
    mixed_step.resample()
    assert mixed_step.particles != prior_particle


def test_resample_uniform(mixed_step):
    mixed_step.resample()
    log_weights = mixed_step.get_log_weights()
    np.testing.assert_almost_equal(max(log_weights) - min(log_weights), 0)


def test_print_particle_info(filled_step, capfd):
    filled_step.print_particle_info(3)
    out, err = capfd.readouterr()
    assert "params = {'a': 1, 'b': 2}" in out
