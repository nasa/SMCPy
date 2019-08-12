import pytest


def test_particle_instance_variables(particle_tester):
    good_particle = particle_tester.when_good_particle()
    assert good_particle.params == particle_tester.params
    assert good_particle.log_weight == particle_tester.log_weight
    assert good_particle.log_like == particle_tester.log_like


def test_type_error_when_param_not_dict(particle_tester):
    with pytest.raises(TypeError):
        particle_tester.when_wrong_param_type()


def test_type_error_when_weight_not_numeric(particle_tester):
    with pytest.raises(TypeError):
        particle_tester.when_wrong_weight_type()


def test_type_error_when_log_like_not_numeric(particle_tester):
    with pytest.raises(TypeError):
        particle_tester.when_wrong_log_like_type()


def test_no_error_zero_log_like(particle_tester):
    particle = particle_tester.when_zero_log_like()
    assert particle.log_like == 0


def test_value_error_pos_log_like(particle_tester):
    with pytest.raises(ValueError):
        particle_tester.when_pos_log_like()


def test_copy(particle_tester):
    p = particle_tester.when_good_particle()
    z = p.copy()
    assert p is not z
    assert z.params == p.params
    assert z.log_weight == z.log_weight
    assert z.log_like == p.log_like


def test_print_particle_info(particle_tester):
    p = particle_tester.when_good_particle()
    particle_info = (p.params, p.log_weight, p.log_like)
    expected = 'params = %s\nlog_weight = %s\nlog_like = %s\n' % particle_info

    stdout = particle_tester.hijack_stdout()
    p.print_particle_info()
    particle_tester.restore_stdout()
    received = stdout.getvalue()

    assert expected == received
