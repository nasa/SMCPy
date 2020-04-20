import pytest
import sys

if sys.version_info.major > 2:
    from io import StringIO
else:
    from cStringIO import StringIO
from smcpy.particles.particle import Particle


class ParticleTester(object):
    '''
    Generates different instances of the Particle class. Intended to make
    reading tests easier.
    '''

    def __init__(self,):
        self.params = {'a': 1, 'b': 2}
        self.log_weight = 0.2
        self.log_like = -0.2

    def when_good_particle(self):
        return Particle(self.params, self.log_weight, self.log_like)

    def when_wrong_param_type(self):
        bad_param_type = [1, 2]
        return Particle(bad_param_type, self.log_weight, self.log_like)

    def when_zero_log_like(self):
        edge_case_log_like = 0.0
        return Particle(self.params, self.log_weight, edge_case_log_like)

    def when_wrong_log_like_type(self):
        bad_log_like_type = 'bad'
        return Particle(self.params, self.log_weight, bad_log_like_type)

    def when_wrong_weight_type(self):
        bad_weight_type = 'bad'
        return Particle(self.params, bad_weight_type, self.log_like)

    @staticmethod
    def hijack_stdout():
        sys.stdout = forked_stdout = StringIO()
        return forked_stdout

    @staticmethod
    def restore_stdout():
        sys.stdout = sys.__stdout__
        return None


@pytest.fixture
def particle_tester():
    return ParticleTester()


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
