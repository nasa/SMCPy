from cStringIO import StringIO
import sys
from smcpy.particles.particle import Particle


class ParticleTester(object):
    '''
    Generates different instances of the Particle class. Intended to make
    reading tests easier.
    '''

    def __init__(self,):
        self.params = {'a': 1, 'b': 2}
        self.weight = 0.2
        self.log_like = -0.2

    def when_good_particle(self):
        return Particle(self.params, self.weight, self.log_like)

    def when_wrong_param_type(self):
        bad_param_type = [1, 2]
        return Particle(bad_param_type, self.weight, self.log_like)

    def when_pos_log_like(self):
        pos_log_like = 0.1
        return Particle(self.params, self.weight, pos_log_like)

    def when_zero_log_like(self):
        edge_case_log_like = 0.0
        return Particle(self.params, self.weight, edge_case_log_like)

    def when_wrong_log_like_type(self):
        bad_log_like_type = 'bad'
        return Particle(self.params, self.weight, bad_log_like_type)

    def when_neg_weight(self):
        neg_weight = -0.1
        return Particle(self.params, neg_weight, self.log_like)

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
