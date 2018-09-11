import copy
import numpy as np

class Particle():
    '''
    Class defining data structure of an SMC particle (a member of an SMC
    particle chain).
    '''

    def __init__(self, params, weight, log_like):
        '''
        :param params: parameters associated with particle; keys = parameter
            name and values = parameter value.
        :type params: dictionary
        :param weight: the computed weight of the particle
        :type weight: float
        :param log_like: the log likelihood of the particle
        :type log_like: float
        '''
        self.params = params
        self.weight = weight
        self.log_like = log_like


    def print_particle_info(self):
        print 'params = %s' % self.params
        print 'weight = %s' % self.weight
        print 'log_like = %s' % self.log_like


    def copy(self):
        return copy.deepcopy(self)
