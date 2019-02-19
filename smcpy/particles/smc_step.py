import numpy as np
from smcpy.particles.particle import Particle


class SMCStep():

    def __init__(self,):
        self._particles = []

    @staticmethod
    def _check_input(params):
        if not isinstance(params, list):
            raise TypeError('Input must be a list')
        for param in params:
            if not isinstance(param, Particle):
                raise TypeError('Input must be a of the Particle class')
        return params

    def add_particle(self, particle):
        '''
        Add a single particle to a given step.
        '''
        self._particles.append(particle)

    def add_step(self, particle_list):
        '''
        Add an entire step to the chain, providing a list of particles.
        '''
        self._particles = particle_list
        return None

    def get_likes(self):
        return [np.exp(p.log_like) for p in self._particles]

    def get_log_likes(self):
        return [p.log_like for p in self._particles]

    def get_mean(self):
        param_names = self._particles[0].params.keys()
        mean = {}
        for pn in param_names:
            mean[pn] = []
            for p in self._particles:
                mean[pn].append(p.weight * p.params[pn])
            mean[pn] = np.sum(mean[pn])
        return mean

    def get_particles(self):
        return self._particles

    def get_weights(self):
        return [p.weight for p in self._particles]

    def calculate_covariance(self):
        particle_list = self._particles

        means = np.array(self.get_mean().values())

        cov_list = []
        for p in particle_list:
            param_vector = p.params.values()
            diff = (param_vector - means).reshape(-1, 1)
            R = np.dot(diff, diff.transpose())
            cov_list.append(p.weight * R)
        cov_matrix = np.sum(cov_list, axis=0)

        return cov_matrix

    def normalize_step_weights(self):
        weights = self.get_weights()
        particles = self.get_particles()
        total_weight = np.sum(weights)
        for p in particles:
            p.weight = p.weight / total_weight
        return None

    def compute_ess(self):
        weights = self.get_weights()
        if not np.isclose(np.sum(weights), 1):
            self.normalize_step_weights()
        return 1 / np.sum([w**2 for w in weights])

    def get_params(self, key):
        particles = self.get_particles()
        return np.array([p.params[key] for p in particles])

    def overwrite_step(self, particle_list):
        '''
        Overwrite an entire step of the chain with the provided list of
        particles.
        '''
        if len(particle_list) != len(self.get_particles()):
            raise ValueError('Number of new particles must equal number of old')
        self._particles = particle_list
        return None

# test = SMCStep()
# test.add_particle(5 * [Particle({'a': 1, 'b': 2}, 0.2, -0.2)])
# print test.get_log_likes()
# print(test.calculate_covariance())
# print(np.array([[0, 0], [0, 0]]))
# print(test.compute_ess())
