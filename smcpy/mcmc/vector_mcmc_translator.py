from copy import copy
import numpy as np

from .translator_base import Translator

class VectorMCMCTranslator(Translator):

    def __init__(self, vector_mcmc_object, param_order):
        self._mcmc = vector_mcmc_object
        self._param_order = param_order

    def mutate_particles(self, particles, num_samples, proposal_cov, phi):
        particles = copy(particles)

        input_array = np.zeros([len(particles), len(self._param_order)])
        for i, p in enumerate(particles):
            input_array[i, :] = [p.params[key] for key in self._param_order]

        params, log_likes = self._mcmc.smc_metropolis(input_array, num_samples,
                                                      proposal_cov, phi)

        for i, p in enumerate(particles):
            p.params = {key: params[i][j] \
                        for j, key in enumerate(self._param_order)}
            p.log_like = log_likes[i, 0]

        return particles

    def sample_from_prior(self, num_samples):
        samples = self._mcmc.sample_from_priors(num_samples)
        return dict(zip(self._param_order, samples.T))

    def get_log_likelihood(self, param_dict):
        input_array = self._translate_param_dict_to_input_array(param_dict)
        return self._mcmc.evaluate_log_likelihood(input_array)

    def get_log_prior(self, param_dict):
        input_array = self._translate_param_dict_to_input_array(param_dict)
        log_priors = self._mcmc.evaluate_log_priors(input_array)
        return np.sum(log_priors, axis=1).reshape(-1, 1)

    def _translate_param_dict_to_input_array(self, param_dict):
        return np.array([[param_dict[k] for k in self._param_order]])
