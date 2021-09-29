from copy import copy
import numpy as np

from .kernel_base import MCMCKernel

class VectorMCMCKernel(MCMCKernel):

    def __init__(self, vector_mcmc_object, param_order):
        self._mcmc = vector_mcmc_object
        self._param_order = param_order

    def mutate_particles(self, param_dict, log_likes, num_samples, cov, phi):
        param_array = self.conv_param_dict_to_array(param_dict)
        param_array, log_likes = self._mcmc.smc_metropolis(param_array,
                                                           num_samples,
                                                           cov, phi)
        param_dict = self.conv_param_array_to_dict(param_array)
        return param_dict, log_likes

    def get_log_likelihoods(self, param_dict):
        param_array = self.conv_param_dict_to_array(param_dict)
        return self._mcmc.evaluate_log_likelihood(param_array)

    def get_log_priors(self, param_dict):
        param_array = self.conv_param_dict_to_array(param_dict)
        log_priors = self._mcmc.evaluate_log_priors(param_array)
        return np.sum(log_priors, axis=1).reshape(-1, 1)

    def conv_param_array_to_dict(self, params):
        return dict(zip(self._param_order, np.transpose(params, (2, 0, 1))))

    def conv_param_dict_to_array(self, param_dict):
        array_shape = param_dict[self._param_order[0]].shape
        array_shape = (array_shape[0], array_shape[1], len(self._param_order))
        array = np.empty(array_shape)

        for i, name in enumerate(self._param_order):
            array[:, :, i] = param_dict[name]

        return array
